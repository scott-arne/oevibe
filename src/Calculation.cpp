#include "oevibe/Calculation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

#include "core/ConstraintMath.h"
#include "core/InputPrep.h"
#include "core/Optimizer.h"
#include "core/Sha256.h"
#include "core/WriteBack.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

/// The engine schema validates ``accuracy`` as a number only when it declares the keyword as
/// DOUBLE, so a schema that declares another type can still hand over text that is not one.
double parse_accuracy(const std::string& text) {
    double value = 0.0;
    bool parsed = false;
    try {
        std::size_t used = 0;
        value = std::stod(text, &used);
        parsed = used == text.size();
    } catch (const std::exception&) {
        parsed = false;
    }
    if (!parsed || !std::isfinite(value)) {
        throw OptionsError("keyword 'accuracy' expects a finite number, got '" + text + "'");
    }
    return value;
}

std::string partial_charge_policy_name(PartialChargePolicy policy) {
    switch (policy) {
        case PartialChargePolicy::ASSIGN_AM1BCC_ELF10:
            return "am1bcc-elf10";
        case PartialChargePolicy::USE_INPUT:
            return "input";
        case PartialChargePolicy::NONE:
        default:
            return "none";
    }
}

std::string constraint_kind_name(ConstraintKind kind) {
    switch (kind) {
        case ConstraintKind::ANGLE:
            return "angle";
        case ConstraintKind::DIHEDRAL:
            return "dihedral";
        case ConstraintKind::DISTANCE:
        default:
            return "distance";
    }
}

std::string restraint_kind_name(RestraintKind kind) {
    switch (kind) {
        case RestraintKind::ANGLE:
            return "angle";
        case RestraintKind::DIHEDRAL:
            return "dihedral";
        case RestraintKind::POSITION:
            return "position";
        case RestraintKind::DISTANCE:
        default:
            return "distance";
    }
}

// The provenance encodings of a ConstraintSet, so that sets that can give different results
// never share an options digest. Terms are joined with ';', a term's fields with ':', and atom
// indices with ','; none of these occur in an index, a kind name, a format_value number, or
// "start", which marks a target taken from the starting geometry. Constraint and restraint atoms
// keep the caller's order, frozen atoms are sorted, and an empty list encodes as "".
std::string encode_indices(const std::vector<unsigned>& indices) {
    std::string text;
    for (std::size_t k = 0; k < indices.size(); ++k) {
        text += (k == 0 ? "" : ",") + std::to_string(indices[k]);
    }
    return text;
}

std::string encode_target(bool has_target, double target) {
    return has_target ? detail::format_value(target) : "start";
}

std::string encode_constraints(const std::vector<Constraint>& constraints) {
    std::string text;
    for (std::size_t k = 0; k < constraints.size(); ++k) {
        const Constraint& term = constraints[k];
        text += (k == 0 ? "" : ";") + constraint_kind_name(term.kind) + ":" +
                encode_indices(term.atoms) + ":" + encode_target(term.has_target, term.target);
    }
    return text;
}

std::string encode_restraints(const std::vector<Restraint>& restraints) {
    std::string text;
    for (std::size_t k = 0; k < restraints.size(); ++k) {
        const Restraint& term = restraints[k];
        text += (k == 0 ? "" : ";") + restraint_kind_name(term.kind) + ":" +
                encode_indices(term.atoms) + ":" + detail::format_value(term.force_constant) + ":" +
                encode_target(term.has_target, term.target);
    }
    return text;
}

Solvent resolve_solvent(const Solvent& requested, const MethodCapabilities& caps) {
    if (requested.IsGasPhase()) {
        return requested;
    }
    const std::string& method = caps.method.GetName();
    if (caps.solvent_models.empty()) {
        throw OptionsError("method '" + method + "' supports gas phase only; solvent '" +
                           requested.GetName() + "' cannot be modeled");
    }
    SolventModel model = requested.GetModel();
    if (model == SolventModel::AUTO) {
        model = caps.solvent_models.front();
    } else if (std::find(caps.solvent_models.begin(), caps.solvent_models.end(), model) ==
               caps.solvent_models.end()) {
        throw OptionsError("method '" + method + "' does not support solvent model '" +
                           solvent_model_name(model) + "'");
    }
    if (!caps.solvents.empty() && std::find(caps.solvents.begin(), caps.solvents.end(),
                                            requested.GetName()) == caps.solvents.end()) {
        throw OptionsError("method '" + method + "' does not support solvent '" +
                           requested.GetName() + "'");
    }
    return requested.WithModel(model);
}

void apply_evaluation(ConformerResult& result, const Evaluation& evaluation,
                      const EvaluationRequest& request, const MolecularSystem& system) {
    result.SetEnergy(evaluation.energy);
    if (request.gradient) {
        result.SetGradient(system.ToOELayout(evaluation.gradient));
    }
    if ((request.properties & static_cast<unsigned>(Property::CHARGES)) != 0) {
        result.SetCharges(system.ToOEScalarLayout(evaluation.charges));
    }
    if ((request.properties & static_cast<unsigned>(Property::DIPOLE)) != 0) {
        result.SetDipole(evaluation.dipole);
    }
    if ((request.properties & static_cast<unsigned>(Property::ORBITAL_ENERGIES)) != 0) {
        result.SetOrbitalEnergies(evaluation.orbital_energies, evaluation.homo_index);
    }
    if ((request.properties & static_cast<unsigned>(Property::BOND_ORDERS)) != 0) {
        std::vector<BondOrder> orders;
        for (const BondOrder& order : evaluation.bond_orders) {
            orders.push_back(
                {system.GetOEAtomIndex(order.i), system.GetOEAtomIndex(order.j), order.order});
        }
        result.SetBondOrders(orders);
    }
}

}  // namespace

struct Calculation::CallState {
    detail::PreparedInput input;
    std::unique_ptr<MolecularSystem> system;
};

Calculation::Calculation(const CalculationOptions& options) : options_(options) {
    options_.Validate();
    EngineRegistry& registry = EngineRegistry::Instance();
    if (options_.HasEngine()) {
        engine_ = registry.Find(options_.GetEngine());
        if (!engine_) {
            throw EngineUnavailableError("no engine named '" + options_.GetEngine() +
                                         "' is registered");
        }
    } else {
        engine_ = registry.ResolveForMethod(options_.GetMethod());
    }
    engine_name_ = engine_->GetName();
    if (!engine_->IsAvailable()) {
        throw EngineUnavailableError("engine '" + engine_name_ +
                                     "' is unavailable: " + engine_->GetUnavailableReason());
    }
    const std::optional<MethodCapabilities> caps = find_method(*engine_, options_.GetMethod());
    if (!caps.has_value()) {
        throw OptionsError("engine '" + engine_name_ + "' does not provide method '" +
                           options_.GetMethod().GetName() + "'");
    }
    if (!caps->available) {
        throw EngineUnavailableError("method '" + options_.GetMethod().GetName() + "' of engine '" +
                                     engine_name_ +
                                     "' is unavailable: " + caps->unavailable_reason);
    }
    engine_version_ = engine_->GetVersion();
    capabilities_ = *caps;
    settings_.method = options_.GetMethod();
    settings_.solvent = resolve_solvent(options_.GetSolvent(), capabilities_);
    settings_.preset = options_.GetPreset();
    settings_.accuracy = preset_values(settings_.preset).engine_accuracy;
    settings_.properties = options_.GetEffectiveProperties();
    settings_.keywords = options_.GetKeywords();
    const unsigned unsupported = settings_.properties & ~capabilities_.properties;
    if (unsupported != 0) {
        throw OptionsError("method '" + options_.GetMethod().GetName() +
                           "' does not provide: " + property_names(unsupported));
    }
    engine_->ValidateSettings(settings_);
    // Applied after validation so the keyword's type and bounds are checked first.
    if (options_.HasKeyword("accuracy")) {
        settings_.accuracy = parse_accuracy(options_.GetKeyword("accuracy"));
    }
}

ResultSet Calculation::Energy(OEChem::OEMolBase& mol) { return RunSinglePoints(mol, false); }
ResultSet Calculation::Gradient(OEChem::OEMolBase& mol) { return RunSinglePoints(mol, true); }
const CalculationOptions& Calculation::GetOptions() const { return options_; }
const std::string& Calculation::GetEngineName() const { return engine_name_; }

Calculation::CallState Calculation::Prepare(OEChem::OEMolBase& mol) const {
    CallState state;
    state.input = detail::prepare_input(mol, options_);
    state.system = std::make_unique<MolecularSystem>(mol, state.input.charge,
                                                     state.input.multiplicity, ChargePolicy());
    engine_->ValidateSystem(*state.system, settings_);
    return state;
}

PartialChargePolicy Calculation::ChargePolicy() const {
    if (!capabilities_.requires_partial_charges) {
        return PartialChargePolicy::NONE;
    }
    return options_.GetUseInputPartialCharges() ? PartialChargePolicy::USE_INPUT
                                                : PartialChargePolicy::ASSIGN_AM1BCC_ELF10;
}

std::map<std::string, std::string> Calculation::BaseResolvedOptions() const {
    // The accuracy is read from the settings, not the keyword, so that provenance records the
    // value the engine actually received.
    std::map<std::string, std::string> resolved = {
        {"method", settings_.method.GetName()},
        {"engine", engine_name_},
        {"solvent", settings_.solvent.GetTag()},
        {"preset", preset_name(settings_.preset)},
        {"accuracy", detail::format_value(settings_.accuracy)},
        {"properties", std::to_string(settings_.properties)},
    };
    for (const auto& [name, value] : settings_.keywords) {
        resolved["keyword." + name] = value;
    }
    return resolved;
}

std::map<std::string, std::string> Calculation::CallResolvedOptions(const CallState& state,
                                                                    const std::string& task) const {
    std::map<std::string, std::string> resolved = BaseResolvedOptions();
    resolved["charge"] = std::to_string(state.input.charge);
    resolved["multiplicity"] = std::to_string(state.input.multiplicity);
    resolved["partial_charges"] = partial_charge_policy_name(ChargePolicy());
    resolved["task"] = task;
    return resolved;
}

ConformerResult Calculation::NewResult(const CallState& state, unsigned position,
                                       const std::map<std::string, std::string>& resolved) const {
    ConformerResult result;
    result.SetConformerIndex(state.system->GetConformerIndices()[position]);
    result.SetAtomIndices(state.system->GetOEAtomIndices());
    for (const std::string& warning : state.input.warnings) {
        result.AddWarning(warning);
    }
    Provenance provenance;
    provenance.method = settings_.method.GetName();
    provenance.engine = engine_name_;
    provenance.engine_version = engine_version_;
    provenance.solvent_model = settings_.solvent.IsGasPhase()
                                   ? std::string("gas")
                                   : solvent_model_name(settings_.solvent.GetModel());
    provenance.solvent = settings_.solvent.GetName();
    provenance.resolved_options = resolved;
    provenance.options_digest = detail::options_digest(resolved);
    result.SetProvenance(provenance);
    return result;
}

ResultSet Calculation::RunSinglePoints(OEChem::OEMolBase& mol, bool gradient) {
    const Clock::time_point call_start = Clock::now();
    CallState state = Prepare(mol);
    const double setup_seconds = seconds_since(call_start);
    const MolecularSystem& system = *state.system;
    const std::map<std::string, std::string> resolved =
        CallResolvedOptions(state, gradient ? "gradient" : "energy");
    EvaluationRequest request;
    request.gradient = gradient;
    request.properties = settings_.properties;

    ResultSet results;
    for (unsigned position = 0; position < system.NumConformers(); ++position) {
        ConformerResult result = NewResult(state, position, resolved);
        const Clock::time_point start = Clock::now();
        try {
            std::shared_ptr<EngineSession> session = engine_->CreateSession(system, settings_);
            const Evaluation evaluation =
                session->Evaluate(system.GetConformerCoordinates(position), request);
            const std::string problem = validate_evaluation(evaluation, request, system.NumAtoms());
            if (problem.empty()) {
                apply_evaluation(result, evaluation, request, system);
                result.SetStatus(ConformerStatus::OK, "");
            } else {
                // Spec 7.5: a failed conformer keeps the values computed before the failure.
                // Only the energy is trusted; the rest of a malformed evaluation is dropped.
                if (std::isfinite(evaluation.energy)) {
                    result.SetEnergy(evaluation.energy);
                }
                result.SetStatus(ConformerStatus::ENGINE_ERROR,
                                 "engine returned a malformed Evaluation: " + problem);
            }
        } catch (const UnsupportedError& e) {
            result.SetStatus(ConformerStatus::UNSUPPORTED, e.what());
        } catch (const std::exception& e) {
            result.SetStatus(ConformerStatus::ENGINE_ERROR, e.what());
        }
        Timings timings;
        timings.setup = setup_seconds;
        timings.compute = seconds_since(start);
        timings.total = seconds_since(call_start);
        result.SetTimings(timings);
        results.Add(std::move(result));
    }
    results.Finalize();
    detail::write_back(mol, system, results, options_, settings_.solvent.GetTag(), false);
    return results;
}

ResultSet Calculation::Optimize(OEChem::OEMolBase& mol, const OptimizerOptions& options) {
    const Clock::time_point call_start = Clock::now();
    options.Validate();
    CallState state = Prepare(mol);
    const MolecularSystem& system = *state.system;
    detail::check_constraint_atoms(options.GetConstraints(), system);
    const ResolvedOptimizerSettings resolved_settings = resolve_optimizer_settings(
        options, settings_.preset, system.NumAtoms(), capabilities_.force_field);
    const double setup_seconds = seconds_since(call_start);

    std::map<std::string, std::string> resolved = CallResolvedOptions(state, "optimize");
    const ConstraintSet& constraints = options.GetConstraints();
    resolved["optimizer.algorithm"] = algorithm_name(resolved_settings.algorithm);
    resolved["optimizer.max_iterations"] = std::to_string(resolved_settings.max_iterations);
    resolved["optimizer.energy_change"] = detail::format_value(resolved_settings.energy_change);
    resolved["optimizer.gradient_norm"] = detail::format_value(resolved_settings.gradient_norm);
    resolved["optimizer.initial_trust_radius"] =
        detail::format_value(resolved_settings.initial_trust_radius);
    resolved["optimizer.max_trust_radius"] =
        detail::format_value(resolved_settings.max_trust_radius);
    resolved["optimizer.frozen_atoms"] = encode_indices(constraints.GetFrozenAtoms());
    resolved["optimizer.constraints"] = encode_constraints(constraints.GetConstraints());
    resolved["optimizer.restraints"] = encode_restraints(constraints.GetRestraints());

    ResultSet results;
    for (unsigned position = 0; position < system.NumConformers(); ++position) {
        ConformerResult result = NewResult(state, position, resolved);
        const Clock::time_point start = Clock::now();
        try {
            std::shared_ptr<EngineSession> session = engine_->CreateSession(system, settings_);
            const std::vector<double>& x0 = system.GetConformerCoordinates(position);
            detail::OptimizationProblem problem;
            problem.x0 = x0;
            problem.atomic_numbers = system.GetAtomicNumbers();
            problem.constraints = detail::resolve_constraint_set(constraints, system, x0);
            problem.settings = resolved_settings;
            const detail::OptimizationOutcome outcome = detail::run_optimization(*session, problem);
            result.SetEnergy(outcome.engine_energy);
            result.SetCoordinates(system.ToOELayout(outcome.x));
            result.SetOptimizationStats(outcome.stats);
            bool properties_ok = true;
            if (settings_.properties != 0) {
                EvaluationRequest request;
                request.properties = settings_.properties;
                const Evaluation evaluation = session->Evaluate(outcome.x, request);
                const std::string problem_text =
                    validate_evaluation(evaluation, request, system.NumAtoms());
                if (problem_text.empty()) {
                    apply_evaluation(result, evaluation, request, system);
                } else {
                    properties_ok = false;
                    result.SetStatus(ConformerStatus::ENGINE_ERROR,
                                     "engine returned a malformed Evaluation: " + problem_text);
                }
            }
            if (properties_ok) {
                result.SetStatus(
                    outcome.converged ? ConformerStatus::OK : ConformerStatus::NOT_CONVERGED,
                    outcome.converged ? "" : outcome.message);
            }
        } catch (const UnsupportedError& e) {
            result.SetStatus(ConformerStatus::UNSUPPORTED, e.what());
        } catch (const std::exception& e) {
            result.SetStatus(ConformerStatus::ENGINE_ERROR, e.what());
        }
        Timings timings;
        timings.setup = setup_seconds;
        timings.compute = seconds_since(start);
        timings.total = seconds_since(call_start);
        result.SetTimings(timings);
        results.Add(std::move(result));
    }
    results.Finalize();
    detail::write_back(mol, system, results, options_, settings_.solvent.GetTag(), true);
    return results;
}

}  // namespace OEVIBE
