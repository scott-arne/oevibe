#include "oevibe/Calculation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "core/ConstraintMath.h"
#include "core/Frequencies.h"
#include "core/Geometry.h"
#include "core/Hessian.h"
#include "core/InputPrep.h"
#include "core/Optimizer.h"
#include "core/ScanGrid.h"
#include "core/Sha256.h"
#include "core/Thermochemistry.h"
#include "core/Wavefront.h"
#include "core/WriteBack.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"

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

/// The engine contract forbids a null session, but dereferencing one would crash the whole call
/// instead of failing one conformer (spec 7.5), so it is reported like any engine failure.
/// :raises std::runtime_error: When the engine returns a null session.
std::shared_ptr<EngineSession> create_session(Engine& engine, const std::string& engine_name,
                                              const MolecularSystem& system,
                                              const EngineSettings& settings) {
    std::shared_ptr<EngineSession> session = engine.CreateSession(system, settings);
    if (!session) {
        throw std::runtime_error("engine '" + engine_name + "' returned a null session");
    }
    return session;
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
    // Spec 10 records solvent model resolution per result, so the model AUTO chose is visible
    // without reading the provenance.
    if (!settings_.solvent.IsGasPhase() && options_.GetSolvent().GetModel() == SolventModel::AUTO) {
        solvent_warning_ = "solvent model AUTO resolved to '" +
                           solvent_model_name(settings_.solvent.GetModel()) + "' for solvent '" +
                           settings_.solvent.GetName() + "', the default of method '" +
                           options_.GetMethod().GetName() + "'";
    }
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
    if (!solvent_warning_.empty()) {
        result.AddWarning(solvent_warning_);
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

ResultSet Calculation::RunConformers(OEChem::OEMolBase& mol, const CallState& state,
                                     const std::map<std::string, std::string>& resolved,
                                     Clock::time_point call_start, double setup_seconds,
                                     bool write_coordinates, const ConformerBody& body) {
    const MolecularSystem& system = *state.system;
    ResultSet results;
    for (unsigned position = 0; position < system.NumConformers(); ++position) {
        ConformerResult result = NewResult(state, position, resolved);
        // Spec 7.5: an UNSUPPORTED conformer keeps no values, whatever the body set before the
        // engine reported the system unsupported, so that case restores this copy.
        const ConformerResult initial = result;
        const Clock::time_point start = Clock::now();
        try {
            body(result, position);
        } catch (const UnsupportedError& e) {
            result = initial;
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
    detail::write_back(mol, system, results, options_, settings_.solvent.GetTag(),
                       write_coordinates);
    return results;
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

    return RunConformers(
        mol, state, resolved, call_start, setup_seconds, false,
        [&](ConformerResult& result, unsigned position) {
            std::shared_ptr<EngineSession> session =
                create_session(*engine_, engine_name_, system, settings_);
            const Evaluation evaluation =
                session->Evaluate(system.GetConformerCoordinates(position), request);
            const std::string problem = validate_evaluation(evaluation, request, system.NumAtoms());
            if (!problem.empty()) {
                // Spec 7.5: a failed conformer keeps the values computed before the failure.
                // Only the energy is trusted; the rest of a malformed evaluation is dropped.
                if (std::isfinite(evaluation.energy)) {
                    result.SetEnergy(evaluation.energy);
                }
                throw std::runtime_error("engine returned a malformed Evaluation: " + problem);
            }
            apply_evaluation(result, evaluation, request, system);
            result.SetStatus(ConformerStatus::OK, "");
        });
}

ResultSet Calculation::Optimize(OEChem::OEMolBase& mol, const OptimizerOptions& options) {
    const Clock::time_point call_start = Clock::now();
    // The optimizer checks run before Prepare, which may add hydrogens to the caller's molecule.
    // Resolving validates the options; its atom count only sets the default iteration limit, so
    // the settings are resolved again for the prepared system below. Adding hydrogens appends
    // atoms without renumbering, so every atom referenced here stays valid.
    resolve_optimizer_settings(options, settings_.preset, mol.NumAtoms(),
                               capabilities_.force_field);
    detail::check_constraint_atoms(options.GetConstraints(), mol);
    CallState state = Prepare(mol);
    const MolecularSystem& system = *state.system;
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

    return RunConformers(
        mol, state, resolved, call_start, setup_seconds, true,
        [&](ConformerResult& result, unsigned position) {
            std::shared_ptr<EngineSession> session =
                create_session(*engine_, engine_name_, system, settings_);
            const std::vector<double>& x0 = system.GetConformerCoordinates(position);
            detail::OptimizationProblem problem;
            problem.x0 = x0;
            problem.atomic_numbers = system.GetAtomicNumbers();
            problem.constraints = detail::resolve_constraint_set(constraints, system, x0);
            problem.settings = resolved_settings;
            detail::OptimizationOutcome outcome;
            try {
                outcome = detail::run_optimization(*session, problem);
            } catch (const detail::OptimizationFailure& e) {
                // Spec 7.5: an engine failure keeps the values computed before it. Write-back
                // skips ENGINE_ERROR conformers, so the molecule keeps its input coordinates.
                if (e.HasPartialOutcome()) {
                    const detail::OptimizationOutcome& partial = e.GetPartialOutcome();
                    result.SetEnergy(partial.engine_energy);
                    result.SetCoordinates(system.ToOELayout(partial.x));
                    result.SetOptimizationStats(partial.stats);
                } else if (e.HasStartEnergy()) {
                    result.SetEnergy(e.GetStartEnergy());
                }
                throw;
            }
            result.SetEnergy(outcome.engine_energy);
            result.SetCoordinates(system.ToOELayout(outcome.x));
            result.SetOptimizationStats(outcome.stats);
            if (settings_.properties != 0) {
                EvaluationRequest request;
                request.properties = settings_.properties;
                const Evaluation evaluation = session->Evaluate(outcome.x, request);
                const std::string problem_text =
                    validate_evaluation(evaluation, request, system.NumAtoms());
                if (!problem_text.empty()) {
                    throw std::runtime_error("engine returned a malformed Evaluation: " +
                                             problem_text);
                }
                apply_evaluation(result, evaluation, request, system);
            }
            result.SetStatus(
                outcome.converged ? ConformerStatus::OK : ConformerStatus::NOT_CONVERGED,
                outcome.converged ? "" : outcome.message);
        });
}

namespace {

/// The stages of ``Calculation::Frequencies`` in order (spec 7.5).
const char* const FREQUENCY_STAGES[] = {"optimization", "Hessian", "analysis"};

/// Gives the conformer the stopping stage's status, with a message naming the stage, and records
/// the later stages it skips as a warning (spec 7.5, 10).
void stop_frequencies(ConformerResult& result, ConformerStatus status, const std::string& stage,
                      const std::string& reason) {
    result.SetStatus(status, stage + " stage: " + reason);
    std::string skipped;
    bool later = false;
    for (const char* name : FREQUENCY_STAGES) {
        if (later) {
            skipped += (skipped.empty() ? "" : ", ") + std::string(name);
        }
        later = later || stage == name;
    }
    if (!skipped.empty()) {
        result.AddWarning("the " + stage +
                          " stage stopped, so these stages were skipped: " + skipped);
    }
}

/// Runs one stage of ``Calculation::Frequencies``. An exception stops the conformer there: the
/// values of completed stages stay (spec 7.5), an ENGINE_ERROR also keeps what the failed stage
/// computed before failing, and an UNSUPPORTED stage keeps nothing of its own, as in single-stage
/// tasks.
/// :returns: Whether the stage completed.
bool run_frequency_stage(ConformerResult& result, const std::string& stage,
                         const std::function<void()>& work) {
    const ConformerResult before = result;
    try {
        work();
        return true;
    } catch (const UnsupportedError& e) {
        result = before;
        stop_frequencies(result, ConformerStatus::UNSUPPORTED, stage, e.what());
    } catch (const std::exception& e) {
        stop_frequencies(result, ConformerStatus::ENGINE_ERROR, stage, e.what());
    }
    return false;
}

std::string imaginary_mode_warning(const detail::FrequencyAnalysis& analysis) {
    const unsigned count = analysis.imaginary_count;
    return std::to_string(count) + (count == 1 ? " imaginary mode" : " imaginary modes") +
           " (lowest " + detail::format_value(analysis.frequencies.front()) +
           " cm^-1): the geometry is not a minimum, and imaginary modes are left out of the ZPE "
           "and thermal corrections";
}

}  // namespace

ResultSet Calculation::Frequencies(OEChem::OEMolBase& mol, const ThermoOptions& options) {
    const Clock::time_point call_start = Clock::now();
    // Every option check runs before Prepare, which may add hydrogens to the caller's molecule.
    // Resolving validates the optimizer options; its atom count only sets the default iteration
    // limit, so the settings are resolved again for the prepared system below.
    options.Validate();
    const OptimizerOptions& optimizer = options.GetOptimizerOptions();
    resolve_optimizer_settings(optimizer, settings_.preset, mol.NumAtoms(),
                               capabilities_.force_field);
    CallState state = Prepare(mol);
    const MolecularSystem& system = *state.system;
    const ResolvedOptimizerSettings resolved_settings = resolve_optimizer_settings(
        optimizer, settings_.preset, system.NumAtoms(), capabilities_.force_field);
    const bool optimize_first = options.GetOptimizeFirst();
    // Spec 7.2: the engine's analytic Hessian when the method declares one.
    const bool analytic = capabilities_.analytic_hessian;
    const double step = preset_values(settings_.preset).fd_step;
    const double setup_seconds = seconds_since(call_start);

    std::map<std::string, std::string> resolved = CallResolvedOptions(state, "frequencies");
    resolved["thermo.temperature"] = detail::format_value(options.GetTemperature());
    resolved["thermo.pressure"] = detail::format_value(options.GetPressure());
    resolved["thermo.quasi_rrho_cutoff"] = detail::format_value(options.GetQuasiRRHOCutoff());
    resolved["thermo.symmetry_number"] = std::to_string(options.GetSymmetryNumber());
    resolved["thermo.optimize_first"] = optimize_first ? "true" : "false";
    resolved["hessian"] = analytic ? "analytic" : "finite_difference";
    if (!analytic) {
        resolved["hessian.step"] = detail::format_value(step);
    }
    // The optimizer settings change the results only when the optimization runs.
    if (optimize_first) {
        resolved["optimizer.algorithm"] = algorithm_name(resolved_settings.algorithm);
        resolved["optimizer.max_iterations"] = std::to_string(resolved_settings.max_iterations);
        resolved["optimizer.energy_change"] = detail::format_value(resolved_settings.energy_change);
        resolved["optimizer.gradient_norm"] = detail::format_value(resolved_settings.gradient_norm);
        resolved["optimizer.initial_trust_radius"] =
            detail::format_value(resolved_settings.initial_trust_radius);
        resolved["optimizer.max_trust_radius"] =
            detail::format_value(resolved_settings.max_trust_radius);
    }

    const ConformerBody body = [&](ConformerResult& result, unsigned position) {
        // One session serves the optimization and every Hessian displacement (spec 6.4).
        std::shared_ptr<EngineSession> session;
        std::vector<double> x = system.GetConformerCoordinates(position);
        if (optimize_first) {
            std::string not_converged;
            const bool completed = run_frequency_stage(result, "optimization", [&] {
                session = create_session(*engine_, engine_name_, system, settings_);
                detail::OptimizationProblem problem;
                problem.x0 = x;
                problem.atomic_numbers = system.GetAtomicNumbers();
                problem.settings = resolved_settings;
                detail::OptimizationOutcome outcome;
                try {
                    outcome = detail::run_optimization(*session, problem);
                } catch (const detail::OptimizationFailure& e) {
                    // Spec 7.5: an engine failure keeps the values computed before it.
                    if (e.HasPartialOutcome()) {
                        const detail::OptimizationOutcome& partial = e.GetPartialOutcome();
                        result.SetEnergy(partial.engine_energy);
                        result.SetCoordinates(system.ToOELayout(partial.x));
                        result.SetOptimizationStats(partial.stats);
                    } else if (e.HasStartEnergy()) {
                        result.SetEnergy(e.GetStartEnergy());
                    }
                    throw;
                }
                result.SetEnergy(outcome.engine_energy);
                result.SetCoordinates(system.ToOELayout(outcome.x));
                result.SetOptimizationStats(outcome.stats);
                x = outcome.x;
                if (!outcome.converged) {
                    not_converged = outcome.message;
                }
            });
            if (!completed) {
                return;
            }
            if (!not_converged.empty()) {
                stop_frequencies(result, ConformerStatus::NOT_CONVERGED, "optimization",
                                 not_converged);
                return;
            }
        }
        std::vector<double> hessian;
        const bool completed = run_frequency_stage(result, "Hessian", [&] {
            if (!session) {
                session = create_session(*engine_, engine_name_, system, settings_);
            }
            // Without an optimization the energy at the analyzed geometry is still unknown, and
            // requested properties always come from that geometry.
            if (!optimize_first || settings_.properties != 0) {
                EvaluationRequest request;
                request.properties = settings_.properties;
                const Evaluation evaluation = session->Evaluate(x, request);
                const std::string problem =
                    validate_evaluation(evaluation, request, system.NumAtoms());
                if (!problem.empty()) {
                    throw std::runtime_error("engine returned a malformed Evaluation: " + problem);
                }
                apply_evaluation(result, evaluation, request, system);
            }
            hessian = detail::compute_hessian(*session, x, analytic, step);
            result.SetHessian(hessian);
        });
        if (!completed) {
            return;
        }
        run_frequency_stage(result, "analysis", [&] {
            const detail::FrequencyAnalysis analysis =
                detail::analyze_frequencies(hessian, x, system.GetAtomicNumbers());
            result.SetFrequencies(analysis.frequencies, analysis.normal_modes,
                                  analysis.imaginary_count);
            result.SetThermochemistry(detail::compute_thermochemistry(
                result.GetEnergy(), analysis.frequencies, analysis.principal_moments,
                analysis.total_mass, options.GetSymmetryNumber(), analysis.linear,
                system.GetMultiplicity(), options.GetTemperature(), options.GetPressure(),
                options.GetQuasiRRHOCutoff()));
            // Spec 7.3: imaginary modes are a warning; the status stays OK.
            if (analysis.imaginary_count > 0) {
                result.AddWarning(imaginary_mode_warning(analysis));
            }
            result.SetStatus(ConformerStatus::OK, "");
        });
    };
    return RunConformers(mol, state, resolved, call_start, setup_seconds, optimize_first, body);
}

ScanResult Calculation::Scan(OEChem::OEMolBase& mol, const ScanOptions& options) {
    // Every check that can reject the call runs before Prepare, which may add hydrogens to the
    // caller's molecule. Adding hydrogens appends atoms without renumbering, so every atom
    // checked here stays valid.
    options.Validate();
    if (settings_.properties != 0) {
        throw OptionsError("scans do not compute properties; requested: " +
                           property_names(settings_.properties));
    }
    const OptimizerOptions& optimizer = options.GetOptimizerOptions();
    // At each grid point the scanned dihedrals join the caller's terms as exact constraints.
    // These placeholder targets only serve the atom check and the algorithm choice, which
    // depend on which terms exist, not on their targets.
    ConstraintSet with_dihedrals = optimizer.GetConstraints();
    for (unsigned k = 0; k < options.GetNumDihedrals(); ++k) {
        const std::vector<unsigned> atoms = options.GetDihedral(k);
        with_dihedrals.AddDihedral(atoms[0], atoms[1], atoms[2], atoms[3], 0.0);
    }
    OptimizerOptions constrained = optimizer;
    constrained.SetConstraints(with_dihedrals);
    resolve_optimizer_settings(constrained, settings_.preset, mol.NumAtoms(),
                               capabilities_.force_field);
    detail::check_constraint_atoms(with_dihedrals, mol);
    CallState state = Prepare(mol);
    const MolecularSystem& system = *state.system;
    const ResolvedOptimizerSettings resolved_settings = resolve_optimizer_settings(
        constrained, settings_.preset, system.NumAtoms(), capabilities_.force_field);
    const detail::ScanGrid grid = detail::build_scan_grid(options);
    // Spec 7.4: by default, 20 optimizations per grid point.
    const std::size_t default_cap = std::min<std::size_t>(
        detail::SCAN_OPTIMIZATIONS_PER_POINT * grid.Size(), std::numeric_limits<unsigned>::max());
    const unsigned max_optimizations = options.HasMaxOptimizations()
                                           ? options.GetMaxOptimizations()
                                           : static_cast<unsigned>(default_cap);

    std::vector<std::array<unsigned, 4>> dense_dihedrals;
    for (unsigned k = 0; k < options.GetNumDihedrals(); ++k) {
        const std::vector<unsigned> atoms = options.GetDihedral(k);
        dense_dihedrals.push_back({system.GetDenseIndex(atoms[0]), system.GetDenseIndex(atoms[1]),
                                   system.GetDenseIndex(atoms[2]), system.GetDenseIndex(atoms[3])});
    }
    // Spec 7.4: each input conformer, in order, seeds the point nearest its dihedral values.
    std::vector<std::pair<std::size_t, std::vector<double>>> seeds;
    for (unsigned position = 0; position < system.NumConformers(); ++position) {
        const std::vector<double>& x = system.GetConformerCoordinates(position);
        std::vector<double> angles;
        for (const std::array<unsigned, 4>& atoms : dense_dihedrals) {
            angles.push_back(
                detail::dihedral(x.data(), atoms[0], atoms[1], atoms[2], atoms[3], nullptr) *
                Units::RADIANS_TO_DEGREES);
        }
        seeds.emplace_back(grid.Nearest(angles), x);
    }

    // Spec 6.6 holds a constraint or restraint without a target at its starting value, and a
    // position restraint at its starting position. A scan's optimizations start from many
    // geometries, so those values come from the first input conformer for every grid point;
    // taking them from each start would let them drift as the wavefront moves.
    const std::vector<double>& reference_x = system.GetConformerCoordinates(0);
    const detail::OptimizeAtPoint optimize = [&](std::size_t point,
                                                 const std::vector<double>& start_x) {
        detail::ScanAttempt attempt;
        try {
            ConstraintSet constraints = optimizer.GetConstraints();
            const std::vector<double> angles = grid.Angles(point);
            for (unsigned k = 0; k < options.GetNumDihedrals(); ++k) {
                const std::vector<unsigned> atoms = options.GetDihedral(k);
                constraints.AddDihedral(atoms[0], atoms[1], atoms[2], atoms[3], angles[k]);
            }
            // Spec 6.4: a fresh session for every grid-point optimization.
            std::shared_ptr<EngineSession> session =
                create_session(*engine_, engine_name_, system, settings_);
            detail::OptimizationProblem problem;
            problem.x0 = start_x;
            problem.atomic_numbers = system.GetAtomicNumbers();
            problem.constraints = detail::resolve_constraint_set(constraints, system, reference_x);
            problem.settings = resolved_settings;
            const detail::OptimizationOutcome outcome = detail::run_optimization(*session, problem);
            attempt.has_energy = true;
            attempt.energy = outcome.engine_energy;
            attempt.converged = outcome.converged;
            attempt.x = outcome.x;
            attempt.message = outcome.converged ? "" : outcome.message;
        } catch (const UnsupportedError& e) {
            attempt = detail::ScanAttempt();
            attempt.unsupported = true;
            attempt.message = e.what();
        } catch (const std::exception& e) {
            // An OptimizationFailure carries the last iterate the optimizer accepted, but that
            // geometry need not satisfy the grid constraints, so a failed attempt keeps no energy
            // and no geometry however far the optimizer got. This is the only place that can
            // enforce it: the wavefront ranks every attempt that reports an energy.
            attempt = detail::ScanAttempt();
            attempt.has_energy = false;
            attempt.message = e.what();
        }
        return attempt;
    };
    const detail::WavefrontResult wavefront =
        detail::run_wavefront(grid, seeds, max_optimizations, optimize);

    std::map<std::string, std::string> resolved = CallResolvedOptions(state, "scan");
    const ConstraintSet& constraints = optimizer.GetConstraints();
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
    std::string dihedrals;
    for (unsigned k = 0; k < options.GetNumDihedrals(); ++k) {
        dihedrals += (k == 0 ? "" : ";") + encode_indices(options.GetDihedral(k));
    }
    resolved["scan.dihedrals"] = dihedrals;
    resolved["scan.step"] = detail::format_value(options.GetStep());
    resolved["scan.range_start"] = detail::format_value(options.GetRangeStart());
    resolved["scan.range_end"] = detail::format_value(options.GetRangeEnd());
    resolved["scan.max_optimizations"] = std::to_string(max_optimizations);

    // NewResult assembles the provenance and the call's warnings exactly as every other task
    // records them; a scan has no per-conformer results, so it borrows one.
    const ConformerResult call = NewResult(state, 0, resolved);
    ScanResult result;
    result.SetProvenance(call.GetProvenance());
    for (const std::string& warning : call.GetWarnings()) {
        result.AddWarning(warning);
    }
    for (unsigned k = 0; k < options.GetNumDihedrals(); ++k) {
        result.AddDihedral(options.GetDihedral(k));
    }
    result.SetMolecule(
        mol, "OEVibe." + settings_.method.GetName() + "." + settings_.solvent.GetTag() + ".");
    std::size_t without_energy = 0;
    for (std::size_t index = 0; index < grid.Size(); ++index) {
        const detail::WavefrontPoint& visited = wavefront.points[index];
        ScanPoint point;
        point.SetAngles(grid.Angles(index));
        // Spec 7.5 scan-point statuses.
        if (visited.has_best) {
            point.SetEnergy(visited.best.energy);
            point.SetCoordinates(system.ToOELayout(visited.best.x));
            point.SetStatus(
                visited.best.converged ? ScanPointStatus::OK : ScanPointStatus::NOT_CONVERGED,
                visited.best.message);
        } else if (visited.attempts == 0) {
            ++without_energy;
            point.SetStatus(ScanPointStatus::NOT_ATTEMPTED,
                            wavefront.truncated ? "the optimization cap was reached first"
                                                : "propagation never reached this point");
        } else {
            ++without_energy;
            point.SetStatus(visited.any_unsupported ? ScanPointStatus::UNSUPPORTED
                                                    : ScanPointStatus::ENGINE_ERROR,
                            visited.last_message);
        }
        result.AddPoint(std::move(point));
    }
    if (wavefront.truncated) {
        result.SetTruncated(true);
        result.AddWarning("the scan stopped at its cap of " + std::to_string(max_optimizations) +
                          " optimizations before propagation finished; " +
                          std::to_string(without_energy) + " of " + std::to_string(grid.Size()) +
                          " grid points have no energy");
    }
    result.Finalize();
    return result;
}

}  // namespace OEVIBE
