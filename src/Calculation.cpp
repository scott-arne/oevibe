#include "oevibe/Calculation.h"

#include <algorithm>
#include <chrono>
#include <memory>

#include "core/InputPrep.h"
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
}

ResultSet Calculation::Energy(OEChem::OEMolBase& mol) { return RunSinglePoints(mol, false); }
ResultSet Calculation::Gradient(OEChem::OEMolBase& mol) { return RunSinglePoints(mol, true); }
const CalculationOptions& Calculation::GetOptions() const { return options_; }
const std::string& Calculation::GetEngineName() const { return engine_name_; }

Calculation::CallState Calculation::Prepare(OEChem::OEMolBase& mol) const {
    CallState state;
    state.input = detail::prepare_input(mol, options_);
    PartialChargePolicy policy = PartialChargePolicy::NONE;
    if (capabilities_.requires_partial_charges) {
        policy = options_.GetUseInputPartialCharges() ? PartialChargePolicy::USE_INPUT
                                                      : PartialChargePolicy::ASSIGN_AM1BCC_ELF10;
    }
    state.system = std::make_unique<MolecularSystem>(mol, state.input.charge,
                                                     state.input.multiplicity, policy);
    engine_->ValidateSystem(*state.system, settings_);
    return state;
}

std::map<std::string, std::string> Calculation::BaseResolvedOptions() const {
    std::map<std::string, std::string> resolved = {
        {"method", settings_.method.GetName()},
        {"engine", engine_name_},
        {"solvent", settings_.solvent.GetTag()},
        {"preset", preset_name(settings_.preset)},
        {"accuracy", options_.HasKeyword("accuracy") ? options_.GetKeyword("accuracy")
                                                     : detail::format_value(settings_.accuracy)},
        {"properties", std::to_string(settings_.properties)},
    };
    for (const auto& [name, value] : settings_.keywords) {
        resolved["keyword." + name] = value;
    }
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
    provenance.engine_version = engine_->GetVersion();
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
    const std::map<std::string, std::string> resolved = BaseResolvedOptions();
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

}  // namespace OEVIBE
