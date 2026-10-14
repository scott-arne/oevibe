#include "core/EngineResolution.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "oevibe/Preset.h"
#include "oevibe/Property.h"
#include "oevibe/Solvent.h"

namespace OEVIBE {
namespace detail {

namespace {

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

}  // namespace

ResolvedEngine resolve_engine(const CalculationOptions& options) {
    ResolvedEngine resolved;
    EngineRegistry& registry = EngineRegistry::Instance();
    if (options.HasEngine()) {
        resolved.engine = registry.Find(options.GetEngine());
        if (!resolved.engine) {
            throw EngineUnavailableError("no engine named '" + options.GetEngine() +
                                         "' is registered");
        }
    } else {
        resolved.engine = registry.ResolveForMethod(options.GetMethod());
    }
    resolved.name = resolved.engine->GetName();
    if (!resolved.engine->IsAvailable()) {
        throw EngineUnavailableError("engine '" + resolved.name + "' is unavailable: " +
                                     resolved.engine->GetUnavailableReason());
    }
    const std::optional<MethodCapabilities> caps =
        find_method(*resolved.engine, options.GetMethod());
    if (!caps.has_value()) {
        throw OptionsError("engine '" + resolved.name + "' does not provide method '" +
                           options.GetMethod().GetName() + "'");
    }
    if (!caps->available) {
        throw EngineUnavailableError("method '" + options.GetMethod().GetName() + "' of engine '" +
                                     resolved.name +
                                     "' is unavailable: " + caps->unavailable_reason);
    }
    resolved.version = resolved.engine->GetVersion();
    resolved.capabilities = *caps;
    resolved.settings.method = options.GetMethod();
    resolved.settings.solvent = resolve_solvent(options.GetSolvent(), resolved.capabilities);
    // Spec 10 records solvent model resolution per result, so the model AUTO chose is visible
    // without reading the provenance.
    if (!resolved.settings.solvent.IsGasPhase() &&
        options.GetSolvent().GetModel() == SolventModel::AUTO) {
        resolved.solvent_warning = "solvent model AUTO resolved to '" +
                                   solvent_model_name(resolved.settings.solvent.GetModel()) +
                                   "' for solvent '" + resolved.settings.solvent.GetName() +
                                   "', the default of method '" + options.GetMethod().GetName() +
                                   "'";
    }
    resolved.settings.preset = options.GetPreset();
    resolved.settings.accuracy = preset_values(resolved.settings.preset).engine_accuracy;
    resolved.settings.properties = options.GetEffectiveProperties();
    resolved.settings.keywords = options.GetKeywords();
    const unsigned unsupported = resolved.settings.properties & ~resolved.capabilities.properties;
    if (unsupported != 0) {
        throw OptionsError("method '" + options.GetMethod().GetName() +
                           "' does not provide: " + property_names(unsupported));
    }
    resolved.engine->ValidateSettings(resolved.settings);
    // Applied after validation so the keyword's type and bounds are checked first.
    if (options.HasKeyword("accuracy")) {
        resolved.settings.accuracy = parse_accuracy(options.GetKeyword("accuracy"));
    }
    return resolved;
}

PartialChargePolicy charge_policy(const CalculationOptions& options,
                                  const MethodCapabilities& capabilities) {
    if (!capabilities.requires_partial_charges) {
        return PartialChargePolicy::NONE;
    }
    return options.GetUseInputPartialCharges() ? PartialChargePolicy::USE_INPUT
                                               : PartialChargePolicy::ASSIGN_AM1BCC_ELF10;
}

}  // namespace detail
}  // namespace OEVIBE
