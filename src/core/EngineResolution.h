#ifndef OEVIBE_CORE_ENGINERESOLUTION_H
#define OEVIBE_CORE_ENGINERESOLUTION_H

#include <memory>
#include <string>

#include "oevibe/Engine.h"
#include "oevibe/MolecularSystem.h"
#include "oevibe/Options.h"

namespace OEVIBE {
namespace detail {

/// One resolution of a ``CalculationOptions`` against the registry (spec 5.3, 5.4, 8.2). Both
/// ``Calculation`` and ``EngineMolFunc`` run it, so they agree on engine, method and settings.
struct ResolvedEngine {
    std::shared_ptr<Engine> engine;
    std::string name;
    std::string version;
    MethodCapabilities capabilities;
    EngineSettings settings;
    std::string solvent_warning;  ///< Set when SolventModel::AUTO resolved to a model.
};

/// Resolves the engine, checks it and the method are available, builds the ``EngineSettings``, and
/// validates them against the engine's keyword schema. The caller validates ``options`` first.
/// :raises EngineUnavailableError: When the named engine is not registered, the engine is
///     unavailable, or the method is unavailable within it.
/// :raises OptionsError: When the engine does not provide the method, a requested property is not
///     supported, or the engine rejects the settings.
ResolvedEngine resolve_engine(const CalculationOptions& options);

/// :returns: How the snapshot's partial charges are prepared for the resolved method (spec 6.2).
PartialChargePolicy charge_policy(const CalculationOptions& options,
                                  const MethodCapabilities& capabilities);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_ENGINERESOLUTION_H
