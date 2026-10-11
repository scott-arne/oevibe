#ifndef OEVIBE_CALCULATION_H
#define OEVIBE_CALCULATION_H

#include <oechem.h>

#include <map>
#include <memory>
#include <string>

#include "oevibe/Engine.h"
#include "oevibe/Options.h"
#include "oevibe/Result.h"

namespace OEVIBE {

/// Runs tasks over every conformer of a molecule with one resolved engine (spec 5.6).
/// Not thread-safe; use one per thread.
class Calculation {
public:
    /// :raises OptionsError: For invalid options, unsupported method/solvent/property
    ///     combinations, or keywords rejected by the engine's schema.
    /// :raises EngineUnavailableError: When the resolved engine or method cannot run.
    explicit Calculation(const CalculationOptions& options);

    /// :raises InputError: When the molecule fails validation; nothing runs then.
    ResultSet Energy(OEChem::OEMolBase& mol);
    /// :raises InputError: When the molecule fails validation; nothing runs then.
    ResultSet Gradient(OEChem::OEMolBase& mol);

    const CalculationOptions& GetOptions() const;
    const std::string& GetEngineName() const;

private:
    struct CallState;

    CallState Prepare(OEChem::OEMolBase& mol) const;
    ResultSet RunSinglePoints(OEChem::OEMolBase& mol, bool gradient);
    ConformerResult NewResult(const CallState& state, unsigned position,
                              const std::map<std::string, std::string>& resolved) const;
    /// :returns: How the snapshot's partial charges are prepared for the resolved method.
    PartialChargePolicy ChargePolicy() const;
    /// :returns: The resolved options that do not depend on the molecule or the task.
    std::map<std::string, std::string> BaseResolvedOptions() const;
    /// :returns: ``BaseResolvedOptions()`` plus the charge, multiplicity, partial-charge policy,
    ///     and ``task`` of one call; the same settings give different results for another
    ///     charge, spin state, or task, so they belong in the options digest.
    std::map<std::string, std::string> CallResolvedOptions(const CallState& state,
                                                           const std::string& task) const;

    CalculationOptions options_;
    std::shared_ptr<Engine> engine_;
    std::string engine_name_;
    std::string engine_version_;  // Read once at construction, where failures may throw.
    MethodCapabilities capabilities_;
    EngineSettings settings_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_CALCULATION_H
