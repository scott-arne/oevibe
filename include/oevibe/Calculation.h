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
    std::map<std::string, std::string> BaseResolvedOptions() const;

    CalculationOptions options_;
    std::shared_ptr<Engine> engine_;
    std::string engine_name_;
    MethodCapabilities capabilities_;
    EngineSettings settings_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_CALCULATION_H
