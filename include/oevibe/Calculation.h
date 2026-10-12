#ifndef OEVIBE_CALCULATION_H
#define OEVIBE_CALCULATION_H

#include <oechem.h>

#include <chrono>
#include <functional>
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
    /// Optimizes every conformer (spec 7.1) and writes final coordinates back unless disabled.
    /// :raises OptionsError: When the optimizer options are invalid.
    /// :raises InputError: When the molecule fails validation or a constraint names an atom
    ///     that is not in the molecule; nothing runs then.
    ResultSet Optimize(OEChem::OEMolBase& mol, const OptimizerOptions& options);

    const CalculationOptions& GetOptions() const;
    const std::string& GetEngineName() const;

private:
    struct CallState;
    /// One conformer's computation inside ``RunConformers``. On success it sets the values and
    /// the final status; on failure it throws, after setting the values that spec 7.5 keeps.
    using ConformerBody = std::function<void(ConformerResult& result, unsigned position)>;

    CallState Prepare(OEChem::OEMolBase& mol) const;
    ResultSet RunSinglePoints(OEChem::OEMolBase& mol, bool gradient);
    /// Runs ``body`` once per conformer on a fresh ``NewResult`` and applies spec 7.5 to its
    /// exceptions: ``UnsupportedError`` discards everything ``body`` set and gives UNSUPPORTED;
    /// any other ``std::exception`` keeps those values and gives ENGINE_ERROR with its text.
    /// Then records timings, finalizes the results, and writes them back (spec 6.5).
    ResultSet RunConformers(OEChem::OEMolBase& mol, const CallState& state,
                            const std::map<std::string, std::string>& resolved,
                            std::chrono::steady_clock::time_point call_start, double setup_seconds,
                            bool write_coordinates, const ConformerBody& body);
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
    std::string solvent_warning_;  // Set when AUTO was resolved; copied into every result.
};

}  // namespace OEVIBE

#endif  // OEVIBE_CALCULATION_H
