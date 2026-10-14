#ifndef OEVIBE_OEADAPTERS_H
#define OEVIBE_OEADAPTERS_H

#include <oemolpotential.h>

#include <memory>
#include <string>
#include <vector>

#include "oevibe/Engine.h"
#include "oevibe/Options.h"

namespace OEVIBE {

/// Any OEVibe engine as an OpenEye potential (spec 8.8), so ``OEBFGSOpt``, the OEOpt adaptors and
/// ``OEGenericFF2::Add`` can drive it. Coordinates and gradients use the OpenEye layout of the
/// molecule passed to ``Setup``: index gaps occupy variables that are read and written as zero.
class EngineMolFunc : public OEMolPotential::OEMolFunc2 {
public:
    /// One evaluation in the OpenEye layout. Unlike ``operator()`` this gradient is a fresh
    /// array, because its callers have no buffer of their own to accumulate into.
    struct FunctionValue {
        double energy = 0.0;
        std::vector<double> gradient;  ///< ``NumVar()`` values, empty when not requested.
    };

    /// Resolves and validates the engine through the registry; Python engines work too.
    /// :raises OptionsError: For invalid options, or a method the engine does not provide.
    /// :raises EngineUnavailableError: When the engine or the method is unavailable.
    explicit EngineMolFunc(const CalculationOptions& options);
    ~EngineMolFunc() override;

    /// Prepares a private copy of ``mol`` (spec 6.1), builds the system and the session.
    /// :returns: False when preparation changes the atom set, validation fails, or the session
    ///     cannot be created; ``GetSetupError()`` then carries the reason.
    bool Setup(const OEChem::OEMolBase& mol) override;
    /// :returns: Three times the molecule's ``GetMaxAtomIdx()``, or 0 before a successful Setup.
    unsigned int NumVar() const override;
    /// :returns: The energy in kcal/mol.
    /// :raises std::exception: Whatever the engine session throws; OpenEye's single-value call
    ///     has no error channel, so the exception unwinds through the caller.
    double operator()(const double* x) override;
    /// Accumulates the gradient in kcal/mol/Angstrom into ``g``; ``g`` is never initialized here.
    double operator()(const double* x, double* g) override;
    /// Accumulates the packed lower triangle of the Hessian into ``h`` and the gradient into ``g``.
    /// ``h`` holds element (row, col <= row) at ``row * (row + 1) / 2 + col``, OpenEye's layout.
    /// :returns: False when the engine fails; ``GetSetupError()`` carries the reason.
    bool operator()(const double* x, double* h, double* g) override;
    /// :returns: Null; OEVibe engines report no energy components (spec 8.8).
    OESystem::OEIterBase<OEOpt::OEFComponent>* GetFComponents(const double* x) override;
    /// :returns: A new function with the same options, not yet set up on a molecule.
    EngineMolFunc* CreateCopy() const override;

    /// The pointer-free companion of ``operator()``, for callers that hold no raw buffer
    /// (Python's ``as_oemolfunc``).
    /// :param x: ``NumVar()`` coordinates in Angstrom, OpenEye layout.
    /// :param want_gradient: Whether to fill ``FunctionValue::gradient``.
    /// :raises OEVibeError: When ``x`` is not ``NumVar()`` long or Setup has not succeeded.
    /// :raises std::exception: Whatever the engine session throws.
    FunctionValue Evaluate(const std::vector<double>& x, bool want_gradient);

    /// :returns: Why the last ``Setup`` or Hessian call failed, or an empty string.
    const std::string& GetSetupError() const;

private:
    /// :returns: Dense coordinates gathered from the OpenEye-layout ``x``.
    std::vector<double> ToDense(const double* x) const;
    /// Evaluates at ``x`` unless the cache already holds it, then returns the cached energy.
    double EvaluateCached(const double* x, bool need_gradient);
    /// Scatters the cached dense gradient into the OpenEye-layout ``g``, adding to what is
    /// there: OpenEye sums the components of a composite potential into one buffer.
    void AccumulateGradient(double* g) const;

    CalculationOptions options_;
    std::shared_ptr<Engine> engine_;
    MethodCapabilities capabilities_;
    EngineSettings settings_;
    // Declared before the system and the session: both may hold references into it.
    OEChem::OEMol molecule_;
    std::unique_ptr<MolecularSystem> system_;
    std::shared_ptr<EngineSession> session_;
    /// The caller's atom index for each dense position, read from the molecule passed to Setup
    /// before it was copied: the copy compacts index gaps, so the snapshot cannot supply this.
    std::vector<unsigned> caller_index_of_dense_;
    unsigned num_var_ = 0;
    double fd_step_ = 0.005;
    std::string setup_error_;
    std::vector<double> cached_x_;
    std::vector<double> cached_gradient_;  ///< Dense 3N, present only when one was computed.
    double cached_energy_ = 0.0;
    bool cached_has_gradient_ = false;
};

/// One OpenEye potential as a single-method OEVibe engine (spec 8.8), so user-composed OpenEye
/// potentials get OEVibe's optimizer and scans. It is not registered automatically.
class MolFuncEngine : public Engine {
public:
    /// :param name: Registry name.
    /// :param version: Reported engine version.
    /// :param method: The one method this engine declares.
    /// :param prototype: Copied with ``CreateCopy()`` now; each session gets its own copy.
    /// :param requires_partial_charges: Declared in the method's capabilities.
    /// :raises OptionsError: When ``prototype.CreateCopy()`` returns null, as some OpenEye wrapper
    ///     potentials do; the engine could not give each session its own copy.
    MolFuncEngine(std::string name, std::string version, Method method,
                  const OEMolPotential::OEMolFunc1& prototype,
                  bool requires_partial_charges = false);

    std::string GetName() const override;
    std::string GetVersion() const override;
    bool IsAvailable() const override;
    std::string GetUnavailableReason() const override;
    std::vector<MethodCapabilities> GetMethods() const override;
    /// :raises UnsupportedError: When the potential cannot be copied or set up on the system.
    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override;

private:
    std::string name_;
    std::string version_;
    Method method_;
    std::unique_ptr<OEMolPotential::OEMolFunc1> prototype_;
    bool requires_partial_charges_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_OEADAPTERS_H
