#ifndef OEVIBE_OPTIONS_H
#define OEVIBE_OPTIONS_H

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "oevibe/Constraints.h"
#include "oevibe/Method.h"
#include "oevibe/Preset.h"
#include "oevibe/Property.h"
#include "oevibe/Solvent.h"

namespace OEVIBE {

// NOLINTBEGIN(performance-enum-size)
/// Geometry optimizer (spec 7.1). AUTO picks LBFGS for a force-field method without exact
/// constraints and ANC_RFO otherwise.
enum class Algorithm { AUTO, ANC_RFO, LBFGS };
// NOLINTEND(performance-enum-size)

/// :returns: ``auto``, ``anc_rfo``, or ``lbfgs``.
std::string algorithm_name(Algorithm algorithm);

/// Model options shared by every task of a ``Calculation`` (spec 5.5). The preset lives only
/// here (spec 5.4).
class CalculationOptions {
public:
    /// STANDARD preset, ``gfn2-xtb``, gas phase.
    CalculationOptions();
    /// :raises OptionsError: For a value outside the enumerators.
    static CalculationOptions FromPreset(Preset preset);

    const Method& GetMethod() const;
    void SetMethod(const Method& method);
    bool HasEngine() const;
    /// :raises OptionsError: When no engine override is set.
    const std::string& GetEngine() const;
    /// :raises OptionsError: For an empty name.
    void SetEngine(const std::string& engine);
    void ClearEngine();
    const Solvent& GetSolvent() const;
    void SetSolvent(const Solvent& solvent);
    Preset GetPreset() const;
    /// :raises OptionsError: For a value outside the enumerators.
    void SetPreset(Preset preset);
    bool HasCharge() const;
    /// :raises OptionsError: When no charge override is set.
    int GetCharge() const;
    void SetCharge(int charge);
    void ClearCharge();
    bool HasMultiplicity() const;
    /// :raises OptionsError: When no multiplicity override is set.
    unsigned GetMultiplicity() const;
    /// :raises OptionsError: When 0.
    void SetMultiplicity(unsigned multiplicity);
    void ClearMultiplicity();
    /// :returns: The requested ``Property`` bit set.
    unsigned GetProperties() const;
    /// :raises OptionsError: For bits outside ``ALL_PROPERTIES``.
    void SetProperties(unsigned properties);
    /// :returns: The requested properties plus CHARGES when partial-charge write-back is on.
    unsigned GetEffectiveProperties() const;
    bool GetAddHydrogens() const;
    void SetAddHydrogens(bool add);
    bool GetUpdateCoordinates() const;
    void SetUpdateCoordinates(bool update);
    bool GetWriteTags() const;
    void SetWriteTags(bool write);
    bool GetAssignPartialCharges() const;
    void SetAssignPartialCharges(bool assign);
    bool GetUseInputPartialCharges() const;
    void SetUseInputPartialCharges(bool use);
    /// Engine-specific escape hatch; the engine's keyword schema validates it (spec 8.1).
    /// :raises OptionsError: For an empty name.
    void SetKeyword(const std::string& name, const std::string& value);
    bool HasKeyword(const std::string& name) const;
    /// :raises OptionsError: When the keyword is not set.
    const std::string& GetKeyword(const std::string& name) const;
    void ClearKeyword(const std::string& name);
    std::map<std::string, std::string> GetKeywords() const;

    /// Cross-field checks; none apply in v1, kept for the house-style contract.
    void Validate() const;

private:
    Method method_;
    std::string engine_;
    Solvent solvent_;
    Preset preset_ = Preset::STANDARD;
    std::optional<int> charge_;
    std::optional<unsigned> multiplicity_;
    unsigned properties_ = 0;
    bool add_hydrogens_ = true;
    bool update_coordinates_ = true;
    bool write_tags_ = true;
    bool assign_partial_charges_ = false;
    bool use_input_partial_charges_ = false;
    std::map<std::string, std::string> keywords_;
};

/// Optimizer options; unset thresholds come from the calculation's preset (spec 5.4).
class OptimizerOptions {
public:
    Algorithm GetAlgorithm() const;
    /// :raises OptionsError: For a value outside the enumerators.
    void SetAlgorithm(Algorithm algorithm);
    bool HasMaxIterations() const;
    /// :raises OptionsError: When unset.
    unsigned GetMaxIterations() const;
    /// :raises OptionsError: When 0.
    void SetMaxIterations(unsigned iterations);
    void ClearMaxIterations();
    bool HasEnergyChangeThreshold() const;
    /// :raises OptionsError: When unset.
    double GetEnergyChangeThreshold() const;
    /// :raises OptionsError: Unless finite and greater than 0 (kcal/mol).
    void SetEnergyChangeThreshold(double threshold);
    void ClearEnergyChangeThreshold();
    bool HasGradientNormThreshold() const;
    /// :raises OptionsError: When unset.
    double GetGradientNormThreshold() const;
    /// :raises OptionsError: Unless finite and greater than 0 (kcal/mol/Angstrom).
    void SetGradientNormThreshold(double threshold);
    void ClearGradientNormThreshold();
    bool HasInitialTrustRadius() const;
    /// :raises OptionsError: When unset.
    double GetInitialTrustRadius() const;
    /// :raises OptionsError: Unless finite and greater than 0 (Angstrom).
    void SetInitialTrustRadius(double radius);
    void ClearInitialTrustRadius();
    bool HasMaxTrustRadius() const;
    /// :raises OptionsError: When unset.
    double GetMaxTrustRadius() const;
    /// :raises OptionsError: Unless finite and greater than 0 (Angstrom).
    void SetMaxTrustRadius(double radius);
    void ClearMaxTrustRadius();
    const ConstraintSet& GetConstraints() const;
    void SetConstraints(const ConstraintSet& constraints);

    /// :raises OptionsError: For LBFGS with exact constraints, or an explicit initial trust
    ///     radius above an explicit maximum.
    void Validate() const;

private:
    Algorithm algorithm_ = Algorithm::AUTO;
    std::optional<unsigned> max_iterations_;
    std::optional<double> energy_change_;
    std::optional<double> gradient_norm_;
    std::optional<double> initial_trust_radius_;
    std::optional<double> max_trust_radius_;
    ConstraintSet constraints_;
};

/// Frequency and thermochemistry options (spec 5.5, 7.3). The optimizer options drive the
/// optional preliminary optimization; their unset fields come from the calculation's preset.
class ThermoOptions {
public:
    double GetTemperature() const;
    /// :raises OptionsError: Unless finite and greater than 0 (K). Default 298.15.
    void SetTemperature(double kelvin);
    double GetPressure() const;
    /// :raises OptionsError: Unless finite and greater than 0 (atm). Default 1.
    void SetPressure(double atm);
    double GetQuasiRRHOCutoff() const;
    /// :raises OptionsError: Unless finite and greater than 0 (cm^-1). Default 100.
    void SetQuasiRRHOCutoff(double cm);
    unsigned GetSymmetryNumber() const;
    /// :raises OptionsError: When 0. Default 1; point groups are not detected (spec 7.3).
    void SetSymmetryNumber(unsigned n);
    bool GetOptimizeFirst() const;
    /// Default true: optimize, then compute the Hessian at the optimized geometry.
    void SetOptimizeFirst(bool optimize);
    const OptimizerOptions& GetOptimizerOptions() const;
    void SetOptimizerOptions(const OptimizerOptions& options);

    /// :raises OptionsError: When the optimizer options fail Validate() or contain any
    ///     frozen atom, constraint, or restraint (frequencies need a stationary point).
    void Validate() const;

private:
    double temperature_ = 298.15;
    double pressure_ = 1.0;
    double quasi_rrho_cutoff_ = 100.0;
    unsigned symmetry_number_ = 1;
    bool optimize_first_ = true;
    OptimizerOptions optimizer_;
};

/// Relaxed dihedral scan options (spec 5.5, 7.4). Each dihedral is four OpenEye atom indices; a
/// 2D scan's two dihedrals share the step and the range. The optimizer options apply at every
/// grid point, where the scanned dihedrals are added as exact constraints.
class ScanOptions {
public:
    /// :raises OptionsError: For repeated atoms, a dihedral already added (in either
    ///     direction), or a third dihedral.
    void AddDihedral(unsigned a, unsigned b, unsigned c, unsigned d);
    unsigned GetNumDihedrals() const;
    /// :raises OptionsError: For an index past the end.
    std::vector<unsigned> GetDihedral(unsigned index) const;
    void ClearDihedrals();
    double GetStep() const;
    /// :raises OptionsError: Unless finite and greater than 0 (degrees).
    void SetStep(double degrees);
    double GetRangeStart() const;
    double GetRangeEnd() const;
    /// Degrees; the ends need not lie in (-180, 180], because grid angles are normalized.
    /// :raises OptionsError: Unless both are finite.
    void SetRange(double start, double end);
    bool HasMaxOptimizations() const;
    /// :raises OptionsError: When unset.
    unsigned GetMaxOptimizations() const;
    /// Caps the constrained optimizations of one scan; unset means 20 per grid point (spec 7.4).
    /// :raises OptionsError: When 0.
    void SetMaxOptimizations(unsigned count);
    void ClearMaxOptimizations();
    const OptimizerOptions& GetOptimizerOptions() const;
    void SetOptimizerOptions(const OptimizerOptions& options);

    /// :raises OptionsError: With no dihedral, end <= start, a range above 360 degrees, a step
    ///     that does not divide the range, optimizer options that fail their own Validate() or
    ///     select LBFGS (every scan has exact constraints, spec 7.1), or embedded constraints or
    ///     restraints that already act on a scanned dihedral (the same four atoms in either
    ///     direction).
    void Validate() const;

private:
    std::vector<std::vector<unsigned>> dihedrals_;
    double step_ = 15.0;
    double range_start_ = -180.0;
    double range_end_ = 180.0;
    std::optional<unsigned> max_optimizations_;
    OptimizerOptions optimizer_;
};

/// Optimizer settings after the spec 5.4 resolution rule; ``algorithm`` is never AUTO.
struct ResolvedOptimizerSettings {
    Algorithm algorithm = Algorithm::ANC_RFO;
    unsigned max_iterations = 200;
    double energy_change = 0.0;         ///< kcal/mol
    double gradient_norm = 0.0;         ///< kcal/mol/Angstrom
    double initial_trust_radius = 0.1;  ///< Angstrom
    double max_trust_radius = 0.3;      ///< Angstrom
};

/// Explicit fields win; unset fields come from the preset table (spec 5.4).
/// :param force_field_method: Whether the method's capabilities declare ``force_field``.
/// :raises OptionsError: When the options fail ``Validate()`` or the resolved initial trust
///     radius exceeds the resolved maximum.
ResolvedOptimizerSettings resolve_optimizer_settings(const OptimizerOptions& options, Preset preset,
                                                     unsigned num_atoms, bool force_field_method);

}  // namespace OEVIBE

#endif  // OEVIBE_OPTIONS_H
