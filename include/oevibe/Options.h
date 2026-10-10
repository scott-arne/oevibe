#ifndef OEVIBE_OPTIONS_H
#define OEVIBE_OPTIONS_H

#include <map>
#include <optional>
#include <string>

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
