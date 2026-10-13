#include "oevibe/Options.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/ScanGrid.h"
#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

// Optional getters check ``has_value`` and throw inline instead of calling this helper:
// clang-tidy's optional-access analysis cannot see that the helper throws.
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw OptionsError(message);
    }
}

void require_positive(double value, const char* name) {
    require(std::isfinite(value) && value > 0.0,
            std::string(name) + " must be finite and greater than 0, got " + std::to_string(value));
}

template <typename Enum>
void require_enumerator(Enum value, Enum last, const char* name) {
    const auto raw = static_cast<long long>(value);
    require(raw >= 0 && raw <= static_cast<long long>(last),
            std::string(name) + " must be one of its enumerators, got " + std::to_string(raw));
}

}  // namespace

std::string algorithm_name(Algorithm algorithm) {
    require_enumerator(algorithm, Algorithm::LBFGS, "Algorithm");
    switch (algorithm) {
        case Algorithm::AUTO:
            return "auto";
        case Algorithm::ANC_RFO:
            return "anc_rfo";
        case Algorithm::LBFGS:
            return "lbfgs";
    }
    return "auto";
}

// CalculationOptions

CalculationOptions::CalculationOptions() = default;

CalculationOptions CalculationOptions::FromPreset(Preset preset) {
    CalculationOptions options;
    options.SetPreset(preset);
    return options;
}

const Method& CalculationOptions::GetMethod() const { return method_; }
void CalculationOptions::SetMethod(const Method& method) { method_ = method; }
bool CalculationOptions::HasEngine() const { return !engine_.empty(); }
const std::string& CalculationOptions::GetEngine() const {
    if (engine_.empty()) {
        throw OptionsError("no engine override is set");
    }
    return engine_;
}
void CalculationOptions::SetEngine(const std::string& engine) {
    require(!engine.empty(), "engine names must not be empty; use ClearEngine for routing");
    engine_ = engine;
}
void CalculationOptions::ClearEngine() { engine_.clear(); }
const Solvent& CalculationOptions::GetSolvent() const { return solvent_; }
void CalculationOptions::SetSolvent(const Solvent& solvent) { solvent_ = solvent; }
Preset CalculationOptions::GetPreset() const { return preset_; }
void CalculationOptions::SetPreset(Preset preset) {
    require_enumerator(preset, Preset::THOROUGH, "Preset");
    preset_ = preset;
}
bool CalculationOptions::HasCharge() const { return charge_.has_value(); }
int CalculationOptions::GetCharge() const {
    if (!charge_.has_value()) {
        throw OptionsError("no charge override is set");
    }
    return *charge_;
}
void CalculationOptions::SetCharge(int charge) { charge_ = charge; }
void CalculationOptions::ClearCharge() { charge_.reset(); }
bool CalculationOptions::HasMultiplicity() const { return multiplicity_.has_value(); }
unsigned CalculationOptions::GetMultiplicity() const {
    if (!multiplicity_.has_value()) {
        throw OptionsError("no multiplicity override is set");
    }
    return *multiplicity_;
}
void CalculationOptions::SetMultiplicity(unsigned multiplicity) {
    require(multiplicity >= 1, "Multiplicity must be at least 1");
    multiplicity_ = multiplicity;
}
void CalculationOptions::ClearMultiplicity() { multiplicity_.reset(); }
unsigned CalculationOptions::GetProperties() const { return properties_; }
void CalculationOptions::SetProperties(unsigned properties) {
    require((properties & ~ALL_PROPERTIES) == 0,
            "Properties contains unknown bits: " + std::to_string(properties & ~ALL_PROPERTIES));
    properties_ = properties;
}
unsigned CalculationOptions::GetEffectiveProperties() const {
    return assign_partial_charges_ ? (properties_ | static_cast<unsigned>(Property::CHARGES))
                                   : properties_;
}
bool CalculationOptions::GetAddHydrogens() const { return add_hydrogens_; }
void CalculationOptions::SetAddHydrogens(bool add) { add_hydrogens_ = add; }
bool CalculationOptions::GetUpdateCoordinates() const { return update_coordinates_; }
void CalculationOptions::SetUpdateCoordinates(bool update) { update_coordinates_ = update; }
bool CalculationOptions::GetWriteTags() const { return write_tags_; }
void CalculationOptions::SetWriteTags(bool write) { write_tags_ = write; }
bool CalculationOptions::GetAssignPartialCharges() const { return assign_partial_charges_; }
void CalculationOptions::SetAssignPartialCharges(bool assign) { assign_partial_charges_ = assign; }
bool CalculationOptions::GetUseInputPartialCharges() const { return use_input_partial_charges_; }
void CalculationOptions::SetUseInputPartialCharges(bool use) { use_input_partial_charges_ = use; }
void CalculationOptions::SetKeyword(const std::string& name, const std::string& value) {
    require(!name.empty(), "keyword names must not be empty");
    keywords_[name] = value;
}
bool CalculationOptions::HasKeyword(const std::string& name) const {
    return keywords_.count(name) != 0;
}
const std::string& CalculationOptions::GetKeyword(const std::string& name) const {
    const auto it = keywords_.find(name);
    if (it == keywords_.end()) {
        throw OptionsError("keyword '" + name + "' is not set");
    }
    return it->second;
}
void CalculationOptions::ClearKeyword(const std::string& name) { keywords_.erase(name); }
std::map<std::string, std::string> CalculationOptions::GetKeywords() const { return keywords_; }
void CalculationOptions::Validate() const {}

// OptimizerOptions

Algorithm OptimizerOptions::GetAlgorithm() const { return algorithm_; }
void OptimizerOptions::SetAlgorithm(Algorithm algorithm) {
    require_enumerator(algorithm, Algorithm::LBFGS, "Algorithm");
    algorithm_ = algorithm;
}
bool OptimizerOptions::HasMaxIterations() const { return max_iterations_.has_value(); }
unsigned OptimizerOptions::GetMaxIterations() const {
    if (!max_iterations_.has_value()) {
        throw OptionsError("MaxIterations is not set");
    }
    return *max_iterations_;
}
void OptimizerOptions::SetMaxIterations(unsigned iterations) {
    require(iterations >= 1, "MaxIterations must be at least 1");
    max_iterations_ = iterations;
}
void OptimizerOptions::ClearMaxIterations() { max_iterations_.reset(); }
bool OptimizerOptions::HasEnergyChangeThreshold() const { return energy_change_.has_value(); }
double OptimizerOptions::GetEnergyChangeThreshold() const {
    if (!energy_change_.has_value()) {
        throw OptionsError("EnergyChangeThreshold is not set");
    }
    return *energy_change_;
}
void OptimizerOptions::SetEnergyChangeThreshold(double threshold) {
    require_positive(threshold, "EnergyChangeThreshold");
    energy_change_ = threshold;
}
void OptimizerOptions::ClearEnergyChangeThreshold() { energy_change_.reset(); }
bool OptimizerOptions::HasGradientNormThreshold() const { return gradient_norm_.has_value(); }
double OptimizerOptions::GetGradientNormThreshold() const {
    if (!gradient_norm_.has_value()) {
        throw OptionsError("GradientNormThreshold is not set");
    }
    return *gradient_norm_;
}
void OptimizerOptions::SetGradientNormThreshold(double threshold) {
    require_positive(threshold, "GradientNormThreshold");
    gradient_norm_ = threshold;
}
void OptimizerOptions::ClearGradientNormThreshold() { gradient_norm_.reset(); }
bool OptimizerOptions::HasInitialTrustRadius() const { return initial_trust_radius_.has_value(); }
double OptimizerOptions::GetInitialTrustRadius() const {
    if (!initial_trust_radius_.has_value()) {
        throw OptionsError("InitialTrustRadius is not set");
    }
    return *initial_trust_radius_;
}
void OptimizerOptions::SetInitialTrustRadius(double radius) {
    require_positive(radius, "InitialTrustRadius");
    initial_trust_radius_ = radius;
}
void OptimizerOptions::ClearInitialTrustRadius() { initial_trust_radius_.reset(); }
bool OptimizerOptions::HasMaxTrustRadius() const { return max_trust_radius_.has_value(); }
double OptimizerOptions::GetMaxTrustRadius() const {
    if (!max_trust_radius_.has_value()) {
        throw OptionsError("MaxTrustRadius is not set");
    }
    return *max_trust_radius_;
}
void OptimizerOptions::SetMaxTrustRadius(double radius) {
    require_positive(radius, "MaxTrustRadius");
    max_trust_radius_ = radius;
}
void OptimizerOptions::ClearMaxTrustRadius() { max_trust_radius_.reset(); }
const ConstraintSet& OptimizerOptions::GetConstraints() const { return constraints_; }
void OptimizerOptions::SetConstraints(const ConstraintSet& constraints) {
    constraints_ = constraints;
}

void OptimizerOptions::Validate() const {
    require(!(algorithm_ == Algorithm::LBFGS && constraints_.HasExactConstraints()),
            "LBFGS does not support exact constraints; use ANC_RFO or AUTO");
    if (initial_trust_radius_.has_value() && max_trust_radius_.has_value()) {
        require(*initial_trust_radius_ <= *max_trust_radius_,
                "InitialTrustRadius must not exceed MaxTrustRadius");
    }
}

// ThermoOptions

double ThermoOptions::GetTemperature() const { return temperature_; }
void ThermoOptions::SetTemperature(double kelvin) {
    require_positive(kelvin, "Temperature");
    temperature_ = kelvin;
}
double ThermoOptions::GetPressure() const { return pressure_; }
void ThermoOptions::SetPressure(double atm) {
    require_positive(atm, "Pressure");
    pressure_ = atm;
}
double ThermoOptions::GetQuasiRRHOCutoff() const { return quasi_rrho_cutoff_; }
void ThermoOptions::SetQuasiRRHOCutoff(double cm) {
    require_positive(cm, "QuasiRRHOCutoff");
    quasi_rrho_cutoff_ = cm;
}
unsigned ThermoOptions::GetSymmetryNumber() const { return symmetry_number_; }
void ThermoOptions::SetSymmetryNumber(unsigned n) {
    require(n >= 1, "SymmetryNumber must be at least 1");
    symmetry_number_ = n;
}
bool ThermoOptions::GetOptimizeFirst() const { return optimize_first_; }
void ThermoOptions::SetOptimizeFirst(bool optimize) { optimize_first_ = optimize; }
const OptimizerOptions& ThermoOptions::GetOptimizerOptions() const { return optimizer_; }
void ThermoOptions::SetOptimizerOptions(const OptimizerOptions& options) { optimizer_ = options; }

void ThermoOptions::Validate() const {
    optimizer_.Validate();
    require(optimizer_.GetConstraints().IsEmpty(),
            "frequencies need an unconstrained stationary point, so the optimizer options must "
            "not freeze atoms or hold constraints or restraints");
}

ResolvedOptimizerSettings resolve_optimizer_settings(const OptimizerOptions& options, Preset preset,
                                                     unsigned num_atoms, bool force_field_method) {
    options.Validate();
    const PresetValues values = preset_values(preset);
    ResolvedOptimizerSettings resolved;
    const bool exact = options.GetConstraints().HasExactConstraints();
    switch (options.GetAlgorithm()) {
        case Algorithm::AUTO:
            resolved.algorithm =
                force_field_method && !exact ? Algorithm::LBFGS : Algorithm::ANC_RFO;
            break;
        case Algorithm::ANC_RFO:
        case Algorithm::LBFGS:
            resolved.algorithm = options.GetAlgorithm();
            break;
    }
    resolved.max_iterations =
        options.HasMaxIterations() ? options.GetMaxIterations() : preset_max_iterations(num_atoms);
    resolved.energy_change = options.HasEnergyChangeThreshold() ? options.GetEnergyChangeThreshold()
                                                                : values.energy_change;
    resolved.gradient_norm = options.HasGradientNormThreshold() ? options.GetGradientNormThreshold()
                                                                : values.gradient_norm;
    resolved.initial_trust_radius = options.HasInitialTrustRadius()
                                        ? options.GetInitialTrustRadius()
                                        : values.initial_trust_radius;
    resolved.max_trust_radius =
        options.HasMaxTrustRadius() ? options.GetMaxTrustRadius() : values.max_trust_radius;
    require(resolved.initial_trust_radius <= resolved.max_trust_radius,
            "the resolved initial trust radius exceeds the resolved maximum trust radius");
    return resolved;
}

// ScanOptions

namespace {

bool same_dihedral(const std::vector<unsigned>& atoms, const std::vector<unsigned>& dihedral) {
    return atoms.size() == 4 && (std::equal(atoms.begin(), atoms.end(), dihedral.begin()) ||
                                 std::equal(atoms.begin(), atoms.end(), dihedral.rbegin()));
}

std::string dihedral_text(const std::vector<unsigned>& dihedral) {
    return std::to_string(dihedral[0]) + "-" + std::to_string(dihedral[1]) + "-" +
           std::to_string(dihedral[2]) + "-" + std::to_string(dihedral[3]);
}

}  // namespace

void ScanOptions::AddDihedral(unsigned a, unsigned b, unsigned c, unsigned d) {
    const std::vector<unsigned> dihedral = {a, b, c, d};
    for (std::size_t i = 0; i < dihedral.size(); ++i) {
        for (std::size_t j = i + 1; j < dihedral.size(); ++j) {
            require(dihedral[i] != dihedral[j],
                    "scanned dihedral atoms must be distinct, got " + dihedral_text(dihedral));
        }
    }
    for (const std::vector<unsigned>& existing : dihedrals_) {
        require(!same_dihedral(existing, dihedral),
                "dihedral " + dihedral_text(dihedral) + " is already scanned");
    }
    require(dihedrals_.size() < 2, "a scan has at most two dihedrals");
    dihedrals_.push_back(dihedral);
}
unsigned ScanOptions::GetNumDihedrals() const { return static_cast<unsigned>(dihedrals_.size()); }
std::vector<unsigned> ScanOptions::GetDihedral(unsigned index) const {
    require(index < dihedrals_.size(), "dihedral index " + std::to_string(index) +
                                           " is out of range for " +
                                           std::to_string(dihedrals_.size()) + " dihedrals");
    return dihedrals_[index];
}
void ScanOptions::ClearDihedrals() { dihedrals_.clear(); }
double ScanOptions::GetStep() const { return step_; }
void ScanOptions::SetStep(double degrees) {
    require_positive(degrees, "Step");
    step_ = degrees;
}
double ScanOptions::GetRangeStart() const { return range_start_; }
double ScanOptions::GetRangeEnd() const { return range_end_; }
void ScanOptions::SetRange(double start, double end) {
    require(std::isfinite(start) && std::isfinite(end), "Range ends must be finite");
    range_start_ = start;
    range_end_ = end;
}
bool ScanOptions::HasMaxOptimizations() const { return max_optimizations_.has_value(); }
unsigned ScanOptions::GetMaxOptimizations() const {
    if (!max_optimizations_.has_value()) {
        throw OptionsError("MaxOptimizations is not set");
    }
    return *max_optimizations_;
}
void ScanOptions::SetMaxOptimizations(unsigned count) {
    require(count >= 1, "MaxOptimizations must be at least 1");
    max_optimizations_ = count;
}
void ScanOptions::ClearMaxOptimizations() { max_optimizations_.reset(); }
const OptimizerOptions& ScanOptions::GetOptimizerOptions() const { return optimizer_; }
void ScanOptions::SetOptimizerOptions(const OptimizerOptions& options) { optimizer_ = options; }

void ScanOptions::Validate() const {
    require(!dihedrals_.empty(), "a scan needs at least one dihedral");
    // Every dihedral shares the axis, so the grid holds its size to the power of their number;
    // dividing first keeps the product from overflowing before it is compared.
    const std::size_t axis_points =
        detail::build_scan_axis(range_start_, range_end_, step_).values.size();
    std::size_t grid_points = 1;
    for (std::size_t k = 0; k < dihedrals_.size(); ++k) {
        require(axis_points <= detail::MAX_SCAN_GRID_POINTS / grid_points,
                "the scan grid would exceed " + std::to_string(detail::MAX_SCAN_GRID_POINTS) +
                    " points; use a larger step or a smaller range");
        grid_points *= axis_points;
    }
    optimizer_.Validate();
    require(optimizer_.GetAlgorithm() != Algorithm::LBFGS,
            "LBFGS does not support exact constraints, and a scan constrains its dihedrals; use "
            "ANC_RFO or AUTO");
    const ConstraintSet& constraints = optimizer_.GetConstraints();
    for (const std::vector<unsigned>& dihedral : dihedrals_) {
        for (const Constraint& constraint : constraints.GetConstraints()) {
            require(!(constraint.kind == ConstraintKind::DIHEDRAL &&
                      same_dihedral(constraint.atoms, dihedral)),
                    "the optimizer options already constrain scanned dihedral " +
                        dihedral_text(dihedral));
        }
        for (const Restraint& restraint : constraints.GetRestraints()) {
            require(!(restraint.kind == RestraintKind::DIHEDRAL &&
                      same_dihedral(restraint.atoms, dihedral)),
                    "the optimizer options already restrain scanned dihedral " +
                        dihedral_text(dihedral));
        }
    }
}

}  // namespace OEVIBE
