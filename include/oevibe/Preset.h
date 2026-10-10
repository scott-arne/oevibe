#ifndef OEVIBE_PRESET_H
#define OEVIBE_PRESET_H

#include <string>

namespace OEVIBE {

// Presets hold a handful of these per calculation, so a narrower underlying type saves nothing.
// NOLINTBEGIN(performance-enum-size)
/// Accuracy and effort level owned by ``CalculationOptions`` (spec 5.4).
enum class Preset { FAST, STANDARD, THOROUGH };
// NOLINTEND(performance-enum-size)

/// One column of the spec 5.4 preset table.
struct PresetValues {
    double energy_change = 0.0;         ///< kcal/mol
    double gradient_norm = 0.0;         ///< kcal/mol/Angstrom, projected gradient
    double initial_trust_radius = 0.1;  ///< Angstrom
    double max_trust_radius = 0.3;      ///< Angstrom
    double fd_step = 0.005;             ///< Angstrom, finite-difference Hessian step
    double engine_accuracy = 1.0;       ///< tblite and xtb ``accuracy`` units
};

/// :raises OptionsError: For a value outside the enumerators.
PresetValues preset_values(Preset preset);

/// :returns: ``max(200, min(10000, 3N))`` for every preset (spec 5.4).
unsigned preset_max_iterations(unsigned num_atoms);

/// :returns: ``fast``, ``standard``, or ``thorough``.
/// :raises OptionsError: For a value outside the enumerators.
std::string preset_name(Preset preset);

/// :raises OptionsError: For a name other than fast, standard, or thorough (case-insensitive).
Preset preset_from_name(const std::string& name);

}  // namespace OEVIBE

#endif  // OEVIBE_PRESET_H
