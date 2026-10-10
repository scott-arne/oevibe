#include "oevibe/Preset.h"

#include <algorithm>
#include <cctype>

#include "oevibe/Error.h"
#include "oevibe/Units.h"

namespace OEVIBE {

namespace {

// Python callers can pass any integer as an enum; Preset numbers its enumerators from 0.
void require_preset(Preset preset) {
    const auto raw = static_cast<long long>(preset);
    if (raw < 0 || raw > static_cast<long long>(Preset::THOROUGH)) {
        throw OptionsError("Preset must be one of its enumerators, got " + std::to_string(raw));
    }
}

// Thresholds are xtb's documented loose, normal, and tight optimization levels (spec 5.4).
PresetValues make_values(double energy_hartree, double gradient_hartree_per_bohr, double accuracy) {
    PresetValues values;
    values.energy_change = energy_hartree * Units::HARTREE_TO_KCAL;
    values.gradient_norm = gradient_hartree_per_bohr * Units::HARTREE_PER_BOHR_TO_KCAL_PER_ANGSTROM;
    values.engine_accuracy = accuracy;
    return values;
}

}  // namespace

PresetValues preset_values(Preset preset) {
    require_preset(preset);
    switch (preset) {
        case Preset::FAST:
            return make_values(5e-5, 4e-3, 2.0);
        case Preset::STANDARD:
            return make_values(5e-6, 1e-3, 1.0);
        case Preset::THOROUGH:
            return make_values(1e-6, 8e-4, 0.2);
    }
    return make_values(5e-6, 1e-3, 1.0);
}

unsigned preset_max_iterations(unsigned num_atoms) {
    return std::max(200u, std::min(10000u, 3u * num_atoms));
}

std::string preset_name(Preset preset) {
    require_preset(preset);
    switch (preset) {
        case Preset::FAST:
            return "fast";
        case Preset::STANDARD:
            return "standard";
        case Preset::THOROUGH:
            return "thorough";
    }
    return "standard";
}

Preset preset_from_name(const std::string& name) {
    std::string lowered;
    for (char c : name) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lowered == "fast") {
        return Preset::FAST;
    }
    if (lowered == "standard") {
        return Preset::STANDARD;
    }
    if (lowered == "thorough") {
        return Preset::THOROUGH;
    }
    throw OptionsError("unknown preset '" + name + "'; expected fast, standard, or thorough");
}

}  // namespace OEVIBE
