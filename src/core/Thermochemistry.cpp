#include "core/Thermochemistry.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "core/Frequencies.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"

namespace OEVIBE {
namespace detail {

namespace {

// CODATA 2018 values; all but the atomic mass constant are exact in the SI.
constexpr double PLANCK = 6.62607015e-34;            // J s
constexpr double BOLTZMANN = 1.380649e-23;           // J/K
constexpr double SPEED_OF_LIGHT_CM = 2.99792458e10;  // cm/s
constexpr double ATOMIC_MASS = 1.66053906660e-27;    // kg
constexpr double PASCALS_PER_ATM = 101325.0;
constexpr double SQUARE_METRES_PER_SQUARE_ANGSTROM = 1e-20;
/// R in cal/(mol K), from the spec 3.1 value in kcal/(mol K).
constexpr double GAS_CONSTANT_CAL = Units::GAS_CONSTANT_KCAL * 1000.0;
/// The damping exponent alpha of Grimme's weight 1 / (1 + (nu0 / nu)^alpha).
constexpr double QUASI_RRHO_EXPONENT = 4.0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw OEVibeError(message);
    }
}

void require_positive(double value, const char* name) {
    require(std::isfinite(value) && value > 0.0,
            std::string(name) + " must be finite and greater than 0, got " + std::to_string(value));
}

/// h^2 / (8 pi^2 I k) in K for a moment in amu Angstrom^2.
double rotational_temperature(double moment) {
    const double moment_si = moment * ATOMIC_MASS * SQUARE_METRES_PER_SQUARE_ANGSTROM;
    return PLANCK * PLANCK / (8.0 * Units::PI * Units::PI * moment_si * BOLTZMANN);
}

/// Harmonic-oscillator entropy of one mode in units of R, x = h nu / (k T).
double harmonic_entropy(double x) { return x / std::expm1(x) - std::log1p(-std::exp(-x)); }

/// Free-rotor entropy in units of R of a mode of ``wavenumber`` cm^-1 (Grimme 2012, eqs. 4-6):
/// mu = h / (8 pi^2 nu), mu' = mu B_av / (mu + B_av), S_R = 1/2 + ln sqrt(8 pi^3 mu' k T / h^2).
double free_rotor_entropy(double wavenumber, double temperature) {
    const double nu = SPEED_OF_LIGHT_CM * wavenumber;  // s^-1
    const double moment = PLANCK / (8.0 * Units::PI * Units::PI * nu);
    const double effective =
        moment * QUASI_RRHO_AVERAGE_MOMENT / (moment + QUASI_RRHO_AVERAGE_MOMENT);
    return 0.5 + std::log(std::sqrt(8.0 * Units::PI * Units::PI * Units::PI * effective *
                                    BOLTZMANN * temperature / (PLANCK * PLANCK)));
}

}  // namespace

Thermochemistry compute_thermochemistry(double electronic_energy,
                                        const std::vector<double>& frequencies,
                                        const std::vector<double>& principal_moments,
                                        double total_mass, unsigned symmetry_number, bool linear,
                                        unsigned multiplicity, double temperature, double pressure,
                                        double quasi_rrho_cutoff) {
    require(std::isfinite(electronic_energy), "the electronic energy must be finite");
    require_positive(temperature, "temperature");
    require_positive(pressure, "pressure");
    require_positive(quasi_rrho_cutoff, "the quasi-RRHO cutoff");
    require_positive(total_mass, "the total mass");
    require(symmetry_number >= 1, "the symmetry number must be at least 1");
    require(multiplicity >= 1, "the multiplicity must be at least 1");
    require(principal_moments.size() == 3 &&
                std::all_of(principal_moments.begin(), principal_moments.end(),
                            [](double m) { return std::isfinite(m) && m >= 0.0; }),
            "three finite, non-negative principal moments are required");
    require(std::all_of(frequencies.begin(), frequencies.end(),
                        [](double f) { return std::isfinite(f); }),
            "every frequency must be finite");
    const double smallest = *std::min_element(principal_moments.begin(), principal_moments.end());
    const double largest = *std::max_element(principal_moments.begin(), principal_moments.end());
    const bool atom = largest < LINEAR_MOMENT_THRESHOLD;
    require(!(linear && atom), "a linear molecule needs a principal moment of at least " +
                                   std::to_string(LINEAR_MOMENT_THRESHOLD) + " amu Angstrom^2");
    require(linear || atom || smallest >= LINEAR_MOMENT_THRESHOLD,
            "a nonlinear molecule needs three principal moments of at least " +
                std::to_string(LINEAR_MOMENT_THRESHOLD) + " amu Angstrom^2");

    Thermochemistry thermo;
    thermo.temperature = temperature;
    thermo.pressure = pressure;
    thermo.electronic_energy = electronic_energy;
    thermo.frequencies = frequencies;
    thermo.principal_moments = principal_moments;
    thermo.total_mass = total_mass;
    thermo.symmetry_number = symmetry_number;
    thermo.linear = linear;
    thermo.multiplicity = multiplicity;
    thermo.quasi_rrho_cutoff = quasi_rrho_cutoff;

    const double rt = Units::GAS_CONSTANT_KCAL * temperature;  // kcal/mol
    const double kt = BOLTZMANN * temperature;                 // J
    const double sigma = static_cast<double>(symmetry_number);

    // Translation: Sackur-Tetrode, S = R (ln[(2 pi m k T / h^2)^(3/2) k T / P] + 5/2).
    const double mass = total_mass * ATOMIC_MASS;
    const double translational = std::pow(2.0 * Units::PI * mass * kt / (PLANCK * PLANCK), 1.5) *
                                 kt / (pressure * PASCALS_PER_ATM);
    double entropy = GAS_CONSTANT_CAL * (std::log(translational) + 2.5);
    double thermal = 1.5 * rt;

    // Rotation: classical rigid rotor; a single atom has none.
    if (linear) {
        const double theta = rotational_temperature(largest);
        entropy += GAS_CONSTANT_CAL * (std::log(temperature / (sigma * theta)) + 1.0);
        thermal += rt;
    } else if (!atom) {
        const double product = rotational_temperature(principal_moments[0]) *
                               rotational_temperature(principal_moments[1]) *
                               rotational_temperature(principal_moments[2]);
        const double rotational = std::sqrt(Units::PI) / sigma *
                                  std::sqrt(temperature * temperature * temperature / product);
        entropy += GAS_CONSTANT_CAL * (std::log(rotational) + 1.5);
        thermal += 1.5 * rt;
    }

    // Vibration: harmonic energies for every real mode; quasi-RRHO entropy.
    const double second_radiation = PLANCK * SPEED_OF_LIGHT_CM / BOLTZMANN;  // cm K
    double zpe = 0.0;
    for (double wavenumber : frequencies) {
        // Imaginary modes (and a zero frequency) are not vibrations of a minimum (spec 7.3).
        if (!(wavenumber > 0.0)) {
            continue;
        }
        const double theta = second_radiation * wavenumber;  // K
        const double x = theta / temperature;
        zpe += 0.5 * Units::GAS_CONSTANT_KCAL * theta;
        thermal += Units::GAS_CONSTANT_KCAL * theta / std::expm1(x);
        const double weight =
            1.0 / (1.0 + std::pow(quasi_rrho_cutoff / wavenumber, QUASI_RRHO_EXPONENT));
        double mode_entropy = (1.0 - weight) * free_rotor_entropy(wavenumber, temperature);
        // For a nearly zero frequency the weight underflows to 0 while the harmonic entropy
        // diverges, and 0 * inf would be NaN.
        if (weight > 0.0) {
            mode_entropy += weight * harmonic_entropy(x);
        }
        entropy += GAS_CONSTANT_CAL * mode_entropy;
    }

    entropy += GAS_CONSTANT_CAL * std::log(static_cast<double>(multiplicity));

    thermo.zpe = zpe;
    thermo.enthalpy = electronic_energy + zpe + thermal + rt;  // + RT: ideal-gas PV
    thermo.entropy = entropy;
    thermo.free_energy = thermo.enthalpy - temperature * entropy / 1000.0;
    thermo.free_energy_correction = thermo.free_energy - electronic_energy;
    return thermo;
}

Thermochemistry recompute_thermochemistry(const Thermochemistry& stored, double temperature) {
    return compute_thermochemistry(stored.electronic_energy, stored.frequencies,
                                   stored.principal_moments, stored.total_mass,
                                   stored.symmetry_number, stored.linear, stored.multiplicity,
                                   temperature, stored.pressure, stored.quasi_rrho_cutoff);
}

}  // namespace detail
}  // namespace OEVIBE
