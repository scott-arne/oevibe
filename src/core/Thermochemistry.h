#ifndef OEVIBE_CORE_THERMOCHEMISTRY_H
#define OEVIBE_CORE_THERMOCHEMISTRY_H

#include <vector>

#include "oevibe/Result.h"

namespace OEVIBE {
namespace detail {

/// Grimme's average moment of inertia for the quasi-RRHO free rotor (Chem. Eur. J. 18, 9955,
/// 2012), in kg m^2.
constexpr double QUASI_RRHO_AVERAGE_MOMENT = 1e-44;

/// Thermochemistry of an ideal gas of rigid rotors with harmonic vibrations (spec 7.3):
/// Sackur-Tetrode translation at ``pressure``; rigid-rotor rotation with ``symmetry_number``
/// (none for a single atom, recognized by its largest moment being below
/// ``LINEAR_MOMENT_THRESHOLD``); harmonic ZPE and vibrational energy over the real modes
/// (frequencies > 0; imaginary modes count nowhere); Grimme's quasi-RRHO vibrational entropy
/// with ``quasi_rrho_cutoff`` as nu0; electronic entropy R ln(multiplicity).
/// :param electronic_energy: kcal/mol.
/// :param frequencies: cm^-1, imaginary modes negative; stored unchanged.
/// :param principal_moments: Three non-negative values in amu Angstrom^2. A linear molecule uses
///     the largest; a nonlinear one needs all three of at least ``LINEAR_MOMENT_THRESHOLD``.
/// :param total_mass: amu.
/// :raises OEVibeError: Unless temperature, pressure, cutoff, and mass are finite and > 0, or
///     when the energy or a frequency is not finite, the symmetry number or multiplicity is 0,
///     or the moments do not fit ``linear`` as described above.
Thermochemistry compute_thermochemistry(double electronic_energy,
                                        const std::vector<double>& frequencies,
                                        const std::vector<double>& principal_moments,
                                        double total_mass, unsigned symmetry_number, bool linear,
                                        unsigned multiplicity, double temperature, double pressure,
                                        double quasi_rrho_cutoff);

/// The same inputs at another temperature, keeping the stored pressure (spec 5.7).
/// :raises OEVibeError: As ``compute_thermochemistry``.
Thermochemistry recompute_thermochemistry(const Thermochemistry& stored, double temperature);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_THERMOCHEMISTRY_H
