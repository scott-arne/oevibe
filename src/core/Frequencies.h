#ifndef OEVIBE_CORE_FREQUENCIES_H
#define OEVIBE_CORE_FREQUENCIES_H

#include <vector>

namespace OEVIBE {
namespace detail {

constexpr double LINEAR_MOMENT_THRESHOLD = 1e-3;  // amu Angstrom^2 (spec 7.3)

/// Turns sqrt(lambda), for a mass-weighted Hessian eigenvalue lambda in
/// kcal/mol/Angstrom^2/amu, into a wavenumber in cm^-1: sqrt(4184 / (N_A 1e-20 m_u)) / (2 pi c)
/// with the CODATA 2018 values N_A = 6.02214076e23 /mol, m_u = 1.66053906660e-27 kg, and
/// c = 2.99792458e10 cm/s (1 kcal = 4184 J). C++17 has no constexpr square root, so the value is
/// written out; a test recomputes it.
constexpr double FREQUENCY_FACTOR = 108.59135861114092;

struct FrequencyAnalysis {
    /// cm^-1, ascending; 3N-6 (3N-5 if linear) vibrational modes; imaginary negative.
    std::vector<double> frequencies;
    /// modes x 3N row-major; Cartesian displacements, unit norm, dense order. Each mode's overall
    /// sign is arbitrary.
    std::vector<double> normal_modes;
    unsigned imaginary_count = 0;
    bool linear = false;
    std::vector<double> principal_moments;  ///< 3 values, ascending, amu Angstrom^2
    double total_mass = 0.0;                ///< amu
};

/// :returns: Abundance-averaged standard atomic weights in amu (``OEChem::OEGetAverageWeight``),
///     which is what ``xtb --hess`` uses.
/// :raises OEVibeError: For an atomic number without a weight.
std::vector<double> atomic_masses(const std::vector<unsigned>& atomic_numbers);

/// Harmonic analysis of a Cartesian Hessian (spec 7.3). The Hessian is mass-weighted, the
/// translations and rotations about the centre of mass are projected out, and the remaining
/// space is diagonalized, so the 3N-6 (3N-5) vibrational modes are selected by being orthogonal
/// to rigid-body motion rather than by eigenvalue size. A molecule is linear when it has at least
/// two atoms and its smallest principal moment is below ``LINEAR_MOMENT_THRESHOLD``; a single
/// atom is not linear and has no modes.
/// :param hessian: 9N^2 values, row-major, kcal/mol/Angstrom^2, dense order.
/// :param x: Dense 3N coordinates in Angstrom.
/// :raises OEVibeError: For inconsistent sizes, values that are not finite, or an unknown element.
FrequencyAnalysis analyze_frequencies(const std::vector<double>& hessian,
                                      const std::vector<double>& x,
                                      const std::vector<unsigned>& atomic_numbers);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_FREQUENCIES_H
