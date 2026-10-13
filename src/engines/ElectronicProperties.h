#ifndef OEVIBE_ENGINES_ELECTRONICPROPERTIES_H
#define OEVIBE_ENGINES_ELECTRONICPROPERTIES_H

#include <vector>

#include "oevibe/Property.h"

namespace OEVIBE {
namespace detail {

/// An orbital is occupied when its spin-summed occupation exceeds half an electron, so the
/// singly occupied orbital of a radical is its HOMO while a Fermi-smearing tail is not.
constexpr double HOMO_OCCUPATION_THRESHOLD = 0.5;  // electrons, spin-summed
/// Pairs below this order are omitted, the cutoff of xtb's bond-order printout.
constexpr double WIBERG_REPORT_THRESHOLD = 0.1;

/// :param spin_summed_occupations: Electrons per orbital, orbitals in ascending energy order.
/// :returns: The highest index whose spin-summed occupation exceeds 0.5, or -1.
int homo_index(const std::vector<double>& spin_summed_occupations);

/// :returns: (i, j, order) for i < j with order >= WIBERG_REPORT_THRESHOLD from a row-major
///     num_atoms x num_atoms matrix, in dense indices. The upper triangle is read. A non-finite
///     order is kept so that evaluation validation reports it instead of hiding it.
/// :raises std::invalid_argument: When the matrix does not hold num_atoms^2 values.
std::vector<BondOrder> wiberg_pairs(const std::vector<double>& matrix, unsigned num_atoms);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_ELECTRONICPROPERTIES_H
