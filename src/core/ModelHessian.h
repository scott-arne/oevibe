#ifndef OEVIBE_CORE_MODELHESSIAN_H
#define OEVIBE_CORE_MODELHESSIAN_H

#include <Eigen/Dense>

#include <vector>

namespace OEVIBE {
namespace detail {

/// Lindh model Hessian (Lindh et al., Chem. Phys. Lett. 241, 423, 1995) for dense coordinates
/// in Angstrom, returned in kcal/mol/Angstrom^2. Stretches, bends, and torsions over all atoms
/// are weighted by exponential distance factors, so no connectivity is needed.
Eigen::MatrixXd lindh_model_hessian(const std::vector<unsigned>& atomic_numbers,
                                    const std::vector<double>& x);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_MODELHESSIAN_H
