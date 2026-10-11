#ifndef OEVIBE_CORE_ANCCOORDINATES_H
#define OEVIBE_CORE_ANCCOORDINATES_H

#include <Eigen/Dense>

#include <vector>

namespace OEVIBE {
namespace detail {

/// Smallest model curvature kept, so soft modes cannot produce huge first steps.
constexpr double MIN_ANC_EIGENVALUE = 2.0;  // kcal/mol/Angstrom^2

/// Approximate normal coordinates (spec 7.1).
struct AncBasis {
    Eigen::MatrixXd vectors;      ///< 3N x m with orthonormal columns.
    Eigen::VectorXd eigenvalues;  ///< m model curvatures, at least MIN_ANC_EIGENVALUE.
};

/// Diagonalizes the model Hessian in the subspace orthogonal to rigid-body motion (when no atom
/// is frozen) or to the frozen atoms' displacements. Rigid rotation is kept when atoms are
/// frozen because frozen atoms break rotational invariance.
AncBasis build_anc_basis(const Eigen::MatrixXd& hessian, const std::vector<double>& x,
                         const std::vector<unsigned>& frozen);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_ANCCOORDINATES_H
