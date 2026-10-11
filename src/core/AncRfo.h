#ifndef OEVIBE_CORE_ANCRFO_H
#define OEVIBE_CORE_ANCRFO_H

#include <Eigen/Dense>

#include <vector>

#include "core/ConstraintMath.h"
#include "core/Minimizer.h"

namespace OEVIBE {
namespace detail {

struct AncRfoProblem {
    std::vector<unsigned> atomic_numbers;  ///< Dense order, for the model Hessian.
    std::vector<unsigned> frozen;          ///< Dense indices.
    std::vector<ResolvedConstraint> constraints;
    /// False when the objective changes under rigid-body motion (a position restraint), so the
    /// coordinates must keep translations and rotations even with no atom frozen.
    bool rigid_invariant = true;
};

struct RfoStep {
    Eigen::VectorXd step;
    double predicted_change = 0.0;  ///< Quadratic-model energy change, kcal/mol.
};

/// Rational function optimization step (Banerjee et al., J. Phys. Chem. 89, 52, 1985), scaled
/// down to ``trust_radius`` when longer.
RfoStep rfo_step(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& gradient,
                 double trust_radius);

/// BFGS Hessian update; skipped when the curvature condition fails, keeping the model positive.
void bfgs_update(Eigen::MatrixXd& hessian, const Eigen::VectorXd& step,
                 const Eigen::VectorXd& gradient_change);

/// Minimizes in approximate normal coordinates with RFO steps, BFGS updates, a trust radius,
/// frozen atoms, and exact constraints (spec 7.1).
MinimizerResult anc_rfo_minimize(const ObjectiveFn& objective, std::vector<double> x0,
                                 const AncRfoProblem& problem, const MinimizerSettings& settings);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_ANCRFO_H
