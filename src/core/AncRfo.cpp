#include "core/AncRfo.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "core/AncCoordinates.h"
#include "core/ModelHessian.h"

namespace OEVIBE {
namespace detail {

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double MIN_TRUST_RADIUS = 1e-4;       // Angstrom
constexpr unsigned REBUILD_INTERVAL = 20;       // accepted steps between basis rebuilds
constexpr double MAX_CORRECTION = 0.3;          // Angstrom, constraint correction per step
constexpr double ENERGY_RISE_TOLERANCE = 1e-6;  // kcal/mol of numerical noise
constexpr unsigned MAX_CONSECUTIVE_REJECTIONS = 10;

VectorXd as_vector(const std::vector<double>& values) {
    return Eigen::Map<const VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
}

/// Constraints linearized at one geometry, in ANC coordinates.
struct ConstraintModel {
    MatrixXd tangent;     ///< m x (m - rank): directions that keep constraints to first order.
    VectorXd correction;  ///< m: minimum-norm step toward the constraint surface.
    MatrixXd jacobian;    ///< k x 3N Cartesian with frozen atoms zeroed; empty if not built.
};

ConstraintModel linearize(const std::vector<ResolvedConstraint>& constraints,
                          const std::vector<double>& x, const std::vector<unsigned>& frozen,
                          const MatrixXd& basis) {
    const Eigen::Index m = basis.cols();
    ConstraintModel model;
    // With no free coordinates the SVD below would run on an empty matrix.
    if (constraints.empty() || m == 0) {
        model.tangent = MatrixXd::Identity(m, m);
        model.correction = VectorXd::Zero(m);
        return model;
    }
    const auto k = static_cast<Eigen::Index>(constraints.size());
    const auto dim = static_cast<Eigen::Index>(x.size());
    MatrixXd jacobian(k, dim);
    VectorXd residual(k);
    std::vector<double> row;
    for (Eigen::Index r = 0; r < k; ++r) {
        residual(r) = constraint_residual(constraints[static_cast<std::size_t>(r)], x, &row);
        zero_frozen(row, frozen);
        jacobian.row(r) = as_vector(row).transpose();
    }
    const MatrixXd projected = jacobian * basis;
    const Eigen::JacobiSVD<MatrixXd> svd(projected, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const VectorXd& singular = svd.singularValues();
    const double cutoff = 1e-8 * std::max(1.0, singular.size() > 0 ? singular(0) : 0.0);
    Eigen::Index rank = 0;
    while (rank < singular.size() && singular(rank) > cutoff) {
        ++rank;
    }
    model.tangent = svd.matrixV().rightCols(m - rank);
    VectorXd correction = VectorXd::Zero(m);
    for (Eigen::Index i = 0; i < rank; ++i) {
        correction -= svd.matrixV().col(i) * (svd.matrixU().col(i).dot(residual) / singular(i));
    }
    const double norm = correction.norm();
    if (norm > MAX_CORRECTION) {
        correction *= MAX_CORRECTION / norm;
    }
    model.correction = correction;
    model.jacobian = std::move(jacobian);
    return model;
}

/// Change of the Lagrangian gradient in ANC coordinates between two geometries, with the
/// least-squares multipliers taken at the new one (Baker, J. Comput. Chem. 13, 240, 1992).
VectorXd lagrangian_gradient_change(const MatrixXd& basis, const VectorXd& old_gradient,
                                    const VectorXd& new_gradient, const MatrixXd& old_jacobian,
                                    const MatrixXd& new_jacobian) {
    const VectorXd new_gradient_anc = basis.transpose() * new_gradient;
    const MatrixXd constraint_gradients = (new_jacobian * basis).transpose();  // m x k
    const VectorXd multipliers =
        constraint_gradients.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
            .solve(new_gradient_anc);
    return basis.transpose() * (new_gradient - old_gradient) -
           basis.transpose() * ((new_jacobian - old_jacobian).transpose() * multipliers);
}

}  // namespace

RfoStep rfo_step(const MatrixXd& hessian, const VectorXd& gradient, double trust_radius) {
    const Eigen::Index m = gradient.size();
    RfoStep result;
    if (m == 0) {
        result.step = VectorXd(0);
        return result;
    }
    MatrixXd augmented = MatrixXd::Zero(m + 1, m + 1);
    augmented.topLeftCorner(m, m) = hessian;
    augmented.topRightCorner(m, 1) = gradient;
    augmented.bottomLeftCorner(1, m) = gradient.transpose();
    const Eigen::SelfAdjointEigenSolver<MatrixXd> solver(augmented);
    const VectorXd lowest = solver.eigenvectors().col(0);
    VectorXd step =
        std::abs(lowest(m)) > 1e-10 ? VectorXd(lowest.head(m) / lowest(m)) : VectorXd(-gradient);
    const double norm = step.norm();
    if (norm > trust_radius && norm > 0.0) {
        step *= trust_radius / norm;
    }
    result.predicted_change = gradient.dot(step) + 0.5 * step.dot(hessian * step);
    result.step = step;
    return result;
}

void bfgs_update(MatrixXd& hessian, const VectorXd& step, const VectorXd& gradient_change) {
    const double sy = step.dot(gradient_change);
    if (sy <= 1e-10) {
        return;
    }
    const VectorXd hs = hessian * step;
    const double shs = step.dot(hs);
    if (shs <= 1e-12) {
        return;
    }
    hessian += (gradient_change * gradient_change.transpose()) / sy - (hs * hs.transpose()) / shs;
}

MinimizerResult anc_rfo_minimize(const ObjectiveFn& objective, std::vector<double> x,
                                 const AncRfoProblem& problem, const MinimizerSettings& settings) {
    MinimizerResult result;
    std::vector<double> gradient(x.size(), 0.0);
    double energy = objective(x, gradient);
    result.evaluations = 1;
    zero_frozen(gradient, problem.frozen);

    AncBasis basis;
    MatrixXd hessian;
    unsigned since_rebuild = 0;
    const auto rebuild = [&]() {
        basis = build_anc_basis(lindh_model_hessian(problem.atomic_numbers, x), x, problem.frozen,
                                problem.rigid_invariant);
        hessian = basis.eigenvalues.asDiagonal();
        since_rebuild = 0;
    };
    rebuild();

    const bool constrained = !problem.constraints.empty();
    double trust = settings.initial_trust_radius;
    unsigned rejections = 0;
    unsigned iteration = 0;
    bool stepped = false;
    bool fresh = true;  // x has not been reported to the observer yet
    double last_change = 0.0;
    while (true) {
        const VectorXd g_anc = basis.vectors.transpose() * as_vector(gradient);
        const ConstraintModel model =
            linearize(problem.constraints, x, problem.frozen, basis.vectors);
        const VectorXd projected = model.tangent * (model.tangent.transpose() * g_anc);
        const ResidualSummary residuals = summarize_residuals(problem.constraints, x);
        const bool residuals_ok = residuals_satisfied(residuals);
        result.gradient_norm = projected.norm();
        result.max_gradient_component =
            basis.vectors.cols() > 0 ? (basis.vectors * projected).cwiseAbs().maxCoeff() : 0.0;
        result.max_distance_residual = residuals.max_distance;
        result.max_angle_residual = residuals.max_angle;
        if (fresh && settings.observer) {
            MinimizerResult iterate = result;  // carries the gradient norms and residuals above
            iterate.x = x;
            iterate.energy = energy;
            iterate.iterations = iteration;
            iterate.energy_change = last_change;
            settings.observer(iterate);
        }
        fresh = false;
        const bool energy_ok = !stepped || std::abs(last_change) < settings.energy_change;
        if (result.gradient_norm < settings.gradient_norm && energy_ok && residuals_ok) {
            result.converged = true;
            break;
        }
        if (basis.vectors.cols() == 0) {
            result.message = constrained && !residuals_ok
                                 ? "no free coordinates remain to satisfy the constraints"
                                 : "no free coordinates remain";
            break;
        }
        if (iteration >= settings.max_iterations) {
            result.message = "reached the iteration limit";
            break;
        }
        ++iteration;

        const MatrixXd reduced_hessian = model.tangent.transpose() * hessian * model.tangent;
        const VectorXd reduced_gradient = model.tangent.transpose() * g_anc;
        const RfoStep tangent_step = rfo_step(reduced_hessian, reduced_gradient, trust);
        const VectorXd step = model.correction + model.tangent * tangent_step.step;
        const double predicted = g_anc.dot(step) + 0.5 * step.dot(hessian * step);

        std::vector<double> trial(x);
        Eigen::Map<VectorXd>(trial.data(), static_cast<Eigen::Index>(trial.size())) +=
            basis.vectors * step;
        std::vector<double> trial_gradient(x.size(), 0.0);
        const double trial_energy = objective(trial, trial_gradient);
        ++result.evaluations;
        zero_frozen(trial_gradient, problem.frozen);
        const double change = trial_energy - energy;

        // While constraints are violated, correction steps may legitimately raise the energy.
        const bool accept = (constrained && !residuals_ok) || change <= ENERGY_RISE_TOLERANCE;
        if (!accept) {
            trust = std::max(0.25 * trust, MIN_TRUST_RADIUS);
            if (++rejections >= MAX_CONSECUTIVE_REJECTIONS) {
                result.message = "steps were rejected repeatedly";
                break;
            }
            rebuild();
            continue;
        }
        rejections = 0;
        const double ratio = predicted != 0.0 ? change / predicted : 1.0;
        if (ratio > 0.75 && step.norm() > 0.9 * trust) {
            trust = std::min(2.0 * trust, settings.max_trust_radius);
        } else if (ratio < 0.25) {
            trust = std::max(0.25 * trust, MIN_TRUST_RADIUS);
        }
        // Under constraints the model Hessian has to approximate the Lagrangian Hessian, which
        // includes the constraint curvature -lambda * Hessian(c). The plain energy-gradient
        // difference omits that term, leaving the model too soft along constrained paths.
        VectorXd curvature = basis.vectors.transpose() * as_vector(trial_gradient) - g_anc;
        if (constrained) {
            const ConstraintModel trial_model =
                linearize(problem.constraints, trial, problem.frozen, basis.vectors);
            curvature = lagrangian_gradient_change(basis.vectors, as_vector(gradient),
                                                   as_vector(trial_gradient), model.jacobian,
                                                   trial_model.jacobian);
        }
        bfgs_update(hessian, step, curvature);

        x = std::move(trial);
        gradient = std::move(trial_gradient);
        energy = trial_energy;
        last_change = change;
        stepped = true;
        fresh = true;
        if (++since_rebuild >= REBUILD_INTERVAL) {
            rebuild();
        }
    }
    result.x = x;
    result.energy = energy;
    result.gradient = gradient;
    result.iterations = iteration;
    result.energy_change = last_change;
    return result;
}

}  // namespace detail
}  // namespace OEVIBE
