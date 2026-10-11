#include "core/Lbfgs.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <deque>

namespace OEVIBE {
namespace detail {

namespace {

constexpr std::size_t LBFGS_MEMORY = 10;
constexpr int MAX_LINE_SEARCH_HALVINGS = 30;
constexpr double ARMIJO_FACTOR = 1e-4;

Eigen::VectorXd as_vector(const std::vector<double>& values) {
    return Eigen::Map<const Eigen::VectorXd>(values.data(),
                                             static_cast<Eigen::Index>(values.size()));
}

}  // namespace

MinimizerResult lbfgs_minimize(const ObjectiveFn& objective, std::vector<double> x,
                               const std::vector<unsigned>& frozen,
                               const MinimizerSettings& settings) {
    using Eigen::VectorXd;
    const auto dim = static_cast<Eigen::Index>(x.size());
    MinimizerResult result;
    std::vector<double> gradient(x.size(), 0.0);
    double energy = objective(x, gradient);
    result.evaluations = 1;
    zero_frozen(gradient, frozen);

    std::deque<VectorXd> s_history;
    std::deque<VectorXd> y_history;
    bool stepped = false;
    double last_change = 0.0;
    unsigned iteration = 0;
    while (true) {
        const VectorXd g = as_vector(gradient);
        const double g_norm = g.norm();
        const bool energy_ok = !stepped || std::abs(last_change) < settings.energy_change;
        if (g_norm < settings.gradient_norm && energy_ok) {
            result.converged = true;
            break;
        }
        if (iteration >= settings.max_iterations) {
            result.message = "reached the iteration limit";
            break;
        }
        ++iteration;

        VectorXd q = g;
        std::vector<double> alpha(s_history.size());
        for (std::size_t k = s_history.size(); k-- > 0;) {
            const double rho = 1.0 / y_history[k].dot(s_history[k]);
            alpha[k] = rho * s_history[k].dot(q);
            q -= alpha[k] * y_history[k];
        }
        const double gamma = s_history.empty() ? 1.0 / std::max(g_norm, 1.0)
                                               : s_history.back().dot(y_history.back()) /
                                                     y_history.back().squaredNorm();
        VectorXd r = gamma * q;
        for (std::size_t k = 0; k < s_history.size(); ++k) {
            const double rho = 1.0 / y_history[k].dot(s_history[k]);
            const double beta = rho * y_history[k].dot(r);
            r += (alpha[k] - beta) * s_history[k];
        }
        VectorXd direction = -r;
        for (unsigned atom : frozen) {
            direction.segment<3>(3 * static_cast<Eigen::Index>(atom)).setZero();
        }
        if (direction.dot(g) >= 0.0) {
            s_history.clear();
            y_history.clear();
            direction = -g;
        }
        double largest = 0.0;
        for (Eigen::Index atom = 0; atom < dim / 3; ++atom) {
            largest = std::max(largest, direction.segment<3>(3 * atom).norm());
        }
        double step =
            largest > settings.max_trust_radius ? settings.max_trust_radius / largest : 1.0;
        const double slope = direction.dot(g);
        std::vector<double> trial(x.size());
        std::vector<double> trial_gradient(x.size(), 0.0);
        double trial_energy = 0.0;
        bool accepted = false;
        for (int attempt = 0; attempt < MAX_LINE_SEARCH_HALVINGS; ++attempt) {
            for (Eigen::Index k = 0; k < dim; ++k) {
                trial[k] = x[k] + step * direction(k);
            }
            trial_energy = objective(trial, trial_gradient);
            ++result.evaluations;
            zero_frozen(trial_gradient, frozen);
            if (trial_energy <= energy + ARMIJO_FACTOR * step * slope) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            if (!s_history.empty()) {
                // A stale curvature history can point uphill; retry once as steepest descent.
                s_history.clear();
                y_history.clear();
                continue;
            }
            result.message = "the line search found no lower energy";
            break;
        }
        const VectorXd s = as_vector(trial) - as_vector(x);
        const VectorXd y = as_vector(trial_gradient) - g;
        if (s.dot(y) > 1e-10) {
            s_history.push_back(s);
            y_history.push_back(y);
            if (s_history.size() > LBFGS_MEMORY) {
                s_history.pop_front();
                y_history.pop_front();
            }
        }
        last_change = trial_energy - energy;
        stepped = true;
        x = std::move(trial);
        gradient = std::move(trial_gradient);
        energy = trial_energy;
    }
    result.x = x;
    result.energy = energy;
    result.gradient = gradient;
    result.iterations = iteration;
    result.energy_change = last_change;
    const VectorXd g = as_vector(gradient);
    result.gradient_norm = g.norm();
    result.max_gradient_component = dim > 0 ? g.cwiseAbs().maxCoeff() : 0.0;
    return result;
}

}  // namespace detail
}  // namespace OEVIBE
