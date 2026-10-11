#ifndef OEVIBE_CORE_MINIMIZER_H
#define OEVIBE_CORE_MINIMIZER_H

#include <functional>
#include <string>
#include <vector>

namespace OEVIBE {
namespace detail {

/// Objective in kcal/mol over dense coordinates in Angstrom; fills the 3N gradient.
using ObjectiveFn =
    std::function<double(const std::vector<double>& x, std::vector<double>& gradient)>;

struct MinimizerResult {
    std::vector<double> x;
    double energy = 0.0;
    std::vector<double> gradient;
    unsigned iterations = 0;
    unsigned evaluations = 0;
    bool converged = false;
    double energy_change = 0.0;
    double gradient_norm = 0.0;
    double max_gradient_component = 0.0;
    double max_distance_residual = 0.0;  ///< Angstrom
    double max_angle_residual = 0.0;     ///< degrees
    std::string message;
};

/// Receives the current iterate once for the starting point and once after every accepted
/// step, before the minimizer evaluates any other point, so the iterate is always the most
/// recently evaluated point. Everything but ``gradient``, ``converged``, and ``message`` is
/// filled in as the minimizer would report it if it stopped there.
using IterateObserver = std::function<void(const MinimizerResult& iterate)>;

struct MinimizerSettings {
    unsigned max_iterations = 200;
    double energy_change = 0.0;         ///< kcal/mol
    double gradient_norm = 0.0;         ///< kcal/mol/Angstrom, projected gradient
    double initial_trust_radius = 0.1;  ///< Angstrom
    double max_trust_radius = 0.3;      ///< Angstrom
    IterateObserver observer;           ///< Optional.
};

/// Zeroes the x, y, z components of every frozen atom.
inline void zero_frozen(std::vector<double>& values, const std::vector<unsigned>& frozen) {
    for (unsigned atom : frozen) {
        for (int axis = 0; axis < 3; ++axis) {
            values[3 * atom + axis] = 0.0;
        }
    }
}

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_MINIMIZER_H
