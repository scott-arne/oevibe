#ifndef OEVIBE_CORE_WAVEFRONT_H
#define OEVIBE_CORE_WAVEFRONT_H

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "core/ScanGrid.h"

namespace OEVIBE {
namespace detail {

/// kcal/mol. A same-class result must be lower by more than this to replace a point's best
/// (spec 7.4), so re-optimizations that land in the same minimum stop the propagation.
constexpr double SCAN_IMPROVEMENT_THRESHOLD = 1e-3;
/// The default optimization cap is this many optimizations per grid point (spec 7.4).
constexpr unsigned SCAN_OPTIMIZATIONS_PER_POINT = 20;

/// One constrained optimization at a grid point.
struct ScanAttempt {
    bool has_energy = false;
    double energy = 0.0;  ///< kcal/mol, engine energy without restraint energy.
    bool converged = false;
    bool unsupported = false;  ///< The engine reported the system unsupported.
    std::vector<double> x;     ///< Dense 3N, Angstrom; empty without a geometry.
    std::string message;
};

/// What the wavefront learned about one grid point.
struct WavefrontPoint {
    unsigned attempts = 0;
    bool has_best = false;
    ScanAttempt best;  ///< The best attempt with an energy (spec 7.4 ranking).
    bool any_unsupported = false;
    std::string last_message;  ///< The latest attempt's message.
};

struct WavefrontResult {
    std::vector<WavefrontPoint> points;  ///< One per grid point, in grid order.
    unsigned optimizations = 0;
    bool truncated = false;  ///< An optimization was due when the cap had been reached.
};

/// Optimizes at grid point ``point`` from ``start_x`` (dense 3N, Angstrom) and reports the
/// outcome; failures are reported in the attempt, not thrown.
using OptimizeAtPoint =
    std::function<ScanAttempt(std::size_t point, const std::vector<double>& start_x)>;

/// :returns: Whether ``candidate`` replaces ``incumbent`` (spec 7.4): only a candidate with an
///     energy can, and it does when the incumbent has none, when it converged and the incumbent
///     did not, or when both are in the same class and it is lower by more than
///     SCAN_IMPROVEMENT_THRESHOLD. Ties keep the incumbent.
bool attempt_improves(const ScanAttempt& candidate, const ScanAttempt& incumbent);

/// Wavefront propagation (Qiu et al., J. Chem. Phys. 152, 244116, 2020; spec 7.4). Each seed,
/// in order, is optimized at its point from its geometry. Then, pass after pass, every point
/// whose best improved in the previous pass, in grid order, seeds an optimization of each of its
/// neighbors (in grid order) from its current best geometry. It stops after a pass without an
/// improvement, or when an optimization is due after ``max_optimizations`` have run.
/// :param seeds: (grid point, dense starting coordinates) pairs.
/// :raises std::invalid_argument: For a seed point outside the grid.
WavefrontResult run_wavefront(const ScanGrid& grid,
                              const std::vector<std::pair<std::size_t, std::vector<double>>>& seeds,
                              unsigned max_optimizations, const OptimizeAtPoint& optimize);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_WAVEFRONT_H
