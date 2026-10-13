#include "core/Wavefront.h"

#include <stdexcept>
#include <string>

namespace OEVIBE {
namespace detail {

bool attempt_improves(const ScanAttempt& candidate, const ScanAttempt& incumbent) {
    if (!candidate.has_energy) {
        return false;
    }
    if (!incumbent.has_energy) {
        return true;
    }
    if (candidate.converged != incumbent.converged) {
        return candidate.converged;
    }
    return candidate.energy < incumbent.energy - SCAN_IMPROVEMENT_THRESHOLD;
}

WavefrontResult run_wavefront(const ScanGrid& grid,
                              const std::vector<std::pair<std::size_t, std::vector<double>>>& seeds,
                              unsigned max_optimizations, const OptimizeAtPoint& optimize) {
    for (const auto& seed : seeds) {
        if (seed.first >= grid.Size()) {
            throw std::invalid_argument("seed point " + std::to_string(seed.first) +
                                        " is outside the scan grid");
        }
    }
    WavefrontResult result;
    result.points.resize(grid.Size());
    std::vector<bool> improved(grid.Size(), false);

    // Runs one optimization unless the cap has been reached; returns false when it has.
    const auto attempt_at = [&](std::size_t point, const std::vector<double>& start_x) {
        if (result.optimizations >= max_optimizations) {
            result.truncated = true;
            return false;
        }
        ScanAttempt attempt = optimize(point, start_x);
        ++result.optimizations;
        WavefrontPoint& target = result.points[point];
        ++target.attempts;
        target.any_unsupported = target.any_unsupported || attempt.unsupported;
        target.last_message = attempt.message;
        if (attempt.has_energy && (!target.has_best || attempt_improves(attempt, target.best))) {
            target.best = std::move(attempt);
            target.has_best = true;
            improved[point] = true;
        }
        return true;
    };

    for (const auto& seed : seeds) {
        if (!attempt_at(seed.first, seed.second)) {
            return result;
        }
    }
    while (true) {
        std::vector<std::size_t> sources;
        for (std::size_t point = 0; point < improved.size(); ++point) {
            if (improved[point]) {
                sources.push_back(point);
            }
        }
        if (sources.empty()) {
            return result;
        }
        improved.assign(improved.size(), false);
        for (const std::size_t source : sources) {
            // A point is never its own neighbor, so this best stays put while its neighbors run.
            const std::vector<double>& start_x = result.points[source].best.x;
            for (const std::size_t neighbor : grid.Neighbors(source)) {
                if (!attempt_at(neighbor, start_x)) {
                    return result;
                }
            }
        }
    }
}

}  // namespace detail
}  // namespace OEVIBE
