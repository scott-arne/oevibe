#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/ScanGrid.h"
#include "core/Wavefront.h"
#include "oevibe/Units.h"

using namespace OEVIBE;

namespace {

using Seeds = std::vector<std::pair<std::size_t, std::vector<double>>>;
using Call = std::pair<std::size_t, double>;  // (point, first start coordinate)

detail::ScanGrid ring(double step) {
    return detail::ScanGrid({detail::build_scan_axis(-180.0, 180.0, step)});
}

detail::ScanAttempt converged_at(double energy, std::size_t point) {
    detail::ScanAttempt attempt;
    attempt.has_energy = true;
    attempt.energy = energy;
    attempt.converged = true;
    // The geometry is opaque to the wavefront; the point index lets tests see which point's
    // best seeded a later optimization.
    attempt.x = {static_cast<double>(point)};
    return attempt;
}

detail::ScanAttempt with(double energy, bool converged) {
    detail::ScanAttempt attempt;
    attempt.has_energy = true;
    attempt.energy = energy;
    attempt.converged = converged;
    return attempt;
}

// For the 6-point ring (-120, -60, 0, 60, 120, 180) seeded at index 5 with an energy that
// depends only on the point, worked out from spec 7.4: the seed; pass 1 from 5 to its
// neighbors 0 and 4; pass 2 from 0 (to 1, 5) and 4 (to 3, 5), where only 1 and 3 are new;
// pass 3 from 1 (to 0, 2) and 3 (to 2, 4), where only the first visit of 2 is new; pass 4 from
// 2 (to 1, 3) improves nothing.
const std::vector<Call> RING6_CALLS = {{5, 999.0}, {0, 5.0}, {4, 5.0}, {1, 0.0}, {5, 0.0},
                                       {3, 4.0},   {5, 4.0}, {0, 1.0}, {2, 1.0}, {2, 3.0},
                                       {4, 3.0},   {1, 2.0}, {3, 2.0}};

}  // namespace

TEST(WavefrontTest, AttemptImprovesRanksConvergedBeforeEnergy) {
    // A better convergence class wins whatever the energies.
    EXPECT_TRUE(detail::attempt_improves(with(5.0, true), with(-10.0, false)));
    EXPECT_FALSE(detail::attempt_improves(with(-10.0, false), with(5.0, true)));
    // Within a class the candidate must be lower by more than 1e-3 kcal/mol.
    for (const bool converged : {true, false}) {
        EXPECT_TRUE(detail::attempt_improves(with(1.0 - 2e-3, converged), with(1.0, converged)));
        EXPECT_FALSE(detail::attempt_improves(with(1.0 - 5e-4, converged), with(1.0, converged)));
        EXPECT_FALSE(detail::attempt_improves(with(1.0, converged), with(1.0, converged)));
        EXPECT_FALSE(detail::attempt_improves(with(1.5, converged), with(1.0, converged)));
    }
    // Only attempts with an energy rank at all.
    detail::ScanAttempt failed;
    failed.message = "engine failure";
    EXPECT_FALSE(detail::attempt_improves(failed, with(1.0, false)));
    EXPECT_TRUE(detail::attempt_improves(with(100.0, false), failed));
}

TEST(WavefrontTest, PeriodicCosineProfileReachesEveryPoint) {
    const detail::ScanGrid grid = ring(15.0);
    const auto profile = [&](std::size_t point) {
        return 1.0 + std::cos(3.0 * grid.Angles(point)[0] * Units::DEGREES_TO_RADIANS);
    };
    const detail::OptimizeAtPoint optimize = [&](std::size_t point, const std::vector<double>&) {
        return converged_at(profile(point), point);
    };
    const detail::WavefrontResult result =
        detail::run_wavefront(grid, {{23, {0.0}}}, 1000, optimize);
    ASSERT_EQ(result.points.size(), 24u);
    EXPECT_FALSE(result.truncated);
    // One seed, 2 optimizations in pass 1, 4 in each of passes 2 to 12 as the two fronts
    // advance and meet at the opposite point, and 2 in pass 13, which improves nothing.
    EXPECT_EQ(result.optimizations, 49u);
    unsigned attempts = 0;
    for (std::size_t point = 0; point < 24; ++point) {
        const detail::WavefrontPoint& state = result.points[point];
        ASSERT_TRUE(state.has_best) << point;
        EXPECT_DOUBLE_EQ(state.best.energy, profile(point)) << point;
        EXPECT_GE(state.attempts, 1u);
        attempts += state.attempts;
    }
    EXPECT_EQ(attempts, result.optimizations);
}

TEST(WavefrontTest, PropagationFollowsGridOrder) {
    const detail::ScanGrid grid = ring(60.0);
    std::vector<Call> calls;
    const detail::OptimizeAtPoint optimize = [&](std::size_t point,
                                                 const std::vector<double>& start_x) {
        calls.emplace_back(point, start_x.at(0));
        return converged_at(1.0, point);
    };
    const detail::WavefrontResult result =
        detail::run_wavefront(grid, {{5, {999.0}}}, 100, optimize);
    EXPECT_EQ(calls, RING6_CALLS);
    EXPECT_EQ(result.optimizations, 13u);
    EXPECT_FALSE(result.truncated);
}

TEST(WavefrontTest, ImprovementThresholdDecidesWhetherPropagationContinues) {
    const detail::ScanGrid grid = ring(60.0);
    for (const double gain : {5e-4, 2e-3}) {
        SCOPED_TRACE(gain);
        // Each new attempt at a point is ``gain`` lower than the previous one there.
        std::map<std::size_t, unsigned> visits;
        const detail::OptimizeAtPoint optimize = [&](std::size_t point,
                                                     const std::vector<double>&) {
            return converged_at(10.0 - gain * visits[point]++, point);
        };
        const detail::WavefrontResult result =
            detail::run_wavefront(grid, {{5, {0.0}}}, 40, optimize);
        if (gain < detail::SCAN_IMPROVEMENT_THRESHOLD) {
            // Revisits never replace the first result, so the run is the 13-call one.
            EXPECT_FALSE(result.truncated);
            EXPECT_EQ(result.optimizations, 13u);
            for (const detail::WavefrontPoint& state : result.points) {
                EXPECT_DOUBLE_EQ(state.best.energy, 10.0);
            }
        } else {
            // Every revisit improves, so only the cap stops the scan.
            EXPECT_TRUE(result.truncated);
            EXPECT_EQ(result.optimizations, 40u);
            for (const detail::WavefrontPoint& state : result.points) {
                ASSERT_TRUE(state.has_best);
                EXPECT_NEAR(state.best.energy, 10.0 - gain * (state.attempts - 1), 1e-12);
            }
        }
    }
}

TEST(WavefrontTest, BetterConvergenceClassReplacesAHigherEnergy) {
    const detail::ScanGrid grid = ring(60.0);
    std::map<std::size_t, unsigned> visits;
    // First visits stop early at a low energy; later visits converge higher.
    const detail::OptimizeAtPoint optimize = [&](std::size_t point, const std::vector<double>&) {
        const bool first = visits[point]++ == 0;
        detail::ScanAttempt attempt = with(first ? 1.0 : 3.0, !first);
        attempt.x = {static_cast<double>(point)};
        return attempt;
    };
    const detail::WavefrontResult result = detail::run_wavefront(grid, {{5, {0.0}}}, 100, optimize);
    EXPECT_FALSE(result.truncated);
    for (const detail::WavefrontPoint& state : result.points) {
        ASSERT_TRUE(state.has_best);
        ASSERT_GE(state.attempts, 2u);
        EXPECT_TRUE(state.best.converged);
        EXPECT_DOUBLE_EQ(state.best.energy, 3.0);
    }
}

TEST(WavefrontTest, CapStopsBeforeTheNextOptimization) {
    const detail::ScanGrid grid = ring(60.0);
    std::vector<Call> calls;
    const detail::OptimizeAtPoint optimize = [&](std::size_t point,
                                                 const std::vector<double>& start_x) {
        calls.emplace_back(point, start_x.at(0));
        return converged_at(1.0, point);
    };
    const detail::WavefrontResult capped = detail::run_wavefront(grid, {{5, {999.0}}}, 5, optimize);
    EXPECT_TRUE(capped.truncated);
    EXPECT_EQ(capped.optimizations, 5u);
    EXPECT_EQ(calls, std::vector<Call>(RING6_CALLS.begin(), RING6_CALLS.begin() + 5));
    EXPECT_EQ(capped.points[2].attempts, 0u);
    EXPECT_FALSE(capped.points[2].has_best);
    EXPECT_EQ(capped.points[3].attempts, 0u);

    // A cap that exactly fits the natural run does not truncate it.
    calls.clear();
    const detail::WavefrontResult exact = detail::run_wavefront(grid, {{5, {999.0}}}, 13, optimize);
    EXPECT_FALSE(exact.truncated);
    EXPECT_EQ(calls, RING6_CALLS);

    // The cap also applies to the seeds.
    calls.clear();
    const detail::WavefrontResult seeds_only =
        detail::run_wavefront(grid, {{5, {50.0}}, {2, {20.0}}}, 1, optimize);
    EXPECT_TRUE(seeds_only.truncated);
    EXPECT_EQ(calls, (std::vector<Call>{{5, 50.0}}));
}

TEST(WavefrontTest, SeedsRunInOrderAtTheirPoints) {
    const detail::ScanGrid grid = ring(60.0);
    std::vector<Call> calls;
    // A seed's energy is its marker coordinate, so the two seeds at point 4 rank by it.
    const detail::OptimizeAtPoint optimize = [&](std::size_t point,
                                                 const std::vector<double>& start_x) {
        calls.emplace_back(point, start_x.at(0));
        return converged_at(start_x.at(0), point);
    };
    const detail::WavefrontResult result =
        detail::run_wavefront(grid, {{4, {40.0}}, {1, {10.0}}, {4, {41.0}}}, 3, optimize);
    EXPECT_EQ(calls, (std::vector<Call>{{4, 40.0}, {1, 10.0}, {4, 41.0}}));
    EXPECT_EQ(result.points[4].attempts, 2u);
    EXPECT_DOUBLE_EQ(result.points[4].best.energy, 40.0);
    EXPECT_DOUBLE_EQ(result.points[1].best.energy, 10.0);
    EXPECT_THROW(detail::run_wavefront(grid, {{6, {0.0}}}, 3, optimize), std::invalid_argument);
}

TEST(WavefrontTest, FailedAttemptsLeaveNoBestAndSeedNothing) {
    const detail::ScanGrid grid = ring(60.0);
    std::vector<Call> calls;
    const detail::OptimizeAtPoint optimize = [&](std::size_t point,
                                                 const std::vector<double>& start_x) {
        calls.emplace_back(point, start_x.at(0));
        detail::ScanAttempt attempt;
        if (point == 2) {
            attempt.message = "SCF did not converge";
        } else if (point == 3) {
            attempt.unsupported = true;
            attempt.message = "element not parameterized";
        } else {
            attempt = converged_at(1.0, point);
        }
        return attempt;
    };
    const detail::WavefrontResult result = detail::run_wavefront(grid, {{5, {0.0}}}, 100, optimize);
    EXPECT_FALSE(result.truncated);
    const detail::WavefrontPoint& failed = result.points[2];
    EXPECT_FALSE(failed.has_best);
    EXPECT_GE(failed.attempts, 1u);
    EXPECT_FALSE(failed.any_unsupported);
    EXPECT_EQ(failed.last_message, "SCF did not converge");
    const detail::WavefrontPoint& unsupported = result.points[3];
    EXPECT_FALSE(unsupported.has_best);
    EXPECT_GE(unsupported.attempts, 1u);
    EXPECT_TRUE(unsupported.any_unsupported);
    EXPECT_EQ(unsupported.last_message, "element not parameterized");
    for (const std::size_t point : {0u, 1u, 4u, 5u}) {
        EXPECT_TRUE(result.points[point].has_best) << point;
    }
    // A point without a best never seeds its neighbors.
    for (const Call& call : calls) {
        EXPECT_NE(call.second, 2.0);
        EXPECT_NE(call.second, 3.0);
    }
}
