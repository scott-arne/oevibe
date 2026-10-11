#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/Lbfgs.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

detail::MinimizerSettings settings(unsigned iterations) {
    detail::MinimizerSettings s;
    s.max_iterations = iterations;
    s.energy_change = 1e-12;
    s.gradient_norm = 1e-6;
    return s;
}

double rosenbrock(const std::vector<double>& x, std::vector<double>& g) {
    g.assign(3, 0.0);
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    g[2] = 2.0 * x[2];
    return a * a + 100.0 * b * b + x[2] * x[2];
}

detail::ObjectiveFn springs(const std::vector<double>& reference) {
    return [reference](const std::vector<double>& x, std::vector<double>& g) {
        return test::spring_energy(reference, x, 50.0, &g);
    };
}

}  // namespace

TEST(LbfgsTest, SolvesRosenbrock) {
    const detail::MinimizerResult result =
        detail::lbfgs_minimize(rosenbrock, {-1.2, 1.0, 0.5}, {}, settings(2000));
    ASSERT_TRUE(result.converged) << result.message;
    EXPECT_NEAR(result.x[0], 1.0, 1e-4);
    EXPECT_NEAR(result.x[1], 1.0, 1e-4);
    EXPECT_NEAR(result.x[2], 0.0, 1e-4);
}

TEST(LbfgsTest, RelaxesASpringNetwork) {
    const std::vector<double> reference = test::sample_geometry();
    const detail::MinimizerResult result = detail::lbfgs_minimize(
        springs(reference), test::perturbed(reference, 0.15), {}, settings(500));
    ASSERT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.energy, 1e-8);
    EXPECT_GE(result.evaluations, result.iterations);
}

TEST(LbfgsTest, FrozenAtomsNeverMove) {
    const std::vector<double> reference = test::sample_geometry();
    const std::vector<double> start = test::perturbed(reference, 0.15);
    const detail::MinimizerResult result =
        detail::lbfgs_minimize(springs(reference), start, {1, 4}, settings(500));
    for (unsigned atom : {1u, 4u}) {
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_DOUBLE_EQ(result.x[3 * atom + axis], start[3 * atom + axis]);
        }
    }
}

TEST(LbfgsTest, StopsAtTheIterationLimit) {
    const detail::MinimizerResult result =
        detail::lbfgs_minimize(rosenbrock, {-1.2, 1.0, 0.5}, {}, settings(3));
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 3u);
    EXPECT_NE(result.message.find("iteration"), std::string::npos);
}
