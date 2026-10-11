#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "core/Geometry.h"
#include "oevibe/Units.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

// Compares an analytic gradient over ``atoms`` with central finite differences.
void expect_gradient_matches(const std::function<double(const double*, double*)>& value,
                             std::vector<double> x, const std::vector<unsigned>& atoms) {
    std::vector<double> analytic(3 * atoms.size());
    value(x.data(), analytic.data());
    const double step = 1e-6;
    for (std::size_t slot = 0; slot < atoms.size(); ++slot) {
        for (int axis = 0; axis < 3; ++axis) {
            const std::size_t k = 3 * atoms[slot] + axis;
            const double saved = x[k];
            x[k] = saved + step;
            const double plus = value(x.data(), nullptr);
            x[k] = saved - step;
            const double minus = value(x.data(), nullptr);
            x[k] = saved;
            EXPECT_NEAR(analytic[3 * slot + axis], (plus - minus) / (2.0 * step), 1e-6);
        }
    }
}

}  // namespace

TEST(GeometryTest, KnownValues) {
    const std::vector<double> x = {0.0, 0.0, 0.0, 3.0, 4.0, 0.0};
    EXPECT_DOUBLE_EQ(detail::distance(x.data(), 0, 1, nullptr), 5.0);
    const std::vector<double> corner = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    EXPECT_NEAR(detail::angle(corner.data(), 0, 1, 2, nullptr), Units::PI / 2.0, 1e-12);
    const std::vector<double> gauche = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0};
    EXPECT_NEAR(detail::dihedral(gauche.data(), 0, 1, 2, 3, nullptr), Units::PI / 2.0, 1e-12);
    const std::vector<double> anti = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0};
    EXPECT_NEAR(std::abs(detail::dihedral(anti.data(), 0, 1, 2, 3, nullptr)), Units::PI, 1e-12);
}

TEST(GeometryTest, GradientsMatchFiniteDifferences) {
    const std::vector<double> x = test::sample_geometry();
    expect_gradient_matches([](const double* p, double* g) { return detail::distance(p, 0, 3, g); },
                            x, {0, 3});
    expect_gradient_matches([](const double* p, double* g) { return detail::angle(p, 4, 0, 2, g); },
                            x, {4, 0, 2});
    expect_gradient_matches(
        [](const double* p, double* g) { return detail::dihedral(p, 0, 1, 2, 3, g); }, x,
        {0, 1, 2, 3});
    expect_gradient_matches(
        [](const double* p, double* g) { return detail::dihedral(p, 4, 0, 1, 5, g); }, x,
        {4, 0, 1, 5});
}

// A linear bend has no derivative. The gradient there is the one-sided derivative for bending
// toward a fixed perpendicular direction: moving along it changes the angle at the rate its norm
// predicts, so a restraint or constraint toward another angle has a direction to follow.
TEST(GeometryTest, LinearBendsHaveAnEscapeDirection) {
    const double axis[3] = {0.48, 0.6, 0.64};  // unit length, skewed off every coordinate axis
    const double nu = 1.0;
    const double nv = 1.5;
    // k beyond j from i (pi), then k folded back onto i's side (0).
    for (const double side : {-1.0, 1.0}) {
        SCOPED_TRACE(side < 0.0 ? "straight" : "folded");
        std::vector<double> x(9, 0.0);
        for (int a = 0; a < 3; ++a) {
            x[a] = nu * axis[a];
            x[6 + a] = side * nv * axis[a];
        }
        double g[9];
        const double theta = detail::angle(x.data(), 0, 1, 2, g);
        EXPECT_NEAR(theta, side < 0.0 ? Units::PI : 0.0, 1e-12);

        double gi_norm = 0.0;
        double gk_norm = 0.0;
        double gi_along_bond = 0.0;
        double gk_along_bond = 0.0;
        for (int a = 0; a < 3; ++a) {
            gi_norm += g[a] * g[a];
            gk_norm += g[6 + a] * g[6 + a];
            gi_along_bond += g[a] * axis[a];
            gk_along_bond += g[6 + a] * axis[a];
            EXPECT_NEAR(g[a] + g[3 + a] + g[6 + a], 0.0, 1e-12);  // no net force
        }
        // The limits of the regular formula, whose atom gradients always have norms 1/r.
        EXPECT_NEAR(std::sqrt(gi_norm), 1.0 / nu, 1e-12);
        EXPECT_NEAR(std::sqrt(gk_norm), 1.0 / nv, 1e-12);
        EXPECT_NEAR(gi_along_bond, 0.0, 1e-12);
        EXPECT_NEAR(gk_along_bond, 0.0, 1e-12);
        // No net torque about j.
        const double torque[3] = {
            x[1] * g[2] - x[2] * g[1] + x[7] * g[8] - x[8] * g[7],
            x[2] * g[0] - x[0] * g[2] + x[8] * g[6] - x[6] * g[8],
            x[0] * g[1] - x[1] * g[0] + x[6] * g[7] - x[7] * g[6],
        };
        for (const double component : torque) {
            EXPECT_NEAR(component, 0.0, 1e-12);
        }

        // The angle can only move into (0, pi): along -g from pi, along +g from 0.
        double norm = 0.0;
        for (const double component : g) {
            norm += component * component;
        }
        norm = std::sqrt(norm);
        const double step = side < 0.0 ? -1e-6 : 1e-6;
        std::vector<double> moved = x;
        for (int c = 0; c < 9; ++c) {
            moved[c] += step * g[c] / norm;
        }
        const double rate = (detail::angle(moved.data(), 0, 1, 2, nullptr) - theta) / step;
        EXPECT_NEAR(rate, norm, 1e-5);
    }

    // Coincident atoms leave the angle undefined in every direction.
    const std::vector<double> coincident = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double g[9];
    detail::angle(coincident.data(), 0, 1, 2, g);
    for (const double component : g) {
        EXPECT_EQ(component, 0.0);
    }
}

TEST(GeometryTest, WrapAngle) {
    EXPECT_NEAR(detail::wrap_angle(3.0 * Units::PI), Units::PI, 1e-12);
    EXPECT_NEAR(detail::wrap_angle(-Units::PI), Units::PI, 1e-12);
    EXPECT_NEAR(detail::wrap_angle(0.5), 0.5, 1e-12);
    EXPECT_NEAR(detail::wrap_angle(-1.5 * Units::PI), 0.5 * Units::PI, 1e-12);
}
