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

TEST(GeometryTest, WrapAngle) {
    EXPECT_NEAR(detail::wrap_angle(3.0 * Units::PI), Units::PI, 1e-12);
    EXPECT_NEAR(detail::wrap_angle(-Units::PI), Units::PI, 1e-12);
    EXPECT_NEAR(detail::wrap_angle(0.5), 0.5, 1e-12);
    EXPECT_NEAR(detail::wrap_angle(-1.5 * Units::PI), 0.5 * Units::PI, 1e-12);
}
