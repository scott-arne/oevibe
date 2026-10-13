#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/ScanGrid.h"
#include "oevibe/Error.h"
#include "oevibe/Options.h"

using namespace OEVIBE;

namespace {

constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();
constexpr double INF_VALUE = std::numeric_limits<double>::infinity();

ScanOptions one_dihedral() {
    ScanOptions options;
    options.AddDihedral(0, 1, 2, 3);
    return options;
}

void expect_values(const detail::ScanAxis& axis, const std::vector<double>& expected) {
    ASSERT_EQ(axis.values.size(), expected.size());
    for (std::size_t k = 0; k < expected.size(); ++k) {
        EXPECT_NEAR(axis.values[k], expected[k], 1e-9) << "value " << k;
    }
}

// Axis 0 is periodic with 3 values (-60, 60, 180) and axis 1 is not, with 4 (0, 30, 60, 90), so
// a transposed index or a wrap on the wrong axis changes the answer.
detail::ScanGrid mixed_grid() {
    return detail::ScanGrid(
        {detail::build_scan_axis(-180.0, 180.0, 120.0), detail::build_scan_axis(0.0, 90.0, 30.0)});
}

}  // namespace

// Spec 5.5 defaults.
TEST(ScanOptionsTest, DefaultsMatchSpec) {
    const ScanOptions options;
    EXPECT_EQ(options.GetNumDihedrals(), 0u);
    EXPECT_DOUBLE_EQ(options.GetStep(), 15.0);
    EXPECT_DOUBLE_EQ(options.GetRangeStart(), -180.0);
    EXPECT_DOUBLE_EQ(options.GetRangeEnd(), 180.0);
    EXPECT_FALSE(options.HasMaxOptimizations());
    EXPECT_THROW(options.GetMaxOptimizations(), OptionsError);
    EXPECT_EQ(options.GetOptimizerOptions().GetAlgorithm(), Algorithm::AUTO);
    EXPECT_TRUE(options.GetOptimizerOptions().GetConstraints().IsEmpty());
}

TEST(ScanOptionsTest, SettersValidateSingleValues) {
    ScanOptions options;
    for (const double bad : {0.0, -15.0, NAN_VALUE, INF_VALUE}) {
        EXPECT_THROW(options.SetStep(bad), OptionsError) << bad;
    }
    EXPECT_DOUBLE_EQ(options.GetStep(), 15.0);
    EXPECT_THROW(options.SetRange(NAN_VALUE, 0.0), OptionsError);
    EXPECT_THROW(options.SetRange(0.0, INF_VALUE), OptionsError);
    EXPECT_DOUBLE_EQ(options.GetRangeStart(), -180.0);
    EXPECT_THROW(options.SetMaxOptimizations(0), OptionsError);
    options.SetMaxOptimizations(7);
    EXPECT_EQ(options.GetMaxOptimizations(), 7u);
    options.ClearMaxOptimizations();
    EXPECT_FALSE(options.HasMaxOptimizations());
    options.SetStep(30.0);
    options.SetRange(0.0, 90.0);
    EXPECT_DOUBLE_EQ(options.GetStep(), 30.0);
    EXPECT_DOUBLE_EQ(options.GetRangeStart(), 0.0);
    EXPECT_DOUBLE_EQ(options.GetRangeEnd(), 90.0);
}

TEST(ScanOptionsTest, DihedralsAreDistinctAndAtMostTwo) {
    ScanOptions options;
    EXPECT_THROW(options.AddDihedral(0, 1, 1, 2), OptionsError);
    EXPECT_THROW(options.AddDihedral(3, 1, 2, 3), OptionsError);
    options.AddDihedral(0, 1, 2, 3);
    EXPECT_THROW(options.AddDihedral(0, 1, 2, 3), OptionsError);
    EXPECT_THROW(options.AddDihedral(3, 2, 1, 0), OptionsError);
    options.AddDihedral(1, 2, 3, 4);
    EXPECT_THROW(options.AddDihedral(2, 3, 4, 5), OptionsError);
    ASSERT_EQ(options.GetNumDihedrals(), 2u);
    EXPECT_EQ(options.GetDihedral(1), (std::vector<unsigned>{1, 2, 3, 4}));
    EXPECT_THROW(options.GetDihedral(2), OptionsError);
    options.ClearDihedrals();
    EXPECT_EQ(options.GetNumDihedrals(), 0u);
}

TEST(ScanOptionsTest, ValidateAcceptsTheDefaultsWithADihedral) {
    EXPECT_NO_THROW(one_dihedral().Validate());
    ScanOptions partial = one_dihedral();
    partial.SetRange(0.0, 90.0);
    partial.SetStep(30.0);
    EXPECT_NO_THROW(partial.Validate());
    // In binary floating point 0.3 / 0.1 is 2.9999999999999996 and 3 * 0.1 misses 0.3 by 6e-17,
    // so only the tolerance lets 0.1 divide 0.3.
    ScanOptions fine = one_dihedral();
    fine.SetRange(0.0, 0.3);
    fine.SetStep(0.1);
    EXPECT_NO_THROW(fine.Validate());
    // A constraint on another dihedral of the same atoms is allowed.
    ScanOptions other = one_dihedral();
    OptimizerOptions optimizer;
    ConstraintSet constraints;
    constraints.AddDihedral(0, 1, 2, 4, 60.0);
    constraints.AddDistance(0, 3);
    optimizer.SetConstraints(constraints);
    other.SetOptimizerOptions(optimizer);
    EXPECT_NO_THROW(other.Validate());
}

TEST(ScanOptionsTest, ValidateRejectsEachInvalidCombination) {
    EXPECT_THROW(ScanOptions().Validate(), OptionsError);

    const auto with_range = [](double start, double end, double step) {
        ScanOptions options = one_dihedral();
        options.SetRange(start, end);
        options.SetStep(step);
        return options;
    };
    EXPECT_THROW(with_range(30.0, 30.0, 15.0).Validate(), OptionsError);
    EXPECT_THROW(with_range(90.0, 0.0, 15.0).Validate(), OptionsError);
    EXPECT_THROW(with_range(-180.0, 181.0, 1.0).Validate(), OptionsError);
    EXPECT_THROW(with_range(-180.0, 180.0, 7.0).Validate(), OptionsError);
    EXPECT_THROW(with_range(0.0, 90.0, 120.0).Validate(), OptionsError);

    const auto with_optimizer = [](const OptimizerOptions& optimizer) {
        ScanOptions options = one_dihedral();
        options.SetOptimizerOptions(optimizer);
        return options;
    };
    OptimizerOptions inverted_trust;
    inverted_trust.SetInitialTrustRadius(0.5);
    inverted_trust.SetMaxTrustRadius(0.2);
    EXPECT_THROW(with_optimizer(inverted_trust).Validate(), OptionsError);
    OptimizerOptions lbfgs;
    lbfgs.SetAlgorithm(Algorithm::LBFGS);
    EXPECT_THROW(with_optimizer(lbfgs).Validate(), OptionsError);

    const auto constraining = [&](const ConstraintSet& constraints) {
        OptimizerOptions optimizer;
        optimizer.SetConstraints(constraints);
        return with_optimizer(optimizer);
    };
    ConstraintSet same;
    same.AddDihedral(0, 1, 2, 3, 60.0);
    EXPECT_THROW(constraining(same).Validate(), OptionsError);
    ConstraintSet reversed;
    reversed.AddDihedral(3, 2, 1, 0);
    EXPECT_THROW(constraining(reversed).Validate(), OptionsError);
    ConstraintSet restrained;
    restrained.AddDihedralRestraint(0, 1, 2, 3, 10.0);
    EXPECT_THROW(constraining(restrained).Validate(), OptionsError);
    ConstraintSet reversed_restraint;
    reversed_restraint.AddDihedralRestraint(3, 2, 1, 0, 10.0, 5.0);
    EXPECT_THROW(constraining(reversed_restraint).Validate(), OptionsError);
}

TEST(ScanGridTest, FullCircleIsHalfOpenAndPeriodic) {
    const detail::ScanAxis axis = detail::build_scan_axis(-180.0, 180.0, 15.0);
    EXPECT_TRUE(axis.periodic);
    std::vector<double> expected;
    for (int k = 0; k < 24; ++k) {
        expected.push_back(-165.0 + 15.0 * k);
    }
    expect_values(axis, expected);
    EXPECT_DOUBLE_EQ(axis.values.front(), -165.0);
    EXPECT_DOUBLE_EQ(axis.values.back(), 180.0);
}

TEST(ScanGridTest, FullCircleValuesAreNormalizedInRangeOrder) {
    const detail::ScanAxis axis = detail::build_scan_axis(0.0, 360.0, 90.0);
    EXPECT_TRUE(axis.periodic);
    expect_values(axis, {90.0, 180.0, -90.0, 0.0});
}

TEST(ScanGridTest, PartialRangesAreInclusiveAndNotPeriodic) {
    const detail::ScanAxis axis = detail::build_scan_axis(-60.0, 60.0, 30.0);
    EXPECT_FALSE(axis.periodic);
    expect_values(axis, {-60.0, -30.0, 0.0, 30.0, 60.0});
    const detail::ScanAxis seam = detail::build_scan_axis(90.0, 270.0, 45.0);
    EXPECT_FALSE(seam.periodic);
    expect_values(seam, {90.0, 135.0, 180.0, -135.0, -90.0});
    // Binary rounding of 0.1 must not lose the last value.
    expect_values(detail::build_scan_axis(0.0, 0.3, 0.1), {0.0, 0.1, 0.2, 0.3});
}

TEST(ScanGridTest, InvalidAxesThrowOptionsError) {
    EXPECT_THROW(detail::build_scan_axis(0.0, 0.0, 15.0), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(10.0, 0.0, 15.0), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(-180.0, 181.0, 1.0), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(0.0, 100.0, 30.0), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(0.0, 90.0, 0.0), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(0.0, 90.0, NAN_VALUE), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(NAN_VALUE, 90.0, 15.0), OptionsError);
}

TEST(ScanGridTest, TwoDimensionalGridsAreRowMajorWithTheFirstAxisSlowest) {
    const detail::ScanGrid grid = mixed_grid();
    ASSERT_EQ(grid.Size(), 12u);
    ASSERT_EQ(grid.GetAxes().size(), 2u);
    const std::vector<double> first = {-60.0, 60.0, 180.0};
    const std::vector<double> second = {0.0, 30.0, 60.0, 90.0};
    for (std::size_t index = 0; index < grid.Size(); ++index) {
        const std::vector<double> angles = grid.Angles(index);
        ASSERT_EQ(angles.size(), 2u);
        EXPECT_DOUBLE_EQ(angles[0], first[index / 4]) << index;
        EXPECT_DOUBLE_EQ(angles[1], second[index % 4]) << index;
    }
    EXPECT_THROW(grid.Angles(12), std::out_of_range);
}

TEST(ScanGridTest, NeighborsWrapOnlyOnPeriodicAxes) {
    const detail::ScanGrid ring({detail::build_scan_axis(-180.0, 180.0, 15.0)});
    EXPECT_EQ(ring.Neighbors(0), (std::vector<std::size_t>{1, 23}));
    EXPECT_EQ(ring.Neighbors(23), (std::vector<std::size_t>{0, 22}));
    EXPECT_EQ(ring.Neighbors(5), (std::vector<std::size_t>{4, 6}));

    const detail::ScanGrid line({detail::build_scan_axis(0.0, 90.0, 30.0)});
    EXPECT_EQ(line.Neighbors(0), (std::vector<std::size_t>{1}));
    EXPECT_EQ(line.Neighbors(1), (std::vector<std::size_t>{0, 2}));
    EXPECT_EQ(line.Neighbors(3), (std::vector<std::size_t>{2}));

    const detail::ScanGrid grid = mixed_grid();
    // (0, 0): the periodic axis wraps to (2, 0) = 8; the other axis has no (0, -1).
    EXPECT_EQ(grid.Neighbors(0), (std::vector<std::size_t>{1, 4, 8}));
    // (1, 3): no (1, 4) on the non-periodic axis.
    EXPECT_EQ(grid.Neighbors(7), (std::vector<std::size_t>{3, 6, 11}));
    // (2, 1): the periodic axis wraps to (0, 1) = 1.
    EXPECT_EQ(grid.Neighbors(9), (std::vector<std::size_t>{1, 5, 8, 10}));

    const detail::ScanGrid pair({detail::build_scan_axis(-180.0, 180.0, 180.0)});
    EXPECT_EQ(pair.Neighbors(0), (std::vector<std::size_t>{1}));
    const detail::ScanGrid single({detail::build_scan_axis(-180.0, 180.0, 360.0)});
    EXPECT_TRUE(single.Neighbors(0).empty());
    EXPECT_THROW(ring.Neighbors(24), std::out_of_range);
}

TEST(ScanGridTest, NearestUsesWrappedAngularDistance) {
    const detail::ScanGrid ring({detail::build_scan_axis(-180.0, 180.0, 15.0)});
    // Index 23 is 180 and index 0 is -165: across the seam -179 is 1 degree from 180.
    EXPECT_EQ(ring.Nearest({-179.0}), 23u);
    EXPECT_EQ(ring.Nearest({179.0}), 23u);
    EXPECT_EQ(ring.Nearest({-172.0}), 0u);
    EXPECT_EQ(ring.Nearest({0.4}), 11u);
    // Equidistant from 0 (index 11) and 15 (index 12): the lower index wins.
    EXPECT_EQ(ring.Nearest({7.5}), 11u);
    EXPECT_EQ(ring.Nearest({367.0}), 11u);

    // A non-periodic axis whose range crosses the seam still measures angles periodically.
    const detail::ScanGrid seam({detail::build_scan_axis(90.0, 270.0, 45.0)});
    EXPECT_EQ(seam.Nearest({-179.0}), 2u);
    EXPECT_EQ(seam.Nearest({-100.0}), 4u);

    const detail::ScanGrid grid = mixed_grid();
    EXPECT_EQ(grid.Nearest({-170.0, 50.0}), 10u);
    EXPECT_EQ(grid.Nearest({55.0, -170.0}), 7u);
    EXPECT_THROW(grid.Nearest({0.0}), std::invalid_argument);
}

TEST(ScanGridTest, BuildScanGridFollowsTheOptions) {
    ScanOptions options = one_dihedral();
    options.AddDihedral(1, 2, 3, 4);
    options.SetStep(90.0);
    const detail::ScanGrid grid = detail::build_scan_grid(options);
    ASSERT_EQ(grid.Size(), 16u);
    EXPECT_TRUE(grid.GetAxes()[0].periodic);
    EXPECT_TRUE(grid.GetAxes()[1].periodic);
    EXPECT_EQ(grid.Angles(1), (std::vector<double>{-90.0, 0.0}));
    options.SetStep(7.0);
    EXPECT_THROW(detail::build_scan_grid(options), OptionsError);
}

TEST(ScanGridTest, ConstructorRejectsMissingAxesAndValues) {
    EXPECT_THROW(detail::ScanGrid(std::vector<detail::ScanAxis>{}), std::invalid_argument);
    EXPECT_THROW(detail::ScanGrid(std::vector<detail::ScanAxis>{detail::ScanAxis()}),
                 std::invalid_argument);
}

// A step too small for any useful grid is rejected before its interval count is converted to an
// integer or any value is allocated.
TEST(ScanGridTest, TinyStepsAreRejectedBeforeAnyAllocation) {
    EXPECT_THROW(detail::scan_axis_intervals(-180.0, 180.0, 1e-300), OptionsError);
    EXPECT_THROW(detail::build_scan_axis(-180.0, 180.0, 1e-4), OptionsError);  // 3.6e6 intervals
    EXPECT_EQ(detail::scan_axis_intervals(-180.0, 180.0, 360.0 / 1e6),
              detail::MAX_SCAN_GRID_POINTS);
}

TEST(ScanGridTest, ConstructorRejectsGridsBeyondTheSizeLimit) {
    detail::ScanAxis wide;
    wide.values.assign(1001, 0.0);
    EXPECT_THROW(detail::ScanGrid({wide, wide}), std::invalid_argument);
    wide.values.resize(1000);
    EXPECT_EQ(detail::ScanGrid({wide, wide}).Size(), detail::MAX_SCAN_GRID_POINTS);
}

// The limit applies to the whole grid, so a step that is fine for one dihedral can be too fine
// for two; the product is checked without overflowing.
TEST(ScanOptionsTest, GridsBeyondTheSizeLimitAreRejected) {
    ScanOptions options = one_dihedral();
    options.SetStep(0.001);  // 360000 points on one axis
    EXPECT_NO_THROW(options.Validate());
    options.AddDihedral(1, 2, 3, 4);  // 360000 squared points
    EXPECT_THROW(options.Validate(), OptionsError);
    options.SetRange(-150.0, 150.0);
    options.SetStep(0.3);  // 1001 inclusive points per axis, 1002001 in all
    EXPECT_THROW(options.Validate(), OptionsError);
    options.SetRange(-149.7, 150.0);  // 1000 points per axis: exactly the limit
    EXPECT_NO_THROW(options.Validate());
}
