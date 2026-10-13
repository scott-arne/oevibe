#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "engines/ElectronicProperties.h"
#include "oevibe/Engine.h"

using namespace OEVIBE;

namespace {

void expect_pair(const BondOrder& pair, unsigned i, unsigned j, double order) {
    EXPECT_EQ(pair.i, i);
    EXPECT_EQ(pair.j, j);
    EXPECT_DOUBLE_EQ(pair.order, order);
}

}  // namespace

TEST(ElectronicPropertiesTest, HomoIsTheHighestOrbitalAboveHalfAnElectron) {
    EXPECT_EQ(detail::homo_index({2.0, 2.0, 2.0, 0.0, 0.0}), 2);
    // Open-shell OH from tblite: the two degenerate pi orbitals share three electrons.
    EXPECT_EQ(detail::homo_index({2.0, 2.0, 1.5, 1.5, 0.0}), 3);
    // A hydrogen atom's single electron.
    EXPECT_EQ(detail::homo_index({1.0, 0.0}), 0);
    // Fermi smearing moves a little density into the LUMO without making it occupied.
    EXPECT_EQ(detail::homo_index({2.0, 1.98, 0.02}), 1);
    // The threshold itself does not count as occupied.
    EXPECT_EQ(detail::homo_index({2.0, 0.5}), 0);
    EXPECT_EQ(detail::homo_index({2.0, 0.51}), 1);
    EXPECT_EQ(detail::homo_index({0.0, 0.0, 0.0}), -1);
    EXPECT_EQ(detail::homo_index({}), -1);
}

TEST(ElectronicPropertiesTest, WibergPairsKeepTheUpperTriangleAboveTheThreshold) {
    // tblite's matrix for water (O, H, H): two O-H bonds and a negligible H-H order.
    const std::vector<BondOrder> water =
        detail::wiberg_pairs({0.0, 0.9202, 0.9202, 0.9202, 0.0, 0.0004, 0.9202, 0.0004, 0.0}, 3);
    ASSERT_EQ(water.size(), 2u);
    expect_pair(water[0], 0, 1, 0.9202);
    expect_pair(water[1], 0, 2, 0.9202);

    // The threshold is inclusive, the diagonal is ignored, and only the upper triangle is read.
    const std::vector<BondOrder> edges =
        detail::wiberg_pairs({5.0, 0.1, 0.0999, 0.0, 5.0, 0.5, 0.0999, 0.0, 5.0}, 3);
    ASSERT_EQ(edges.size(), 2u);
    expect_pair(edges[0], 0, 1, 0.1);
    expect_pair(edges[1], 1, 2, 0.5);

    EXPECT_TRUE(detail::wiberg_pairs({0.0}, 1).empty());
    EXPECT_TRUE(detail::wiberg_pairs({}, 0).empty());
}

TEST(ElectronicPropertiesTest, NonFiniteOrdersReachValidation) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<BondOrder> pairs = detail::wiberg_pairs({0.0, nan, nan, 0.0}, 2);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_TRUE(std::isnan(pairs[0].order));
    Evaluation evaluation;
    evaluation.energy = 0.0;
    evaluation.bond_orders = pairs;
    EvaluationRequest request;
    request.properties = static_cast<unsigned>(Property::BOND_ORDERS);
    EXPECT_NE(validate_evaluation(evaluation, request, 2), "");
}

TEST(ElectronicPropertiesTest, AMatrixOfTheWrongSizeIsRejected) {
    EXPECT_THROW(detail::wiberg_pairs({0.0, 1.0, 1.0}, 2), std::invalid_argument);
    EXPECT_THROW(detail::wiberg_pairs(std::vector<double>(8, 0.0), 2), std::invalid_argument);
}
