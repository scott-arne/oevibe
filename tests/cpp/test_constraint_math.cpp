#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/ConstraintMath.h"
#include "core/Geometry.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"
#include "test_support.h"

using namespace OEVIBE;

TEST(ConstraintMathTest, ResolvesDenseAtomsAndStartingTargets) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    const MolecularSystem system(mol, 0, 1);
    ConstraintSet set;
    set.FreezeAtomIndices({0});
    set.AddDistance(2, 3);
    set.AddAngle(2, 0, 3, 100.0);
    set.AddPositionRestraint(5, 2.0);
    const std::vector<double>& x0 = system.GetConformerCoordinates(0);
    const detail::ResolvedConstraintSet resolved = detail::resolve_constraint_set(set, system, x0);
    EXPECT_EQ(resolved.frozen, (std::vector<unsigned>{0}));
    ASSERT_EQ(resolved.constraints.size(), 2u);
    EXPECT_EQ(resolved.constraints[0].atoms, (std::vector<unsigned>{1, 2}));
    EXPECT_NEAR(resolved.constraints[0].target, detail::distance(x0.data(), 1, 2, nullptr), 1e-12);
    EXPECT_NEAR(resolved.constraints[1].target, 100.0 * Units::DEGREES_TO_RADIANS, 1e-12);
    ASSERT_EQ(resolved.restraints.size(), 1u);
    EXPECT_DOUBLE_EQ(resolved.restraints[0].reference[0], x0[3 * 4]);
}

TEST(ConstraintMathTest, MissingAtomsAreInputErrors) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    const MolecularSystem system(mol, 0, 1);
    ConstraintSet set;
    set.AddDistance(1, 2);  // atom 1 was deleted
    EXPECT_THROW(detail::check_constraint_atoms(set, system), InputError);
}

TEST(ConstraintMathTest, MissingAtomsOfAMoleculeAreInputErrors) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    ConstraintSet deleted;
    deleted.AddDistance(1, 2);  // atom 1 was deleted
    EXPECT_THROW(detail::check_constraint_atoms(deleted, mol), InputError);
    ConstraintSet beyond;
    beyond.FreezeAtomIndices({6});
    EXPECT_THROW(detail::check_constraint_atoms(beyond, mol), InputError);
    ConstraintSet present;
    present.AddDistance(0, 5);
    present.FreezeAtomIndices({2});
    EXPECT_NO_THROW(detail::check_constraint_atoms(present, mol));
}

TEST(ConstraintMathTest, ResidualJacobianMatchesFiniteDifferences) {
    std::vector<double> x = test::sample_geometry();
    detail::ResolvedConstraint dihedral;
    dihedral.kind = ConstraintKind::DIHEDRAL;
    dihedral.atoms = {0, 1, 2, 3};
    dihedral.target = 0.25;
    std::vector<double> row;
    detail::constraint_residual(dihedral, x, &row);
    ASSERT_EQ(row.size(), x.size());
    const double step = 1e-6;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double saved = x[k];
        x[k] = saved + step;
        const double plus = detail::constraint_residual(dihedral, x, nullptr);
        x[k] = saved - step;
        const double minus = detail::constraint_residual(dihedral, x, nullptr);
        x[k] = saved;
        EXPECT_NEAR(row[k], (plus - minus) / (2.0 * step), 1e-6);
    }
}

TEST(ConstraintMathTest, DihedralResidualsWrap) {
    const std::vector<double> x = test::sample_geometry();
    detail::ResolvedConstraint dihedral;
    dihedral.kind = ConstraintKind::DIHEDRAL;
    dihedral.atoms = {0, 1, 2, 3};
    const double value = detail::dihedral(x.data(), 0, 1, 2, 3, nullptr);
    // The target lies 0.4 rad past +pi, so it wraps to the far side of the discontinuity and
    // the raw difference exceeds pi; only a wrapped residual is small.
    dihedral.target = detail::wrap_angle(value + 0.4);
    ASSERT_GT(value - dihedral.target, Units::PI);
    EXPECT_NEAR(detail::constraint_residual(dihedral, x, nullptr),
                value - dihedral.target - 2.0 * Units::PI, 1e-12);
}

TEST(ConstraintMathTest, ResidualSummaryAndTolerances) {
    const std::vector<double> x = test::sample_geometry();
    detail::ResolvedConstraint bond;
    bond.kind = ConstraintKind::DISTANCE;
    bond.atoms = {0, 1};
    bond.target = 1.530 - 5e-5;
    detail::ResolvedConstraint bend;
    bend.kind = ConstraintKind::ANGLE;
    bend.atoms = {0, 1, 2};
    bend.target = detail::angle(x.data(), 0, 1, 2, nullptr) + 0.01 * Units::DEGREES_TO_RADIANS;
    const detail::ResidualSummary summary = detail::summarize_residuals({bond, bend}, x);
    EXPECT_NEAR(summary.max_distance, 5e-5, 1e-12);
    EXPECT_NEAR(summary.max_angle, 0.01, 1e-9);
    EXPECT_FALSE(detail::residuals_satisfied(summary));
    EXPECT_TRUE(detail::residuals_satisfied(detail::summarize_residuals({bond}, x)));
}

TEST(ConstraintMathTest, RestraintEnergyAndGradient) {
    std::vector<double> x = test::sample_geometry();
    std::vector<detail::ResolvedRestraint> restraints(4);
    restraints[0].kind = RestraintKind::DISTANCE;
    restraints[0].atoms = {0, 3};
    restraints[0].force_constant = 10.0;
    restraints[0].target = 3.0;
    restraints[1].kind = RestraintKind::ANGLE;
    restraints[1].atoms = {4, 0, 2};
    restraints[1].force_constant = 5.0;
    restraints[1].target = 1.5;
    restraints[2].kind = RestraintKind::DIHEDRAL;
    restraints[2].atoms = {0, 1, 2, 3};
    restraints[2].force_constant = 2.0;
    // Across the discontinuity from the geometry's dihedral (about 2.86 rad): the wrapped
    // deviation is about -0.42 rad, far from the +-pi jump the finite-difference step must avoid.
    restraints[2].target = -3.0;
    restraints[3].kind = RestraintKind::POSITION;
    restraints[3].atoms = {5};
    restraints[3].force_constant = 1.0;
    restraints[3].reference = {{2.0, -0.5, -1.0}};

    const double r = detail::distance(x.data(), 0, 3, nullptr);
    const double expected_distance_term = 0.5 * 10.0 * (r - 3.0) * (r - 3.0);
    std::vector<double> gradient(x.size(), 0.0);
    const double energy = detail::restraint_energy(restraints, x, &gradient);
    EXPECT_GT(energy, expected_distance_term);

    // The gradient check below cannot catch a missing wrap, because an unwrapped energy and its
    // gradient are still consistent with each other, so pin the dihedral term's value.
    const double phi = detail::dihedral(x.data(), 0, 1, 2, 3, nullptr);
    const double wrapped_deviation = phi - restraints[2].target - 2.0 * Units::PI;
    ASSERT_GT(phi - restraints[2].target, Units::PI);
    EXPECT_NEAR(detail::restraint_energy({restraints[2]}, x, nullptr),
                0.5 * restraints[2].force_constant * wrapped_deviation * wrapped_deviation, 1e-12);

    const double step = 1e-6;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double saved = x[k];
        x[k] = saved + step;
        const double plus = detail::restraint_energy(restraints, x, nullptr);
        x[k] = saved - step;
        const double minus = detail::restraint_energy(restraints, x, nullptr);
        x[k] = saved;
        EXPECT_NEAR(gradient[k], (plus - minus) / (2.0 * step), 1e-5);
    }
}
