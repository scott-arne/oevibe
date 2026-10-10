#include <gtest/gtest.h>

#include <oechem.h>

#include <vector>

#include "oevibe/Constraints.h"
#include "oevibe/Error.h"

using namespace OEVIBE;

TEST(ConstraintSetTest, StartsEmpty) {
    const ConstraintSet set;
    EXPECT_TRUE(set.IsEmpty());
    EXPECT_FALSE(set.HasFrozenAtoms());
    EXPECT_FALSE(set.HasExactConstraints());
    EXPECT_FALSE(set.HasRestraints());
}

TEST(ConstraintSetTest, FrozenAtomsAreSortedAndUnique) {
    ConstraintSet set;
    set.FreezeAtomIndices({5, 2, 5, 1});
    EXPECT_EQ(set.GetFrozenAtoms(), (std::vector<unsigned>{1, 2, 5}));
    EXPECT_TRUE(set.HasFrozenAtoms());
}

TEST(ConstraintSetTest, FreezeAtomsEvaluatesThePredicateNow) {
    OEChem::OEGraphMol mol;
    OEChem::OESmilesToMol(mol, "CCO");
    OEChem::OEAddExplicitHydrogens(mol);
    ConstraintSet set;
    set.FreezeAtoms(mol, OEChem::OEIsHeavy());
    EXPECT_EQ(set.GetFrozenAtoms().size(), 3u);
}

TEST(ConstraintSetTest, TargetsAreValidatedAndNormalized) {
    ConstraintSet set;
    set.AddDihedral(0, 1, 2, 3, 190.0);
    ASSERT_EQ(set.GetConstraints().size(), 1u);
    EXPECT_TRUE(set.GetConstraints()[0].has_target);
    EXPECT_DOUBLE_EQ(set.GetConstraints()[0].target, -170.0);
    set.AddDistance(0, 1);
    EXPECT_FALSE(set.GetConstraints()[1].has_target);
    EXPECT_THROW(set.AddAngle(0, 1, 2, 181.0), OptionsError);
    EXPECT_THROW(set.AddDistance(0, 1, 0.0), OptionsError);
    EXPECT_THROW(set.AddDistance(1, 1), OptionsError);
    EXPECT_TRUE(set.HasExactConstraints());
}

TEST(ConstraintSetTest, RestraintsNeedPositiveForceConstants) {
    ConstraintSet set;
    set.AddDistanceRestraint(0, 1, 10.0);
    set.AddAngleRestraint(0, 1, 2, 5.0, 109.5);
    set.AddDihedralRestraint(0, 1, 2, 3, 2.0, -60.0);
    set.AddPositionRestraint(4, 1.0);
    EXPECT_EQ(set.GetRestraints().size(), 4u);
    EXPECT_EQ(set.GetRestraints()[3].kind, RestraintKind::POSITION);
    EXPECT_THROW(set.AddDistanceRestraint(0, 1, 0.0), OptionsError);
    EXPECT_THROW(set.AddPositionRestraint(4, -1.0), OptionsError);
    EXPECT_FALSE(set.HasExactConstraints());
    EXPECT_TRUE(set.HasRestraints());
}

TEST(ConstraintSetTest, ReferencedAtomsCoverEveryTerm) {
    ConstraintSet set;
    set.FreezeAtomIndices({9});
    set.AddAngle(3, 1, 2);
    set.AddPositionRestraint(7, 1.0);
    EXPECT_EQ(set.GetReferencedAtoms(), (std::vector<unsigned>{1, 2, 3, 7, 9}));
}

TEST(ConstraintSetTest, NormalizeDihedral) {
    EXPECT_DOUBLE_EQ(normalize_dihedral_degrees(180.0), 180.0);
    EXPECT_DOUBLE_EQ(normalize_dihedral_degrees(-180.0), 180.0);
    EXPECT_DOUBLE_EQ(normalize_dihedral_degrees(540.0), 180.0);
    EXPECT_DOUBLE_EQ(normalize_dihedral_degrees(-190.0), 170.0);
    EXPECT_DOUBLE_EQ(normalize_dihedral_degrees(30.0), 30.0);
}
