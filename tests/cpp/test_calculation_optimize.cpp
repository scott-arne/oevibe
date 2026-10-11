#include <gtest/gtest.h>

#include <oechem.h>

#include <cmath>
#include <vector>

#include "oevibe/Calculation.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions mmff94() {
    CalculationOptions options;
    options.SetMethod(Method(MMFF94));
    return options;
}

std::vector<double> coordinates_of(const OEChem::OEMolBase& conf) {
    std::vector<double> coords(3 * conf.GetMaxAtomIdx());
    conf.GetCoords(coords.data());
    return coords;
}

void jitter(OEChem::OEMol& mol) {
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf) {
        conf->SetCoords(test::perturbed(coordinates_of(*conf), 0.08).data());
    }
}

// OpenEye stores coordinates in single precision, so written-back coordinates match the
// double-precision result only to about 1e-6 Angstrom.
void expect_close(const std::vector<double>& a, const std::vector<double>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        EXPECT_NEAR(a[k], b[k], 1e-4);
    }
}

}  // namespace

TEST(CalculationOptimizeTest, Mmff94LowersEveryConformerAndWritesCoordinates) {
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    jitter(mol);
    OEChem::OEMol reference(mol);
    const ResultSet singles = Calculation(mmff94()).Energy(reference);
    std::vector<std::vector<double>> before;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf) {
        before.push_back(coordinates_of(*conf));
    }
    const ResultSet results = Calculation(mmff94()).Optimize(mol, OptimizerOptions());
    unsigned position = 0;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf, ++position) {
        const ConformerResult& result = results.Get(position);
        ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
        EXPECT_LE(result.GetEnergy(), singles.Get(position).GetEnergy() + 1e-6);
        EXPECT_TRUE(result.GetOptimizationStats().converged);
        expect_close(coordinates_of(*conf), result.GetCoordinates());
        EXPECT_NE(coordinates_of(*conf), before[position]);
    }
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("optimizer.algorithm"), "lbfgs");
}

TEST(CalculationOptimizeTest, UpdateCoordinatesFalseKeepsTheInput) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    CalculationOptions options = mmff94();
    options.SetUpdateCoordinates(false);
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    EXPECT_EQ(coordinates_of(*mol.GetActive()), before);
    EXPECT_TRUE(results.Get(0).HasCoordinates());
    EXPECT_NE(results.Get(0).GetCoordinates(), before);
}

TEST(CalculationOptimizeTest, FrozenAtomsStayPut) {
    OEChem::OEMol mol = test::make_conformers("CCCCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    ConstraintSet constraints;
    constraints.FreezeAtoms(mol, OEChem::OEIsHeavy());
    OptimizerOptions options;
    options.SetConstraints(constraints);
    Calculation(mmff94()).Optimize(mol, options);
    const std::vector<double> after = coordinates_of(*mol.GetActive());
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(OEChem::OEIsHeavy()); atom;
         ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(after[3 * atom->GetIdx() + axis], before[3 * atom->GetIdx() + axis], 1e-6);
        }
    }
}

TEST(CalculationOptimizeTest, DihedralConstraintIsHeld) {
    OEChem::OEMol mol = test::make_conformers("CCCC");
    std::vector<unsigned> carbons;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(OEChem::OEIsHeavy()); atom;
         ++atom) {
        carbons.push_back(atom->GetIdx());
    }
    ConstraintSet constraints;
    constraints.AddDihedral(carbons[0], carbons[1], carbons[2], carbons[3], 60.0);
    OptimizerOptions options;
    options.SetConstraints(constraints);
    const ResultSet results = Calculation(mmff94()).Optimize(mol, options);
    ASSERT_EQ(results.Get(0).GetStatus(), ConformerStatus::OK) << results.Get(0).GetMessage();
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("optimizer.algorithm"), "anc_rfo");
    OEChem::OEConfBase* conf = mol.GetActive();
    const double torsion =
        OEChem::OEGetTorsion(*conf, conf->GetAtom(OEChem::OEHasAtomIdx(carbons[0])),
                             conf->GetAtom(OEChem::OEHasAtomIdx(carbons[1])),
                             conf->GetAtom(OEChem::OEHasAtomIdx(carbons[2])),
                             conf->GetAtom(OEChem::OEHasAtomIdx(carbons[3])));
    EXPECT_NEAR(torsion * Units::RADIANS_TO_DEGREES, 60.0, 1e-3);
}

TEST(CalculationOptimizeTest, OptionAndInputErrorsComeFirst) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    OptimizerOptions lbfgs;
    ConstraintSet exact;
    exact.AddDistance(0, 1);
    lbfgs.SetConstraints(exact);
    lbfgs.SetAlgorithm(Algorithm::LBFGS);
    EXPECT_THROW(Calculation(mmff94()).Optimize(mol, lbfgs), OptionsError);
    OptimizerOptions missing;
    ConstraintSet far;
    far.AddDistance(0, 999);
    missing.SetConstraints(far);
    EXPECT_THROW(Calculation(mmff94()).Optimize(mol, missing), InputError);
}

TEST(CalculationOptimizeTest, IterationLimitGivesNotConverged) {
    OEChem::OEMol mol = test::make_conformers("CCCCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    OptimizerOptions options;
    options.SetMaxIterations(1);
    const ResultSet results = Calculation(mmff94()).Optimize(mol, options);
    EXPECT_EQ(results.Get(0).GetStatus(), ConformerStatus::NOT_CONVERGED);
    EXPECT_FALSE(results.Get(0).HasRelativeEnergy());
    EXPECT_TRUE(results.Get(0).HasEnergy());
    EXPECT_NE(coordinates_of(*mol.GetActive()), before);
}

TEST(CalculationOptimizeTest, NonForceFieldEnginesUseAncRfo) {
    test::register_fake("optimize-fake");
    CalculationOptions options;
    options.SetMethod(Method("fake-method"));
    options.SetEngine("optimize-fake");
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    // At the STANDARD gradient-norm threshold the residual spring energy is a few hundredths.
    for (std::size_t i = 0; i < results.Size(); ++i) {
        ASSERT_EQ(results.Get(i).GetStatus(), ConformerStatus::OK) << results.Get(i).GetMessage();
        EXPECT_LT(results.Get(i).GetEnergy(), 0.05);
    }
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("optimizer.algorithm"), "anc_rfo");
}
