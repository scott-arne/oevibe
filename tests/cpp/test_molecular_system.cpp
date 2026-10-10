#include <gtest/gtest.h>

#include <oechem.h>

#include <cmath>
#include <vector>

#include "oevibe/Error.h"
#include "oevibe/MolecularSystem.h"
#include "test_support.h"

using namespace OEVIBE;

TEST(MolecularSystemTest, SnapshotIsDense) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    const MolecularSystem system(mol, 0, 1);
    EXPECT_EQ(system.NumAtoms(), 5u);
    EXPECT_EQ(system.GetMaxOEAtomIndex(), 6u);
    EXPECT_EQ(system.GetOEAtomIndex(1), 2u);
    EXPECT_EQ(system.GetDenseIndex(2), 1u);
    EXPECT_FALSE(system.HasDenseIndex(1));
    EXPECT_THROW(system.GetDenseIndex(1), InputError);
    EXPECT_EQ(system.GetAtomicNumbers(), (std::vector<unsigned>{6, 1, 1, 1, 1}));
    EXPECT_EQ(system.GetMolecule().GetMaxAtomIdx(), 5u);
    EXPECT_EQ(system.GetMolecule().NumBonds(), 4u);
}

TEST(MolecularSystemTest, CoordinateLayoutsRoundTrip) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    const MolecularSystem system(mol, 0, 1);
    std::vector<double> original(3 * mol.GetMaxAtomIdx());
    mol.GetCoords(original.data());
    EXPECT_EQ(system.ToOELayout(system.GetConformerCoordinates(0)), original);
    const std::vector<double> scalars = system.ToOEScalarLayout({1.0, 2.0, 3.0, 4.0, 5.0});
    EXPECT_EQ(scalars, (std::vector<double>{1.0, 0.0, 2.0, 3.0, 4.0, 5.0}));
}

TEST(MolecularSystemTest, KeepsEveryConformerInOrder) {
    const OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    const MolecularSystem system(mol, 0, 1);
    ASSERT_EQ(system.NumConformers(), mol.NumConfs());
    unsigned position = 0;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf, ++position) {
        EXPECT_EQ(system.GetConformerIndices()[position], conf->GetIdx());
        std::vector<double> coords(3 * mol.GetMaxAtomIdx());
        conf->GetCoords(coords.data());
        EXPECT_EQ(system.ToOELayout(system.GetConformerCoordinates(position)), coords);
    }
}

TEST(MolecularSystemTest, FormalChargeSum) {
    const OEChem::OEMol mol = test::make_conformers("C[NH3+]");
    EXPECT_EQ(MolecularSystem(mol, 1, 1).GetFormalChargeSum(), 1);
}

TEST(MolecularSystemTest, AssignsAm1BccElf10WithoutTouchingTheInput) {
    const OEChem::OEMol mol = test::make_conformers("CCO", 2);
    const MolecularSystem system(mol, 0, 1, PartialChargePolicy::ASSIGN_AM1BCC_ELF10);
    double sum = 0.0;
    double largest = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = system.GetMolecule().GetAtoms(); atom;
         ++atom) {
        sum += atom->GetPartialCharge();
        largest = std::max(largest, std::abs(atom->GetPartialCharge()));
    }
    EXPECT_NEAR(sum, 0.0, 1e-3);
    EXPECT_GT(largest, 0.1);
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        EXPECT_DOUBLE_EQ(atom->GetPartialCharge(), 0.0);
    }
}

TEST(MolecularSystemTest, InputChargesMustBeFiniteAndSumToTheCharge) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EXPECT_NO_THROW(MolecularSystem(mol, 0, 1, PartialChargePolicy::USE_INPUT));
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        atom->SetPartialCharge(0.1);
    }
    EXPECT_THROW(MolecularSystem(mol, 0, 1, PartialChargePolicy::USE_INPUT), InputError);
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        atom->SetPartialCharge(std::nan(""));
    }
    EXPECT_THROW(MolecularSystem(mol, 0, 1, PartialChargePolicy::USE_INPUT), InputError);
}

TEST(MolecularSystemTest, CopyMoleculeGivesEveryConformer) {
    const OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    const MolecularSystem system(mol, 0, 1);
    OEChem::OEMol out;
    system.CopyMolecule(out);
    EXPECT_EQ(out.NumConfs(), system.NumConformers());
    EXPECT_EQ(out.NumAtoms(), system.NumAtoms());
}
