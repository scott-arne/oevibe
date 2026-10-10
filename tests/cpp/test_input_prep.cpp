#include <gtest/gtest.h>

#include <oechem.h>

#include <cmath>

#include "core/InputPrep.h"
#include "oevibe/Error.h"
#include "oevibe/Options.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

double distance(const OEChem::OEMolBase& conf, const OEChem::OEAtomBase* a,
                const OEChem::OEAtomBase* b) {
    double pa[3];
    double pb[3];
    conf.GetCoords(a, pa);
    conf.GetCoords(b, pb);
    return std::sqrt((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]) +
                     (pa[2] - pb[2]) * (pa[2] - pb[2]));
}

}  // namespace

TEST(InputPrepTest, RejectsEmptyAnd2DMolecules) {
    OEChem::OEGraphMol empty;
    EXPECT_THROW(detail::prepare_input(empty, CalculationOptions()), InputError);
    OEChem::OEGraphMol flat;
    OEChem::OESmilesToMol(flat, "CCO");
    EXPECT_THROW(detail::prepare_input(flat, CalculationOptions()), InputError);
}

TEST(InputPrepTest, AddsHydrogensWithAWarning) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    OEChem::OESuppressHydrogens(mol);
    const detail::PreparedInput prepared = detail::prepare_input(mol, CalculationOptions());
    EXPECT_FALSE(OEChem::OEHasImplicitHydrogens(mol));
    EXPECT_EQ(mol.NumAtoms(), 9u);
    ASSERT_EQ(prepared.warnings.size(), 1u);
    EXPECT_NE(prepared.warnings[0].find("added 6 explicit hydrogens"), std::string::npos);
}

TEST(InputPrepTest, RefusesHydrogensWhenDisabledOrUsingInputCharges) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    OEChem::OESuppressHydrogens(mol);
    CalculationOptions no_add;
    no_add.SetAddHydrogens(false);
    EXPECT_THROW(detail::prepare_input(mol, no_add), InputError);
    CalculationOptions input_charges;
    input_charges.SetUseInputPartialCharges(true);
    EXPECT_THROW(detail::prepare_input(mol, input_charges), InputError);
}

TEST(InputPrepTest, ChargeAndMultiplicity) {
    OEChem::OEMol ethanol = test::make_conformers("CCO");
    detail::PreparedInput neutral = detail::prepare_input(ethanol, CalculationOptions());
    EXPECT_EQ(neutral.charge, 0);
    EXPECT_EQ(neutral.multiplicity, 1u);
    OEChem::OEMol ammonium = test::make_conformers("C[NH3+]");
    EXPECT_EQ(detail::prepare_input(ammonium, CalculationOptions()).charge, 1);
    CalculationOptions cation;
    cation.SetCharge(1);
    EXPECT_EQ(detail::prepare_input(ethanol, cation).multiplicity, 2u);
    CalculationOptions bad_spin;
    bad_spin.SetMultiplicity(2);
    EXPECT_THROW(detail::prepare_input(ethanol, bad_spin), InputError);
}

// Spike S3 showed OEAddExplicitHydrogens places hydrogens in every conformer.
TEST(InputPrepTest, AddsHydrogensToEveryConformer) {
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    ASSERT_GT(mol.NumConfs(), 1u);
    OEChem::OESuppressHydrogens(mol);
    detail::prepare_input(mol, CalculationOptions());
    EXPECT_EQ(mol.NumAtoms(), 15u);
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf) {
        unsigned hydrogens = 0;
        for (OESystem::OEIter<OEChem::OEAtomBase> h = conf->GetAtoms(OEChem::OEIsHydrogen()); h;
             ++h) {
            OESystem::OEIter<OEChem::OEAtomBase> heavy = h->GetAtoms();
            const double d = distance(*conf, h, heavy);
            EXPECT_GT(d, 0.9);
            EXPECT_LT(d, 1.2);
            ++hydrogens;
        }
        EXPECT_EQ(hydrogens, 10u);
    }
}
