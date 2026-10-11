#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <vector>

#include "oevibe/Error.h"
#include "oevibe/Result.h"
#include "oevibe/Units.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

ConformerResult ok_result(unsigned index, double energy) {
    ConformerResult result;
    result.SetConformerIndex(index);
    result.SetEnergy(energy);
    result.SetStatus(ConformerStatus::OK, "");
    return result;
}

}  // namespace

TEST(ConformerResultTest, AbsentValuesThrow) {
    const ConformerResult result;
    EXPECT_FALSE(result.HasEnergy());
    EXPECT_THROW(result.GetEnergy(), OEVibeError);
    EXPECT_THROW(result.GetGradient(), OEVibeError);
    EXPECT_THROW(result.GetOptimizationStats(), OEVibeError);
    EXPECT_FALSE(result.HasRelativeEnergy());
}

TEST(ConformerResultTest, ChargesUseTheOpenEyeLayout) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    ConformerResult result;
    result.SetCharges({-0.4, 0.0, 0.1, 0.1, 0.1, 0.1});
    OESystem::OEIter<OEChem::OEAtomBase> carbon = mol.GetAtoms();
    EXPECT_DOUBLE_EQ(result.GetCharge(*carbon), -0.4);
    OEChem::OEAtomBase* hydrogen = mol.GetAtom(OEChem::OEHasAtomIdx(2));
    ASSERT_TRUE(hydrogen != nullptr);
    EXPECT_DOUBLE_EQ(result.GetCharge(*hydrogen), 0.1);
}

TEST(ConformerResultTest, OrbitalFrontier) {
    ConformerResult result;
    result.SetOrbitalEnergies({-12.0, -9.0, 1.0}, 1);
    EXPECT_TRUE(result.HasHOMO());
    EXPECT_DOUBLE_EQ(result.GetHOMO(), -9.0);
    EXPECT_TRUE(result.HasLUMO());
    EXPECT_DOUBLE_EQ(result.GetLUMO(), 1.0);
    result.SetOrbitalEnergies({-12.0, -9.0}, 1);
    EXPECT_FALSE(result.HasLUMO());
    EXPECT_THROW(result.GetLUMO(), OEVibeError);
}

TEST(ConformerResultTest, BondOrdersAreSymmetric) {
    ConformerResult result;
    result.SetBondOrders({{0, 2, 1.0}, {2, 3, 1.5}});
    EXPECT_TRUE(result.HasBondOrders());
    EXPECT_DOUBLE_EQ(result.GetBondOrder(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(result.GetBondOrder(3, 2), 1.5);
    EXPECT_DOUBLE_EQ(result.GetBondOrder(0, 3), 0.0);
    EXPECT_EQ(result.GetBondOrders().size(), 2u);
}

TEST(ConformerResultTest, InvalidHomoIndexIsRejected) {
    ConformerResult result;
    EXPECT_THROW(result.SetOrbitalEnergies({-12.0, -9.0}, 2), OEVibeError);
    EXPECT_THROW(result.SetOrbitalEnergies({-12.0, -9.0}, -2), OEVibeError);
    result.SetOrbitalEnergies({-12.0, -9.0}, -1);
    EXPECT_FALSE(result.HasHOMO());
    EXPECT_TRUE(result.HasLUMO());
    EXPECT_DOUBLE_EQ(result.GetLUMO(), -12.0);
}

TEST(ResultSetTest, RelativeEnergiesCoverOkConformersOnly) {
    ResultSet results;
    results.Add(ok_result(0, -10.0));
    results.Add(ok_result(1, -12.0));
    ConformerResult failed;
    failed.SetConformerIndex(2);
    failed.SetEnergy(-20.0);
    failed.SetStatus(ConformerStatus::NOT_CONVERGED, "iteration limit");
    results.Add(failed);
    results.Finalize();
    const std::map<unsigned, double> relative = results.GetRelativeEnergies();
    ASSERT_EQ(relative.size(), 2u);
    EXPECT_DOUBLE_EQ(relative.at(0), 2.0);
    EXPECT_DOUBLE_EQ(relative.at(1), 0.0);
    EXPECT_FALSE(results.Get(2).HasRelativeEnergy());
    EXPECT_EQ(results.GetLowest().GetConformerIndex(), 1u);
    EXPECT_THROW(results.Get(3), OEVibeError);
}

TEST(ResultSetTest, LowestBreaksTiesByConformerIndex) {
    ResultSet results;
    results.Add(ok_result(5, -1.0));
    results.Add(ok_result(2, -1.0));
    results.Finalize();
    EXPECT_EQ(results.GetLowest().GetConformerIndex(), 2u);
}

TEST(ResultSetTest, BoltzmannWeightsFromElectronicEnergies) {
    const double temperature = 298.15;
    const double gap = Units::GAS_CONSTANT_KCAL * temperature * std::log(2.0);
    ResultSet results;
    results.Add(ok_result(0, 0.0));
    results.Add(ok_result(1, gap));
    results.Finalize();
    const std::map<unsigned, double> weights = results.GetBoltzmannWeights(temperature);
    EXPECT_NEAR(weights.at(0), 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(weights.at(1), 1.0 / 3.0, 1e-12);
    EXPECT_THROW(results.GetBoltzmannWeights(0.0), OEVibeError);
}

TEST(ResultSetTest, NoOkConformerMeansEmptyAggregates) {
    ResultSet results;
    ConformerResult failed;
    failed.SetStatus(ConformerStatus::ENGINE_ERROR, "boom");
    results.Add(failed);
    results.Finalize();
    EXPECT_FALSE(results.HasLowest());
    EXPECT_THROW(results.GetLowest(), OEVibeError);
    EXPECT_TRUE(results.GetRelativeEnergies().empty());
    EXPECT_TRUE(results.GetBoltzmannWeights(298.15).empty());
}

TEST(ResultSetTest, StatusNames) {
    EXPECT_EQ(conformer_status_name(ConformerStatus::OK), "OK");
    EXPECT_EQ(conformer_status_name(ConformerStatus::NOT_CONVERGED), "NOT_CONVERGED");
    EXPECT_EQ(conformer_status_name(ConformerStatus::ENGINE_ERROR), "ENGINE_ERROR");
    EXPECT_EQ(conformer_status_name(ConformerStatus::UNSUPPORTED), "UNSUPPORTED");
}

TEST(ResultSetTest, MixedMethodsCannotBeCompared) {
    ResultSet results;
    ConformerResult result0 = ok_result(0, -10.0);
    Provenance prov0;
    prov0.method = "mmff94";
    result0.SetProvenance(prov0);
    results.Add(result0);

    ConformerResult result1 = ok_result(1, -12.0);
    Provenance prov1;
    prov1.method = "sage";
    result1.SetProvenance(prov1);
    results.Add(result1);

    EXPECT_THROW(results.Finalize(), OEVibeError);

    ResultSet results2;
    results2.Add(result0);
    results2.Add(result1);
    EXPECT_THROW(results2.GetLowest(), OEVibeError);

    ResultSet results3;
    results3.Add(result0);
    results3.Add(result1);
    EXPECT_THROW(results3.GetBoltzmannWeights(298.15), OEVibeError);
}

TEST(ResultSetTest, MixedSolventModelsCannotBeCompared) {
    ResultSet results;
    ConformerResult result0 = ok_result(0, -10.0);
    Provenance prov0;
    prov0.method = "mmff94";
    prov0.solvent_model = "gas";
    result0.SetProvenance(prov0);
    results.Add(result0);

    ConformerResult result1 = ok_result(1, -12.0);
    Provenance prov1;
    prov1.method = "mmff94";
    prov1.solvent_model = "sheffield";
    result1.SetProvenance(prov1);
    results.Add(result1);

    EXPECT_THROW(results.GetLowest(), OEVibeError);
}
