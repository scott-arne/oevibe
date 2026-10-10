#include <gtest/gtest.h>

#include <limits>

#include "oevibe/Error.h"
#include "oevibe/Options.h"

using namespace OEVIBE;

namespace {
constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();
constexpr unsigned CHARGES_BIT = static_cast<unsigned>(Property::CHARGES);
}  // namespace

// Spec 5.5 defaults.
TEST(CalculationOptionsTest, DefaultsMatchSpec) {
    const CalculationOptions opts;
    EXPECT_EQ(opts.GetMethod().GetName(), "gfn2-xtb");
    EXPECT_FALSE(opts.HasEngine());
    EXPECT_TRUE(opts.GetSolvent().IsGasPhase());
    EXPECT_EQ(opts.GetPreset(), Preset::STANDARD);
    EXPECT_FALSE(opts.HasCharge());
    EXPECT_FALSE(opts.HasMultiplicity());
    EXPECT_EQ(opts.GetProperties(), 0u);
    EXPECT_TRUE(opts.GetAddHydrogens());
    EXPECT_TRUE(opts.GetUpdateCoordinates());
    EXPECT_TRUE(opts.GetWriteTags());
    EXPECT_FALSE(opts.GetAssignPartialCharges());
    EXPECT_FALSE(opts.GetUseInputPartialCharges());
    EXPECT_TRUE(opts.GetKeywords().empty());
}

TEST(CalculationOptionsTest, FromPresetSetsThePreset) {
    EXPECT_EQ(CalculationOptions::FromPreset(Preset::THOROUGH).GetPreset(), Preset::THOROUGH);
}

TEST(CalculationOptionsTest, SettersValidate) {
    CalculationOptions opts;
    EXPECT_THROW(opts.SetMultiplicity(0), OptionsError);
    EXPECT_THROW(opts.SetProperties(16u), OptionsError);
    EXPECT_THROW(opts.SetEngine(""), OptionsError);
    EXPECT_THROW(opts.SetKeyword("", "1"), OptionsError);
    EXPECT_THROW(opts.GetKeyword("missing"), OptionsError);
    EXPECT_THROW(opts.GetCharge(), OptionsError);
    EXPECT_THROW(opts.GetEngine(), OptionsError);
    EXPECT_THROW(opts.SetPreset(static_cast<Preset>(5)), OptionsError);
    opts.SetKeyword("accuracy", "0.5");
    EXPECT_TRUE(opts.HasKeyword("accuracy"));
    EXPECT_EQ(opts.GetKeyword("accuracy"), "0.5");
    opts.ClearKeyword("accuracy");
    EXPECT_FALSE(opts.HasKeyword("accuracy"));
}

TEST(CalculationOptionsTest, AssignPartialChargesImpliesTheChargesProperty) {
    CalculationOptions opts;
    EXPECT_EQ(opts.GetEffectiveProperties() & CHARGES_BIT, 0u);
    opts.SetAssignPartialCharges(true);
    EXPECT_EQ(opts.GetEffectiveProperties() & CHARGES_BIT, CHARGES_BIT);
    EXPECT_EQ(opts.GetProperties() & CHARGES_BIT, 0u);
}

TEST(OptimizerOptionsTest, OptionalFieldsStartUnset) {
    const OptimizerOptions opts;
    EXPECT_EQ(opts.GetAlgorithm(), Algorithm::AUTO);
    EXPECT_FALSE(opts.HasMaxIterations());
    EXPECT_FALSE(opts.HasEnergyChangeThreshold());
    EXPECT_FALSE(opts.HasGradientNormThreshold());
    EXPECT_FALSE(opts.HasInitialTrustRadius());
    EXPECT_FALSE(opts.HasMaxTrustRadius());
    EXPECT_TRUE(opts.GetConstraints().IsEmpty());
    EXPECT_THROW(opts.GetMaxIterations(), OptionsError);
}

TEST(OptimizerOptionsTest, SettersValidate) {
    OptimizerOptions opts;
    EXPECT_THROW(opts.SetMaxIterations(0), OptionsError);
    EXPECT_THROW(opts.SetEnergyChangeThreshold(-1.0), OptionsError);
    EXPECT_THROW(opts.SetGradientNormThreshold(NAN_VALUE), OptionsError);
    EXPECT_THROW(opts.SetInitialTrustRadius(0.0), OptionsError);
    EXPECT_THROW(opts.SetAlgorithm(static_cast<Algorithm>(4)), OptionsError);
}

TEST(OptimizerOptionsTest, LbfgsRejectsExactConstraints) {
    OptimizerOptions opts;
    ConstraintSet set;
    set.AddDistance(0, 1);
    opts.SetConstraints(set);
    opts.SetAlgorithm(Algorithm::LBFGS);
    EXPECT_THROW(opts.Validate(), OptionsError);
}

TEST(OptimizerOptionsTest, TrustRadiiMustBeOrdered) {
    OptimizerOptions opts;
    opts.SetInitialTrustRadius(0.5);
    opts.SetMaxTrustRadius(0.2);
    EXPECT_THROW(opts.Validate(), OptionsError);
}

TEST(ResolveOptimizerTest, PresetFillsUnsetFields) {
    const ResolvedOptimizerSettings resolved =
        resolve_optimizer_settings(OptimizerOptions(), Preset::STANDARD, 10, false);
    const PresetValues standard = preset_values(Preset::STANDARD);
    EXPECT_EQ(resolved.algorithm, Algorithm::ANC_RFO);
    EXPECT_EQ(resolved.max_iterations, 200u);
    EXPECT_DOUBLE_EQ(resolved.energy_change, standard.energy_change);
    EXPECT_DOUBLE_EQ(resolved.gradient_norm, standard.gradient_norm);
    EXPECT_DOUBLE_EQ(resolved.initial_trust_radius, 0.1);
    EXPECT_DOUBLE_EQ(resolved.max_trust_radius, 0.3);
}

TEST(ResolveOptimizerTest, ExplicitFieldsWin) {
    OptimizerOptions opts;
    opts.SetMaxIterations(7);
    opts.SetGradientNormThreshold(0.5);
    const ResolvedOptimizerSettings resolved =
        resolve_optimizer_settings(opts, Preset::THOROUGH, 10, false);
    EXPECT_EQ(resolved.max_iterations, 7u);
    EXPECT_DOUBLE_EQ(resolved.gradient_norm, 0.5);
    EXPECT_DOUBLE_EQ(resolved.energy_change, preset_values(Preset::THOROUGH).energy_change);
}

TEST(ResolveOptimizerTest, AutoPicksLbfgsOnlyForForceFieldsWithoutExactConstraints) {
    OptimizerOptions opts;
    EXPECT_EQ(resolve_optimizer_settings(opts, Preset::STANDARD, 10, true).algorithm,
              Algorithm::LBFGS);
    ConstraintSet frozen_and_restrained;
    frozen_and_restrained.FreezeAtomIndices({0});
    frozen_and_restrained.AddDistanceRestraint(1, 2, 5.0);
    opts.SetConstraints(frozen_and_restrained);
    EXPECT_EQ(resolve_optimizer_settings(opts, Preset::STANDARD, 10, true).algorithm,
              Algorithm::LBFGS);
    ConstraintSet exact;
    exact.AddDistance(1, 2);
    opts.SetConstraints(exact);
    EXPECT_EQ(resolve_optimizer_settings(opts, Preset::STANDARD, 10, true).algorithm,
              Algorithm::ANC_RFO);
    EXPECT_EQ(resolve_optimizer_settings(OptimizerOptions(), Preset::STANDARD, 10, false).algorithm,
              Algorithm::ANC_RFO);
}

TEST(ResolveOptimizerTest, InconsistentTrustRadiiAfterResolution) {
    OptimizerOptions opts;
    opts.SetInitialTrustRadius(0.5);
    EXPECT_THROW(resolve_optimizer_settings(opts, Preset::STANDARD, 10, false), OptionsError);
}
