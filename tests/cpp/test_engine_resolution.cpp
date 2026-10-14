#include <gtest/gtest.h>

#include "core/EngineResolution.h"
#include "oevibe/Error.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions options_for(const std::string& engine, const std::string& method) {
    CalculationOptions options;
    options.SetEngine(engine);
    options.SetMethod(Method(method));
    return options;
}

}  // namespace

TEST(EngineResolutionTest, ResolvesTheEngineMethodAndSettings) {
    auto fake = test::register_fake("resolution-fake");
    const detail::ResolvedEngine resolved =
        detail::resolve_engine(options_for("resolution-fake", "fake-method"));
    EXPECT_EQ(resolved.name, "resolution-fake");
    EXPECT_EQ(resolved.version, "1.0");
    EXPECT_EQ(resolved.settings.method.GetName(), "fake-method");
    EXPECT_TRUE(resolved.settings.solvent.IsGasPhase());
    EXPECT_TRUE(resolved.solvent_warning.empty());
    EXPECT_EQ(resolved.engine.get(), fake.get());
}

TEST(EngineResolutionTest, AutoSolventResolvesToTheMethodDefaultAndWarns) {
    test::register_fake("resolution-solvent");
    CalculationOptions options = options_for("resolution-solvent", "fake-method");
    options.SetSolvent(Solvent("water"));
    const detail::ResolvedEngine resolved = detail::resolve_engine(options);
    EXPECT_EQ(resolved.settings.solvent.GetModel(), SolventModel::ALPB);
    EXPECT_NE(resolved.solvent_warning.find("resolved to 'alpb'"), std::string::npos);
}

TEST(EngineResolutionTest, AccuracyKeywordOverridesThePresetAfterValidation) {
    auto fake = test::register_fake("resolution-accuracy");
    KeywordSpec accuracy;
    accuracy.name = "accuracy";
    accuracy.type = KeywordType::DOUBLE;
    accuracy.has_min = true;
    accuracy.min = 0.01;
    accuracy.has_max = true;
    accuracy.max = 1000.0;
    fake->keyword_specs = {accuracy};
    CalculationOptions options = options_for("resolution-accuracy", "fake-method");
    options.SetKeyword("accuracy", "0.25");
    EXPECT_DOUBLE_EQ(detail::resolve_engine(options).settings.accuracy, 0.25);
}

TEST(EngineResolutionTest, UnknownEngineNameThrows) {
    EXPECT_THROW(detail::resolve_engine(options_for("no-such-engine", "fake-method")),
                 EngineUnavailableError);
}

TEST(EngineResolutionTest, ChargePolicyFollowsTheCapabilitiesAndTheOption) {
    MethodCapabilities capabilities;
    CalculationOptions options;
    EXPECT_EQ(detail::charge_policy(options, capabilities), PartialChargePolicy::NONE);
    capabilities.requires_partial_charges = true;
    EXPECT_EQ(detail::charge_policy(options, capabilities),
              PartialChargePolicy::ASSIGN_AM1BCC_ELF10);
    options.SetUseInputPartialCharges(true);
    EXPECT_EQ(detail::charge_policy(options, capabilities), PartialChargePolicy::USE_INPUT);
}
