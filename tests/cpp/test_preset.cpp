#include <gtest/gtest.h>

#include "oevibe/Error.h"
#include "oevibe/Preset.h"

using namespace OEVIBE;

// Spec 5.4 preset table.
TEST(PresetTest, TableMatchesSpec) {
    const PresetValues fast = preset_values(Preset::FAST);
    EXPECT_NEAR(fast.energy_change, 0.03138, 1e-5);
    EXPECT_NEAR(fast.gradient_norm, 4.743, 1e-3);
    EXPECT_DOUBLE_EQ(fast.engine_accuracy, 2.0);
    const PresetValues standard = preset_values(Preset::STANDARD);
    EXPECT_NEAR(standard.energy_change, 0.003138, 1e-6);
    EXPECT_NEAR(standard.gradient_norm, 1.186, 1e-3);
    EXPECT_DOUBLE_EQ(standard.engine_accuracy, 1.0);
    const PresetValues thorough = preset_values(Preset::THOROUGH);
    EXPECT_NEAR(thorough.energy_change, 0.0006275, 1e-7);
    EXPECT_NEAR(thorough.gradient_norm, 0.9487, 1e-4);
    EXPECT_DOUBLE_EQ(thorough.engine_accuracy, 0.2);
    for (Preset preset : {Preset::FAST, Preset::STANDARD, Preset::THOROUGH}) {
        const PresetValues values = preset_values(preset);
        EXPECT_DOUBLE_EQ(values.initial_trust_radius, 0.1);
        EXPECT_DOUBLE_EQ(values.max_trust_radius, 0.3);
        EXPECT_DOUBLE_EQ(values.fd_step, 0.005);
    }
}

TEST(PresetTest, MaxIterationsScaleWithAtoms) {
    EXPECT_EQ(preset_max_iterations(10), 200u);
    EXPECT_EQ(preset_max_iterations(100), 300u);
    EXPECT_EQ(preset_max_iterations(5000), 10000u);
}

TEST(PresetTest, NamesRoundTrip) {
    EXPECT_EQ(preset_name(Preset::FAST), "fast");
    EXPECT_EQ(preset_name(Preset::STANDARD), "standard");
    EXPECT_EQ(preset_name(Preset::THOROUGH), "thorough");
    EXPECT_EQ(preset_from_name("Standard"), Preset::STANDARD);
    EXPECT_THROW(preset_from_name("slow"), OptionsError);
}

TEST(PresetTest, RejectsOutOfRangeEnum) {
    EXPECT_THROW(preset_values(static_cast<Preset>(7)), OptionsError);
}
