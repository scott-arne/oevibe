#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

#include "oevibe/Error.h"

TEST(ErrorTest, HierarchyRootsAtRuntimeError) {
    EXPECT_TRUE((std::is_base_of<std::runtime_error, OEVIBE::OEVibeError>::value));
    EXPECT_TRUE((std::is_base_of<OEVIBE::OEVibeError, OEVIBE::OptionsError>::value));
    EXPECT_TRUE((std::is_base_of<OEVIBE::OEVibeError, OEVIBE::EngineUnavailableError>::value));
    EXPECT_TRUE((std::is_base_of<OEVIBE::OEVibeError, OEVIBE::InputError>::value));
    EXPECT_TRUE((std::is_base_of<OEVIBE::OEVibeError, OEVIBE::UnsupportedError>::value));
}

TEST(ErrorTest, MessageIsPreserved) {
    try {
        throw OEVIBE::InputError("molecule needs 3D coordinates");
    } catch (const OEVIBE::OEVibeError& e) {
        EXPECT_STREQ(e.what(), "molecule needs 3D coordinates");
    }
}
