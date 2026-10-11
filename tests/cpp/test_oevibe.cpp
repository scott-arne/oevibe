#include <gtest/gtest.h>

#include "oevibe/oevibe.h"

TEST(VersionTest, MacrosDefined) {
    EXPECT_GE(OEVIBE_VERSION_MAJOR, 0);
    EXPECT_GE(OEVIBE_VERSION_MINOR, 0);
    EXPECT_GE(OEVIBE_VERSION_PATCH, 0);
}
