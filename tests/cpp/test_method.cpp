#include <gtest/gtest.h>

#include "oevibe/Error.h"
#include "oevibe/Method.h"

using namespace OEVIBE;

TEST(MethodTest, DefaultIsGfn2Xtb) { EXPECT_EQ(Method().GetName(), "gfn2-xtb"); }

TEST(MethodTest, NamesAreLowercased) {
    EXPECT_EQ(Method("MMFF94S").GetName(), "mmff94s");
    EXPECT_EQ(Method("wB97X-D3BJ/def2-TZVP").GetName(), "wb97x-d3bj/def2-tzvp");
    EXPECT_EQ(Method("ff14sb+sage.2").GetName(), "ff14sb+sage.2");
}

TEST(MethodTest, RejectsInvalidNames) {
    EXPECT_THROW(Method(""), OptionsError);
    EXPECT_THROW(Method("gfn2 xtb"), OptionsError);
    EXPECT_THROW(Method("gfn2_xtb"), OptionsError);
}

TEST(MethodTest, Equality) {
    EXPECT_EQ(Method("Sage"), Method(SAGE));
    EXPECT_NE(Method(SAGE), Method(PARSLEY));
    EXPECT_TRUE(Method(MMFF94) < Method(MMFF94S));
}

TEST(MethodTest, BuiltinRoutingFollowsSpec) {
    EXPECT_EQ(builtin_engine_for(Method(GFN2_XTB)), "tblite");
    EXPECT_EQ(builtin_engine_for(Method(GFN1_XTB)), "tblite");
    EXPECT_EQ(builtin_engine_for(Method(GFN0_XTB)), "xtb");
    EXPECT_EQ(builtin_engine_for(Method(GFN_FF)), "xtb");
    EXPECT_EQ(builtin_engine_for(Method(MMFF94)), "oeff");
    EXPECT_EQ(builtin_engine_for(Method(MMFF94S)), "oeff");
    EXPECT_EQ(builtin_engine_for(Method(SAGE)), "oeff");
    EXPECT_EQ(builtin_engine_for(Method(PARSLEY)), "oeff");
    EXPECT_EQ(builtin_engine_for(Method("my-custom-ff")), "");
}
