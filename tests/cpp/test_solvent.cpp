#include <gtest/gtest.h>

#include "oevibe/Error.h"
#include "oevibe/Property.h"
#include "oevibe/Solvent.h"

using namespace OEVIBE;

TEST(SolventTest, DefaultIsGasPhase) {
    const Solvent solvent;
    EXPECT_TRUE(solvent.IsGasPhase());
    EXPECT_EQ(solvent.GetName(), "");
    EXPECT_EQ(solvent.GetTag(), "gas");
}

TEST(SolventTest, NamesAreLowercased) {
    const Solvent solvent("Water");
    EXPECT_FALSE(solvent.IsGasPhase());
    EXPECT_EQ(solvent.GetName(), "water");
    EXPECT_EQ(solvent.GetModel(), SolventModel::AUTO);
}

TEST(SolventTest, TagNeedsConcreteModel) {
    EXPECT_THROW(Solvent("water").GetTag(), OptionsError);
    EXPECT_EQ(Solvent("water").WithModel(SolventModel::ALPB).GetTag(), "alpb-water");
    EXPECT_EQ(Solvent("water", SolventModel::SHEFFIELD).GetTag(), "sheffield-water");
    EXPECT_TRUE(Solvent().WithModel(SolventModel::ALPB).IsGasPhase());
}

TEST(SolventTest, RejectsBadNamesAndModels) {
    EXPECT_THROW(Solvent(""), OptionsError);
    EXPECT_THROW(Solvent("di methyl"), OptionsError);
    EXPECT_THROW(Solvent("water", static_cast<SolventModel>(9)), OptionsError);
}

TEST(SolventTest, ModelNames) {
    EXPECT_EQ(solvent_model_name(SolventModel::AUTO), "auto");
    EXPECT_EQ(solvent_model_name(SolventModel::ALPB), "alpb");
    EXPECT_EQ(solvent_model_name(SolventModel::GBSA), "gbsa");
    EXPECT_EQ(solvent_model_name(SolventModel::SHEFFIELD), "sheffield");
}

TEST(PropertyTest, NamesListSetBits) {
    EXPECT_EQ(property_names(0), "");
    EXPECT_EQ(property_names(static_cast<unsigned>(Property::CHARGES) |
                             static_cast<unsigned>(Property::BOND_ORDERS)),
              "charges, bond_orders");
    EXPECT_EQ(ALL_PROPERTIES, 15u);
}
