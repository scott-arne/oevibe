#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "oevibe/Engine.h"
#include "oevibe/Error.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

KeywordSpec spec(const std::string& name, KeywordType type) {
    KeywordSpec keyword;
    keyword.name = name;
    keyword.type = type;
    return keyword;
}

bool mentions(const std::string& text, const std::string& word) {
    return text.find(word) != std::string::npos;
}

}  // namespace

TEST(EngineTest, FindMethod) {
    const test::FakeEngine engine("fake-find");
    EXPECT_TRUE(find_method(engine, Method("fake-method")).has_value());
    EXPECT_FALSE(find_method(engine, Method("other-method")).has_value());
}

TEST(EngineTest, DefaultValidateSettingsChecksKeywords) {
    test::FakeEngine engine("fake-keywords", {test::fake_capabilities("fake-method"),
                                              test::fake_capabilities("fake-other")});
    KeywordSpec iterations = spec("iterations", KeywordType::INT);
    iterations.has_min = true;
    iterations.min = 1.0;
    iterations.has_max = true;
    iterations.max = 100.0;
    KeywordSpec smearing = spec("smearing", KeywordType::DOUBLE);
    smearing.methods = {Method("fake-method")};
    engine.keyword_specs = {iterations, smearing, spec("verbose", KeywordType::BOOL)};

    EngineSettings settings;
    settings.method = Method("fake-method");
    settings.keywords = {{"iterations", "50"}, {"smearing", "0.5"}, {"verbose", "true"}};
    EXPECT_NO_THROW(engine.ValidateSettings(settings));
    settings.keywords = {{"iterations", "0"}};
    EXPECT_THROW(engine.ValidateSettings(settings), OptionsError);
    settings.keywords = {{"iterations", "2.5"}};
    EXPECT_THROW(engine.ValidateSettings(settings), OptionsError);
    settings.keywords = {{"unknown", "1"}};
    EXPECT_THROW(engine.ValidateSettings(settings), OptionsError);
    settings.keywords = {{"verbose", "maybe"}};
    EXPECT_THROW(engine.ValidateSettings(settings), OptionsError);
    settings.method = Method("fake-other");
    settings.keywords = {{"smearing", "0.1"}};
    EXPECT_THROW(engine.ValidateSettings(settings), OptionsError);
}

TEST(EngineTest, DefaultValidateSettingsRejectsAnUnknownMethod) {
    const test::FakeEngine engine("fake-unknown-method");
    EngineSettings settings;
    settings.method = Method("not-provided");
    EXPECT_THROW(engine.ValidateSettings(settings), OptionsError);
}

TEST(EngineTest, DefaultValidateSystemChecksElementsSpinAndCharge) {
    const OEChem::OEMol mol = test::make_gapped_methane();
    EngineSettings settings;
    settings.method = Method("fake-method");

    MethodCapabilities caps = test::fake_capabilities("fake-method");
    caps.elements = {1, 6};
    EXPECT_NO_THROW(
        test::FakeEngine("e1", {caps}).ValidateSystem(MolecularSystem(mol, 0, 1), settings));
    caps.elements = {1};
    EXPECT_THROW(
        test::FakeEngine("e2", {caps}).ValidateSystem(MolecularSystem(mol, 0, 1), settings),
        InputError);
    caps.elements.clear();
    caps.open_shell = false;
    EXPECT_THROW(
        test::FakeEngine("e3", {caps}).ValidateSystem(MolecularSystem(mol, 1, 2), settings),
        InputError);
    caps.open_shell = true;
    caps.charge_from_topology = true;
    EXPECT_THROW(
        test::FakeEngine("e4", {caps}).ValidateSystem(MolecularSystem(mol, 1, 2), settings),
        InputError);
}

TEST(EngineTest, ValidateEvaluationChecksShapesAndPresence) {
    EvaluationRequest request;
    request.gradient = true;
    request.properties = static_cast<unsigned>(Property::CHARGES);
    Evaluation evaluation;
    evaluation.energy = 1.0;
    evaluation.gradient.assign(6, 0.0);
    evaluation.charges = {0.0, 0.0};
    EXPECT_EQ(validate_evaluation(evaluation, request, 2), "");

    Evaluation short_gradient = evaluation;
    short_gradient.gradient.pop_back();
    EXPECT_TRUE(mentions(validate_evaluation(short_gradient, request, 2), "gradient"));

    Evaluation bad_energy = evaluation;
    bad_energy.energy = std::nan("");
    EXPECT_TRUE(mentions(validate_evaluation(bad_energy, request, 2), "energy"));

    Evaluation extra_dipole = evaluation;
    extra_dipole.dipole = {0.0, 0.0, 1.0};
    EXPECT_TRUE(mentions(validate_evaluation(extra_dipole, request, 2), "dipole"));

    EvaluationRequest bonds;
    bonds.properties = static_cast<unsigned>(Property::BOND_ORDERS);
    Evaluation bad_bond;
    bad_bond.bond_orders = {{1, 2, 1.0}};
    EXPECT_TRUE(mentions(validate_evaluation(bad_bond, bonds, 2), "bond"));

    EvaluationRequest orbitals;
    orbitals.properties = static_cast<unsigned>(Property::ORBITAL_ENERGIES);
    Evaluation unsorted;
    unsorted.orbital_energies = {1.0, -1.0};
    unsorted.homo_index = 0;
    EXPECT_TRUE(mentions(validate_evaluation(unsorted, orbitals, 2), "orbital"));
}
