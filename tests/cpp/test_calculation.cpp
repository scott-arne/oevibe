#include <gtest/gtest.h>

#include <oechem.h>
#include <oemmff.h>

#include <string>
#include <vector>

#include "oevibe/Calculation.h"
#include "oevibe/Error.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions fake_options(const std::string& engine) {
    CalculationOptions options;
    options.SetMethod(Method("fake-method"));
    options.SetEngine(engine);
    return options;
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

OEChem::OEConfBase* conformer(OEChem::OEMol& mol, unsigned position) {
    OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs();
    for (unsigned k = 0; k < position; ++k) {
        ++conf;
    }
    return conf;
}

}  // namespace

TEST(CalculationTest, ConstructionResolvesAndValidates) {
    test::register_fake("calc-ok");
    EXPECT_NO_THROW(Calculation calc(fake_options("calc-ok")));
    EXPECT_THROW(Calculation calc(fake_options("calc-missing")), EngineUnavailableError);

    auto down = test::register_fake("calc-down");
    down->available = false;
    down->unavailable_reason = "library missing";
    try {
        Calculation calc(fake_options("calc-down"));
        FAIL() << "expected EngineUnavailableError";
    } catch (const EngineUnavailableError& e) {
        EXPECT_TRUE(contains(e.what(), "library missing"));
    }

    auto partial = test::register_fake("calc-partial");
    partial->methods[0].available = false;
    partial->methods[0].unavailable_reason = "file state";
    EXPECT_THROW(Calculation calc(fake_options("calc-partial")), EngineUnavailableError);

    CalculationOptions wrong_method = fake_options("calc-ok");
    wrong_method.SetMethod(Method("not-there"));
    EXPECT_THROW(Calculation calc(wrong_method), OptionsError);
}

TEST(CalculationTest, SolventResolution) {
    auto engine = test::register_fake("calc-solvent");
    CalculationOptions options = fake_options("calc-solvent");
    options.SetSolvent(Solvent("water"));
    OEChem::OEMol mol = test::make_conformers("CCO");
    Calculation(options).Energy(mol);
    EXPECT_EQ(engine->last_settings.solvent.GetModel(), SolventModel::ALPB);
    EXPECT_TRUE(OEChem::OEHasSDData(*conformer(mol, 0), "OEVibe.fake-method.alpb-water.Energy"));

    options.SetSolvent(Solvent("water", SolventModel::GBSA));
    EXPECT_THROW(Calculation calc(options), OptionsError);
    engine->methods[0].solvent_models.clear();
    options.SetSolvent(Solvent("water"));
    EXPECT_THROW(Calculation calc(options), OptionsError);
}

TEST(CalculationTest, UnsupportedPropertiesAndKeywordsAreOptionsErrors) {
    auto engine = test::register_fake("calc-props");
    engine->methods[0].properties = 0;
    CalculationOptions options = fake_options("calc-props");
    options.SetProperties(static_cast<unsigned>(Property::DIPOLE));
    EXPECT_THROW(Calculation calc(options), OptionsError);
    options.SetProperties(0);
    options.SetKeyword("nope", "1");
    EXPECT_THROW(Calculation calc(options), OptionsError);
}

TEST(CalculationTest, EnergyOverEveryConformer) {
    test::register_fake("calc-energy");
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    ASSERT_GE(mol.NumConfs(), 2u);
    const ResultSet results = Calculation(fake_options("calc-energy")).Energy(mol);
    ASSERT_EQ(results.Size(), mol.NumConfs());
    const ConformerResult& first = results.Get(0);
    EXPECT_EQ(first.GetStatus(), ConformerStatus::OK);
    EXPECT_NEAR(first.GetEnergy(), 0.0, 1e-9);  // conformer 0 is the fake engine's reference
    EXPECT_TRUE(first.HasRelativeEnergy());
    EXPECT_FALSE(first.HasGradient());
    EXPECT_FALSE(first.HasCoordinates());
    for (std::size_t i = 1; i < results.Size(); ++i) {
        EXPECT_GT(results.Get(i).GetEnergy(), 0.0);
    }
    EXPECT_EQ(first.GetProvenance().engine, "calc-energy");
    EXPECT_EQ(first.GetProvenance().method, "fake-method");
    EXPECT_EQ(first.GetProvenance().solvent_model, "gas");
    EXPECT_EQ(first.GetProvenance().options_digest.size(), 64u);
    EXPECT_EQ(OEChem::OEGetSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.Status"), "OK");
    EXPECT_TRUE(OEChem::OEHasSDData(*conformer(mol, 1), "OEVibe.fake-method.gas.RelativeEnergy"));
}

TEST(CalculationTest, GradientUsesTheOpenEyeLayout) {
    auto engine = test::register_fake("calc-gradient");
    OEChem::OEMol mol = test::make_gapped_methane();
    // The fake's gradient is zero at its own reference, so displace the reference to make the
    // gradient at the real atoms non-trivial.
    engine->reference = MolecularSystem(mol, 0, 1).GetConformerCoordinates(0);
    engine->reference[0] += 0.1;
    const ResultSet results = Calculation(fake_options("calc-gradient")).Gradient(mol);
    ASSERT_TRUE(results.Get(0).HasGradient());
    const std::vector<double> gradient = results.Get(0).GetGradient();
    ASSERT_EQ(gradient.size(), 18u);
    EXPECT_DOUBLE_EQ(gradient[3], 0.0);
    // OpenEye index 0 is the carbon: its components must carry the engine's gradient.
    EXPECT_TRUE(gradient[0] != 0.0 || gradient[1] != 0.0 || gradient[2] != 0.0);
    // OpenEye index 1 was deleted, so its slots stay exactly zero.
    for (std::size_t k = 3; k < 6; ++k) {
        EXPECT_EQ(gradient[k], 0.0);
    }
}

TEST(CalculationTest, InputErrorsComeBeforeComputation) {
    auto engine = test::register_fake("calc-input");
    OEChem::OEGraphMol flat;
    OEChem::OESmilesToMol(flat, "CCO");
    EXPECT_THROW(Calculation(fake_options("calc-input")).Energy(flat), InputError);
    EXPECT_EQ(engine->sessions_created, 0);
}

TEST(CalculationTest, EngineFailuresBecomeStatuses) {
    auto engine = test::register_fake("calc-fail");
    OEChem::OEMol mol = test::make_conformers("CCO");
    engine->mode = test::FakeMode::THROW_UNSUPPORTED;
    ResultSet results = Calculation(fake_options("calc-fail")).Energy(mol);
    EXPECT_EQ(results.Get(0).GetStatus(), ConformerStatus::UNSUPPORTED);
    EXPECT_FALSE(results.Get(0).HasEnergy());
    EXPECT_EQ(OEChem::OEGetSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.Status"),
              "UNSUPPORTED");
    engine->mode = test::FakeMode::THROW_RUNTIME;
    results = Calculation(fake_options("calc-fail")).Energy(mol);
    EXPECT_EQ(results.Get(0).GetStatus(), ConformerStatus::ENGINE_ERROR);
    EXPECT_TRUE(contains(results.Get(0).GetMessage(), "fake engine failure"));
    engine->mode = test::FakeMode::MALFORMED_GRADIENT;
    results = Calculation(fake_options("calc-fail")).Gradient(mol);
    EXPECT_EQ(results.Get(0).GetStatus(), ConformerStatus::ENGINE_ERROR);
    EXPECT_TRUE(contains(results.Get(0).GetMessage(), "gradient"));
    EXPECT_FALSE(results.HasLowest());
    // A finite energy computed before the malformed part is kept; the malformed gradient is not.
    EXPECT_TRUE(results.Get(0).HasEnergy());
    EXPECT_FALSE(results.Get(0).HasGradient());
}

// A null session breaks the engine contract; dereferencing it would crash the whole call
// instead of failing one conformer (spec 7.5).
TEST(CalculationTest, NullSessionsBecomeEngineErrors) {
    auto engine = test::register_fake("calc-null-session");
    engine->null_session = true;
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    ASSERT_GE(mol.NumConfs(), 2u);
    for (const bool gradient : {false, true}) {
        SCOPED_TRACE(gradient ? "gradient" : "energy");
        engine->sessions_created = 0;
        Calculation calc(fake_options("calc-null-session"));
        const ResultSet results = gradient ? calc.Gradient(mol) : calc.Energy(mol);
        ASSERT_EQ(results.Size(), mol.NumConfs());
        // Every conformer still gets its own attempt.
        EXPECT_EQ(engine->sessions_created, static_cast<int>(mol.NumConfs()));
        for (std::size_t i = 0; i < results.Size(); ++i) {
            const ConformerResult& result = results.Get(i);
            EXPECT_EQ(result.GetStatus(), ConformerStatus::ENGINE_ERROR);
            EXPECT_TRUE(contains(result.GetMessage(), "null session")) << result.GetMessage();
            EXPECT_FALSE(result.HasEnergy());
        }
    }
}

TEST(CalculationTest, AutoSolventResolutionIsRecordedAsAWarning) {
    OEChem::OEMol mol = test::make_conformers("CCCCO", 2);
    CalculationOptions options;
    options.SetMethod(Method(MMFF94));
    options.SetSolvent(Solvent("water"));
    const auto names_the_resolution = [](const ConformerResult& result) {
        for (const std::string& warning : result.GetWarnings()) {
            if (contains(warning, "water") && contains(warning, "sheffield")) {
                return true;
            }
        }
        return false;
    };
    Calculation calc(options);
    for (const bool gradient : {false, true}) {
        SCOPED_TRACE(gradient ? "gradient" : "energy");
        const ResultSet results = gradient ? calc.Gradient(mol) : calc.Energy(mol);
        ASSERT_EQ(results.Size(), mol.NumConfs());
        for (std::size_t i = 0; i < results.Size(); ++i) {
            EXPECT_TRUE(names_the_resolution(results.Get(i)));
        }
    }

    // Nothing was resolved for an explicit model or for gas phase.
    options.SetSolvent(Solvent("water", SolventModel::SHEFFIELD));
    EXPECT_TRUE(Calculation(options).Energy(mol).Get(0).GetWarnings().empty());
    options.SetSolvent(Solvent());
    EXPECT_TRUE(Calculation(options).Energy(mol).Get(0).GetWarnings().empty());
}

TEST(CalculationTest, PartialChargeWriteBackIgnoresConformerOrder) {
    auto engine = test::register_fake("calc-charges");
    CalculationOptions options = fake_options("calc-charges");
    options.SetAssignPartialCharges(true);
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    ASSERT_GE(mol.NumConfs(), 3u);
    const MolecularSystem probe(mol, 0, 1);
    engine->reference = probe.GetConformerCoordinates(2);  // the third conformer is lowest

    OEChem::OEMol reversed(mol);
    reversed.DeleteConfs();
    for (unsigned position = mol.NumConfs(); position-- > 0;) {
        std::vector<double> coords(3 * mol.GetMaxAtomIdx());
        conformer(mol, position)->GetCoords(coords.data());
        reversed.NewConf(coords.data());
    }

    Calculation(options).Energy(mol);
    Calculation(options).Energy(reversed);
    const std::vector<double>& lowest = probe.GetConformerCoordinates(2);
    OESystem::OEIter<OEChem::OEAtomBase> a = mol.GetAtoms();
    OESystem::OEIter<OEChem::OEAtomBase> b = reversed.GetAtoms();
    for (unsigned dense = 0; a && b; ++a, ++b, ++dense) {
        EXPECT_NEAR(a->GetPartialCharge(), 0.001 * lowest[3 * dense], 1e-6);
        EXPECT_DOUBLE_EQ(a->GetPartialCharge(), b->GetPartialCharge());
    }
}

TEST(CalculationTest, Mmff94EnergiesMatchOpenEye) {
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    CalculationOptions options;
    options.SetMethod(Method(MMFF94));
    const ResultSet results = Calculation(options).Energy(mol);
    OEMolMMFF::OEMMFF direct{OEMolMMFF::OEMMFF94Params()};
    OEChem::OEMol copy(mol);
    ASSERT_TRUE(direct.PrepMol(copy));
    ASSERT_TRUE(direct.Setup(copy));
    unsigned position = 0;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = copy.GetConfs(); conf; ++conf, ++position) {
        std::vector<double> coords(3 * copy.GetMaxAtomIdx());
        conf->GetCoords(coords.data());
        EXPECT_NEAR(results.Get(position).GetEnergy(), direct(coords.data()), 1e-8);
    }
}

TEST(CalculationTest, ForceFieldsRejectAChargeOverride) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    CalculationOptions options;
    options.SetMethod(Method(MMFF94));
    options.SetCharge(1);
    EXPECT_THROW(Calculation(options).Energy(mol), InputError);
}

TEST(CalculationTest, TagsCanBeDisabled) {
    test::register_fake("calc-notags");
    CalculationOptions options = fake_options("calc-notags");
    options.SetWriteTags(false);
    OEChem::OEMol mol = test::make_conformers("CCO");
    Calculation(options).Energy(mol);
    EXPECT_FALSE(OEChem::OEHasSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.Status"));
}

TEST(CalculationTest, AccuracyKeywordOverridesThePreset) {
    auto engine = test::register_fake("calc-accuracy");
    KeywordSpec spec;
    spec.name = "accuracy";
    spec.type = KeywordType::DOUBLE;
    engine->keyword_specs = {spec};
    CalculationOptions options = fake_options("calc-accuracy");
    options.SetKeyword("accuracy", "0.5");
    OEChem::OEMol mol = test::make_conformers("CCO");
    const ResultSet results = Calculation(options).Energy(mol);
    EXPECT_DOUBLE_EQ(engine->last_settings.accuracy, 0.5);
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("accuracy"), "0.500000");
}

TEST(CalculationTest, FailedRerunClearsNumericTags) {
    auto engine = test::register_fake("calc-rerun");
    const CalculationOptions options = fake_options("calc-rerun");
    OEChem::OEMol mol = test::make_conformers("CCO");
    Calculation(options).Energy(mol);
    ASSERT_TRUE(OEChem::OEHasSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.Energy"));
    ASSERT_TRUE(OEChem::OEHasSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.RelativeEnergy"));

    engine->mode = test::FakeMode::THROW_RUNTIME;
    Calculation(options).Energy(mol);
    EXPECT_EQ(OEChem::OEGetSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.Status"),
              "ENGINE_ERROR");
    EXPECT_FALSE(OEChem::OEHasSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.Energy"));
    EXPECT_FALSE(OEChem::OEHasSDData(*conformer(mol, 0), "OEVibe.fake-method.gas.RelativeEnergy"));
}

TEST(CalculationTest, DigestCoversChargeMultiplicityAndTask) {
    test::register_fake("calc-digest");
    OEChem::OEMol mol = test::make_conformers("CCO");
    const auto provenance_of = [&mol](const CalculationOptions& options, bool gradient) {
        Calculation calc(options);
        const ResultSet results = gradient ? calc.Gradient(mol) : calc.Energy(mol);
        return results.Get(0).GetProvenance();
    };

    const CalculationOptions base = fake_options("calc-digest");
    CalculationOptions charged = base;
    charged.SetCharge(1);
    CalculationOptions triplet = base;
    triplet.SetMultiplicity(3);

    const Provenance energy = provenance_of(base, false);
    const Provenance gradient = provenance_of(base, true);
    const Provenance cation = provenance_of(charged, false);
    const Provenance high_spin = provenance_of(triplet, false);

    EXPECT_NE(energy.options_digest, gradient.options_digest);
    EXPECT_NE(energy.options_digest, cation.options_digest);
    EXPECT_NE(energy.options_digest, high_spin.options_digest);

    EXPECT_EQ(energy.resolved_options.at("charge"), "0");
    EXPECT_EQ(energy.resolved_options.at("multiplicity"), "1");
    EXPECT_EQ(energy.resolved_options.at("partial_charges"), "none");
    EXPECT_EQ(energy.resolved_options.at("task"), "energy");
    EXPECT_EQ(gradient.resolved_options.at("task"), "gradient");
    EXPECT_EQ(cation.resolved_options.at("charge"), "1");
    EXPECT_EQ(cation.resolved_options.at("multiplicity"), "2");
    EXPECT_EQ(high_spin.resolved_options.at("multiplicity"), "3");
}
