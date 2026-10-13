#include <gtest/gtest.h>

#include <oechem.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "engines/XtbLibrary.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"
#include "oevibe/engines/XtbEngine.h"
#include "test_runtime_libraries.h"

using namespace OEVIBE;

namespace {

struct Geometry {
    std::vector<unsigned> numbers;
    std::vector<double> coords;  ///< Angstrom
};

// The ethanol of spike S4. Every GFN0 reference below came from the xtb 6.7.1 C API at these
// exact coordinates, accuracy 1.0, with XTBPATH and XTBHOME unset (probe 2026-10-11).
const Geometry ETHANOL = {{6, 6, 8, 1, 1, 1, 1, 1, 1},
                          {-0.748, -0.015, 0.024,  0.558,  0.420,  -0.619, 0.599,  1.811,  -0.873,
                           -1.564, 0.226,  -0.657, -0.865, 0.495,  0.985,  -0.764, -1.094, 0.210,
                           1.398,  0.169,  0.035,  0.687,  -0.104, -1.572, 1.463,  2.034,  -1.247}};
// The water of spike S5 (tblite GFN2 gives -5.070370764 Eh there).
const Geometry WATER = {{8, 1, 1}, {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572, -0.4692}};
// O-H at 0.97 Angstrom: the hydroxyl radical (doublet) and hydroxide (charge -1).
const Geometry HYDROXYL = {{8, 1}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.97}};

constexpr double ETHANOL_ENERGY_EH = -10.829696338254681;
constexpr double WATER_ENERGY_EH = -4.366773861537583;
constexpr double HYDROXYL_RADICAL_ENERGY_EH = -3.731464148978133;
constexpr double HYDROXIDE_ENERGY_EH = -5.2001680570127435;
constexpr double TRIPLET_WATER_ENERGY_EH = -4.1413397882478415;

/// A system whose conformer holds ``geometry``. Engines read only atomic numbers, charge, and
/// multiplicity from it here: the tests pass exact coordinates to Evaluate, because OpenEye
/// keeps conformer coordinates in single precision.
MolecularSystem make_system(const Geometry& geometry, int charge, unsigned multiplicity) {
    OEChem::OEGraphMol mol;
    for (unsigned z : geometry.numbers) {
        mol.NewAtom(z);
    }
    mol.SetCoords(geometry.coords.data());
    mol.SetDimension(3);
    return MolecularSystem(mol, charge, multiplicity);
}

EngineSettings gfn0_settings(double accuracy = 1.0) {
    EngineSettings settings;
    settings.method = Method(GFN0_XTB);
    settings.accuracy = accuracy;
    return settings;
}

Evaluation evaluate(XtbEngine& engine, const MolecularSystem& system, const Geometry& geometry,
                    const EvaluationRequest& request, double accuracy = 1.0) {
    return engine.CreateSession(system, gfn0_settings(accuracy))
        ->Evaluate(geometry.coords, request);
}

/// A system whose conformers are ``first`` and then ``second``, in that order.
MolecularSystem make_two_conformers(const Geometry& first, const Geometry& second) {
    OEChem::OEGraphMol graph;
    for (unsigned z : first.numbers) {
        graph.NewAtom(z);
    }
    graph.SetCoords(first.coords.data());
    graph.SetDimension(3);
    OEChem::OEMol mol(graph);
    mol.NewConf(second.coords.data());
    return MolecularSystem(mol, 0, 1);
}

/// The energy of one conformer, evaluated as Calculation does: a new session for that conformer
/// alone, fed that conformer's coordinates.
double evaluate_conformer(XtbEngine& engine, const MolecularSystem& system, unsigned position) {
    return engine.CreateSession(system, gfn0_settings())
        ->Evaluate(system.GetConformerCoordinates(position), EvaluationRequest())
        .energy;
}

/// Fake xtb installations: ``prefix`` holds lib/ and share/xtb/param_gfn0-xtb.txt, ``home``
/// and ``path_b`` hold the parameter file, ``path_a`` and ``bare_prefix`` hold none.
class XtbParameterFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = std::filesystem::temp_directory_path() /
                ("oevibe-xtb-" + std::to_string(getpid()) + "-" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(root_);
        for (const char* directory :
             {"prefix/lib", "prefix/share/xtb", "home", "path_a", "path_b", "bare_prefix/lib"}) {
            std::filesystem::create_directories(root_ / directory);
        }
        for (const char* directory : {"prefix/share/xtb", "home", "path_b"}) {
            std::ofstream(root_ / directory / detail::XTB_GFN0_PARAMETER_FILE_NAME) << "fake\n";
        }
    }
    void TearDown() override { std::filesystem::remove_all(root_); }

    std::string Path(const std::string& relative) const { return (root_ / relative).string(); }
    std::string File(const std::string& directory) const {
        return (root_ / directory / detail::XTB_GFN0_PARAMETER_FILE_NAME).string();
    }

    std::filesystem::path root_;
};

/// Engine tests that need libxtb skip when OEVIBE_XTB_LIBRARY does not name a file.
class XtbEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        library_ = test::xtb_library_path();
        if (library_.empty()) {
            GTEST_SKIP() << test::XTB_SKIP_REASON;
        }
    }

    std::string library_;
};

}  // namespace

TEST(XtbLibraryTest, OnlyCApi10IsSupported) {
    EXPECT_TRUE(detail::xtb_api_version_supported(10000));
    for (int version : {-1, 0, 9999, 10001, 10100, 20000}) {
        EXPECT_FALSE(detail::xtb_api_version_supported(version)) << version;
    }
}

TEST(XtbLibraryTest, ApiVersionText) {
    EXPECT_EQ(detail::xtb_api_version_text(10000), "1.0.0");
    EXPECT_EQ(detail::xtb_api_version_text(20103), "2.1.3");
}

TEST(XtbLibraryTest, GfnFfReasonNamesTheWorkingDirectoryFile) {
    const std::string reason = detail::XTB_GFNFF_UNAVAILABLE_REASON;
    EXPECT_NE(reason.find("gfnff_topo"), std::string::npos);
    EXPECT_NE(reason.find("working directory"), std::string::npos);
    EXPECT_NE(reason.find("2026-10-11"), std::string::npos);
}

TEST(XtbLibraryTest, KeywordsNeedNoLibrary) {
    const XtbEngine engine("/nonexistent/libxtb.6.dylib");
    EXPECT_EQ(engine.GetName(), "xtb");
    const std::vector<KeywordSpec> specs = engine.GetKeywordSpecs();
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].name, "accuracy");
    EXPECT_EQ(specs[0].type, KeywordType::DOUBLE);
    EXPECT_TRUE(specs[0].methods.empty());
    EXPECT_TRUE(specs[0].has_min);
    EXPECT_DOUBLE_EQ(specs[0].min, 1e-4);
    EXPECT_TRUE(specs[0].has_max);
    EXPECT_DOUBLE_EQ(specs[0].max, 1e3);
}

TEST_F(XtbParameterFileTest, XtbpathDirectoriesAreSearchedInOrder) {
    const std::string library = Path("bare_prefix/lib/libxtb.6.dylib");
    const std::string xtbpath = Path("path_a") + ":" + Path("path_b");
    EXPECT_EQ(detail::xtb_gfn0_parameter_file(library, xtbpath, ""), File("path_b"));
    // XTBPATH wins over XTBHOME and the prefix.
    EXPECT_EQ(
        detail::xtb_gfn0_parameter_file(Path("prefix/lib/libxtb.6.dylib"), xtbpath, Path("home")),
        File("path_b"));
}

TEST_F(XtbParameterFileTest, XtbhomeIsSearchedAfterXtbpath) {
    const std::string library = Path("prefix/lib/libxtb.6.dylib");
    EXPECT_EQ(detail::xtb_gfn0_parameter_file(library, Path("path_a"), Path("home")), File("home"));
}

TEST_F(XtbParameterFileTest, PrefixShareXtbIsTheFallback) {
    EXPECT_EQ(detail::xtb_gfn0_parameter_file(Path("prefix/lib/libxtb.6.dylib"), "", ""),
              File("prefix/share/xtb"));
    EXPECT_EQ(detail::xtb_gfn0_parameter_candidates(Path("prefix/lib/libxtb.6.dylib"),
                                                    Path("path_a") + "::", Path("home")),
              (std::vector<std::string>{File("path_a"), File("home"), File("prefix/share/xtb")}));
}

TEST_F(XtbParameterFileTest, NoFileGivesAnEmptyPathAndAReasonNamingEveryCandidate) {
    const std::string library = Path("bare_prefix/lib/libxtb.6.dylib");
    EXPECT_EQ(detail::xtb_gfn0_parameter_file(library, Path("path_a"), ""), "");
    const std::string reason = detail::xtb_gfn0_missing_reason(library, Path("path_a"), "");
    EXPECT_NE(reason.find(File("path_a")), std::string::npos) << reason;
    EXPECT_NE(reason.find(File("bare_prefix/share/xtb")), std::string::npos) << reason;
    EXPECT_NE(reason.find("XTBPATH"), std::string::npos) << reason;
}

TEST_F(XtbParameterFileTest, ALibraryFoundByBareNameHasNoPrefixCandidate) {
    EXPECT_TRUE(detail::xtb_gfn0_parameter_candidates("libxtb.6.dylib", "", "").empty());
    EXPECT_EQ(detail::xtb_gfn0_parameter_file("libxtb.6.dylib", "", ""), "");
    const std::string reason = detail::xtb_gfn0_missing_reason("libxtb.6.dylib", "", "");
    EXPECT_NE(reason.find("names no directory"), std::string::npos) << reason;
}

TEST_F(XtbParameterFileTest, TheProcessEnvironmentIsRead) {
    const std::string library = Path("bare_prefix/lib/libxtb.6.dylib");
    {
        const test::ScopedEnvironment xtbpath("XTBPATH", Path("path_b").c_str());
        const test::ScopedEnvironment xtbhome("XTBHOME", nullptr);
        EXPECT_EQ(detail::xtb_gfn0_parameter_file(library), File("path_b"));
    }
    {
        const test::ScopedEnvironment xtbpath("XTBPATH", nullptr);
        const test::ScopedEnvironment xtbhome("XTBHOME", Path("home").c_str());
        EXPECT_EQ(detail::xtb_gfn0_parameter_file(library), File("home"));
    }
    {
        const test::ScopedEnvironment xtbpath("XTBPATH", Path("path_a").c_str());
        const test::ScopedEnvironment xtbhome("XTBHOME", nullptr);
        EXPECT_EQ(detail::xtb_gfn0_parameter_file(library), "");
        EXPECT_NE(detail::xtb_gfn0_missing_reason(library).find(File("path_a")), std::string::npos);
    }
}

TEST_F(XtbEngineTest, ReportsTheCApiVersion) {
    const XtbEngine engine(library_);
    EXPECT_TRUE(engine.IsAvailable()) << engine.GetUnavailableReason();
    EXPECT_EQ(engine.GetUnavailableReason(), "");
    EXPECT_EQ(engine.GetVersion(), "C API 1.0.0");
}

TEST_F(XtbEngineTest, DeclaresGfn0AndAnUnavailableGfnFf) {
    const XtbEngine engine(library_);
    ASSERT_EQ(engine.GetMethods().size(), 2u);
    const std::optional<MethodCapabilities> gfn0 = find_method(engine, Method(GFN0_XTB));
    ASSERT_TRUE(gfn0.has_value());
    EXPECT_TRUE(gfn0->available) << gfn0->unavailable_reason;
    EXPECT_TRUE(gfn0->solvent_models.empty());
    EXPECT_EQ(gfn0->properties, ALL_PROPERTIES);
    EXPECT_TRUE(gfn0->open_shell);
    EXPECT_FALSE(gfn0->charge_from_topology);
    EXPECT_FALSE(gfn0->force_field);
    EXPECT_TRUE(gfn0->analytic_gradient);
    EXPECT_FALSE(gfn0->analytic_hessian);
    ASSERT_EQ(gfn0->elements.size(), 86u);
    EXPECT_EQ(gfn0->elements.front(), 1u);
    EXPECT_EQ(gfn0->elements.back(), 86u);

    const std::optional<MethodCapabilities> gfnff = find_method(engine, Method(GFN_FF));
    ASSERT_TRUE(gfnff.has_value());
    EXPECT_FALSE(gfnff->available);
    EXPECT_EQ(gfnff->unavailable_reason, detail::XTB_GFNFF_UNAVAILABLE_REASON);
}

TEST_F(XtbEngineTest, GfnFfSessionsAreRefusedWithTheReason) {
    XtbEngine engine(library_);
    const MolecularSystem system = make_system(WATER, 0, 1);
    EngineSettings settings;
    settings.method = Method(GFN_FF);
    try {
        engine.CreateSession(system, settings);
        FAIL() << "expected EngineUnavailableError";
    } catch (const EngineUnavailableError& e) {
        EXPECT_NE(std::string(e.what()).find(detail::XTB_GFNFF_UNAVAILABLE_REASON),
                  std::string::npos);
    }
}

TEST_F(XtbEngineTest, Gfn0EnergiesMatchTheCApiProbe) {
    XtbEngine engine(library_);
    EXPECT_NEAR(evaluate(engine, make_system(ETHANOL, 0, 1), ETHANOL, EvaluationRequest()).energy,
                ETHANOL_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-6);
    EXPECT_NEAR(evaluate(engine, make_system(WATER, 0, 1), WATER, EvaluationRequest()).energy,
                WATER_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-6);
}

TEST_F(XtbEngineTest, ChargeAndUnpairedElectronsReachTheMolecule) {
    XtbEngine engine(library_);
    EvaluationRequest request;
    request.properties = static_cast<unsigned>(Property::CHARGES) |
                         static_cast<unsigned>(Property::ORBITAL_ENERGIES);
    const Evaluation hydroxide = evaluate(engine, make_system(HYDROXYL, -1, 1), HYDROXYL, request);
    EXPECT_NEAR(hydroxide.energy, HYDROXIDE_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-6);
    EXPECT_NEAR(hydroxide.charges[0] + hydroxide.charges[1], -1.0, 1e-8);
    // Triplet water has occupations 2, 2, 2, 1, 1, 0, 0, 0 (probe), so the HOMO is index 4;
    // a session that dropped the unpaired-electron count would return the singlet instead.
    const Evaluation triplet = evaluate(engine, make_system(WATER, 0, 3), WATER, request);
    EXPECT_NEAR(triplet.energy, TRIPLET_WATER_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-6);
    EXPECT_EQ(triplet.homo_index, 4);
    // The hydroxyl doublet's occupations are 2, 2, 1.5, 1.5, 0, 0 (probe), so the HOMO is 3.
    const Evaluation radical = evaluate(engine, make_system(HYDROXYL, 0, 2), HYDROXYL, request);
    EXPECT_NEAR(radical.energy, HYDROXYL_RADICAL_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-6);
    EXPECT_EQ(radical.homo_index, 3);
}

TEST_F(XtbEngineTest, AccuracyReachesTheCalculator) {
    // GFN0-xTB is not self-consistent, so accuracy only moves integral screening: the probe
    // measured E(1000) - E(1) = 1.05e-11 Eh = 6.6e-9 kcal/mol for ethanol. libxtb's default
    // accuracy is 1.0, so a session that never called xtb_setAccuracy would give equal values.
    XtbEngine engine(library_);
    const MolecularSystem system = make_system(ETHANOL, 0, 1);
    const double tight = evaluate(engine, system, ETHANOL, EvaluationRequest(), 1.0).energy;
    const double loose = evaluate(engine, system, ETHANOL, EvaluationRequest(), 1000.0).energy;
    EXPECT_GT(std::abs(loose - tight), 1e-9);
    EXPECT_NEAR(loose, tight, 1e-6);
}

TEST_F(XtbEngineTest, Gfn0GradientMatchesCentralDifferencesOfItsEnergy) {
    XtbEngine engine(library_);
    const MolecularSystem system = make_system(ETHANOL, 0, 1);
    auto session = engine.CreateSession(system, gfn0_settings());
    EvaluationRequest request;
    request.gradient = true;
    const Evaluation analytic = session->Evaluate(ETHANOL.coords, request);
    ASSERT_EQ(analytic.gradient.size(), ETHANOL.coords.size());
    // The probe found a largest deviation of 9.0e-6 kcal/mol/Angstrom at this step.
    const double step = 1e-4;
    for (std::size_t k = 0; k < ETHANOL.coords.size(); ++k) {
        std::vector<double> plus = ETHANOL.coords;
        std::vector<double> minus = ETHANOL.coords;
        plus[k] += step;
        minus[k] -= step;
        const double numeric = (session->Evaluate(plus, EvaluationRequest()).energy -
                                session->Evaluate(minus, EvaluationRequest()).energy) /
                               (2.0 * step);
        EXPECT_NEAR(analytic.gradient[k], numeric, 1e-4) << "component " << k;
    }
}

TEST_F(XtbEngineTest, Gfn0PropertiesMatchTheCApiProbe) {
    XtbEngine engine(library_);
    EvaluationRequest request;
    request.properties = ALL_PROPERTIES;
    const Evaluation result = evaluate(engine, make_system(ETHANOL, 0, 1), ETHANOL, request);

    const std::vector<double> charges = {
        -0.3913919046692989, 0.16331433466392375, -0.5510316240393265,
        0.15287845415957457, 0.15160259447564736, 0.11256780161050041,
        0.0399718325472433,  0.04170455583261888, 0.28038395541911715};
    ASSERT_EQ(result.charges.size(), charges.size());
    double total = 0.0;
    for (std::size_t i = 0; i < charges.size(); ++i) {
        EXPECT_NEAR(result.charges[i], charges[i], 1e-6) << "atom " << i;
        total += result.charges[i];
    }
    EXPECT_NEAR(total, 0.0, 1e-8);

    const double dipole_ebohr[3] = {0.17445201779723293, -0.6884117330462874, 0.05535563094984518};
    ASSERT_EQ(result.dipole.size(), 3u);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(result.dipole[axis], dipole_ebohr[axis] * Units::EBOHR_TO_DEBYE, 1e-6);
    }

    // Exactly the eight bonded pairs reach the 0.1 reporting threshold; the largest other
    // entry is 0.018.
    const std::vector<BondOrder> bonds = {{0, 1, 1.0261825240871665}, {0, 3, 0.9817868195317259},
                                          {0, 4, 0.9813514481990051}, {0, 5, 0.9803161382471591},
                                          {1, 2, 0.9866956146809046}, {1, 6, 0.9774747045874694},
                                          {1, 7, 0.9768758128940933}, {2, 8, 0.9856266006513925}};
    ASSERT_EQ(result.bond_orders.size(), bonds.size());
    for (std::size_t k = 0; k < bonds.size(); ++k) {
        EXPECT_EQ(result.bond_orders[k].i, bonds[k].i);
        EXPECT_EQ(result.bond_orders[k].j, bonds[k].j);
        EXPECT_NEAR(result.bond_orders[k].order, bonds[k].order, 1e-6);
    }

    // 24 orbitals, ten doubly occupied (20 valence electrons).
    ASSERT_EQ(result.orbital_energies.size(), 24u);
    EXPECT_EQ(result.homo_index, 9);
    EXPECT_NEAR(result.orbital_energies.front(), -0.6922356655194147 * Units::HARTREE_TO_EV, 1e-6);
    EXPECT_NEAR(result.orbital_energies[9], -0.41394267915962196 * Units::HARTREE_TO_EV, 1e-6);
    EXPECT_NEAR(result.orbital_energies[10], -0.17063774355135816 * Units::HARTREE_TO_EV, 1e-6);
    EXPECT_LT(result.orbital_energies[9], result.orbital_energies[10]);
    EXPECT_EQ(validate_evaluation(result, request, 9), "");
}

TEST_F(XtbEngineTest, Gfn0IsGasPhaseOnly) {
    XtbEngine engine(library_);
    EngineSettings settings = gfn0_settings();
    settings.solvent = Solvent("water", SolventModel::GBSA);
    EXPECT_THROW(engine.CreateSession(make_system(WATER, 0, 1), settings), OptionsError);
}

TEST_F(XtbEngineTest, MalformedCoordinatesAreRejected) {
    XtbEngine engine(library_);
    auto session = engine.CreateSession(make_system(WATER, 0, 1), gfn0_settings());
    EXPECT_THROW(session->Evaluate({0.0, 0.0, 0.0}, EvaluationRequest()), std::runtime_error);
}

// Spec 6.4: a session is never shared across conformers, so results do not depend on conformer
// order. Calculation gives every conformer its own session, so a session has to start from the
// coordinates it is asked to evaluate, never from conformer 0: here one conformer has two
// coincident atoms, which libxtb refuses to build a molecule from.
TEST_F(XtbEngineTest, AConformersResultDoesNotDependOnItsPosition) {
    XtbEngine engine(library_);
    const Geometry coincident = {WATER.numbers,
                                 {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, 0.7572, -0.4692}};
    const MolecularSystem valid_first = make_two_conformers(WATER, coincident);
    const MolecularSystem invalid_first = make_two_conformers(coincident, WATER);
    ASSERT_EQ(valid_first.NumConformers(), 2u);
    ASSERT_EQ(invalid_first.NumConformers(), 2u);

    // OpenEye stores single precision, so compare the conformers with the same coordinates alone.
    const double alone = evaluate_conformer(engine, make_system(WATER, 0, 1), 0);
    EXPECT_NEAR(alone, WATER_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-3);
    double valid_at_first = 0.0;
    double valid_at_second = 0.0;
    ASSERT_NO_THROW(valid_at_first = evaluate_conformer(engine, valid_first, 0));
    ASSERT_NO_THROW(valid_at_second = evaluate_conformer(engine, invalid_first, 1));
    EXPECT_NEAR(valid_at_first, alone, 1e-8);
    EXPECT_NEAR(valid_at_second, alone, 1e-8);
    EXPECT_THROW(evaluate_conformer(engine, valid_first, 1), std::runtime_error);
    EXPECT_THROW(evaluate_conformer(engine, invalid_first, 0), std::runtime_error);
}

// libxtb refuses a structure with coincident atoms ("Could not generate molecular structure",
// probe 2026-10-11); that must surface as a plain runtime error, which a calculation reports as
// ENGINE_ERROR, and never as UnsupportedError. The session builds its molecule from the first
// evaluation's coordinates, so the error comes from that evaluation.
TEST_F(XtbEngineTest, LibxtbErrorsBecomeRuntimeErrors) {
    XtbEngine engine(library_);
    const Geometry coincident = {{1, 1}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    auto session = engine.CreateSession(make_system(coincident, 0, 1), gfn0_settings());
    try {
        session->Evaluate(coincident.coords, EvaluationRequest());
        FAIL() << "expected std::runtime_error";
    } catch (const UnsupportedError& e) {
        FAIL() << "libxtb errors must not be UnsupportedError: " << e.what();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("xtb_newMolecule"), std::string::npos) << e.what();
    }
}

// A first evaluation that libxtb rejects releases what it built, so the next evaluation builds
// the session again from its own coordinates.
TEST_F(XtbEngineTest, ASessionWhoseFirstEvaluationFailedBuildsItselfAgain) {
    XtbEngine engine(library_);
    auto session = engine.CreateSession(make_system(WATER, 0, 1), gfn0_settings());
    const std::vector<double> coincident = {0.0,     0.0, 0.1173, 0.0,    0.7572,
                                            -0.4692, 0.0, 0.7572, -0.4692};
    EXPECT_THROW(session->Evaluate(coincident, EvaluationRequest()), std::runtime_error);
    EXPECT_NEAR(session->Evaluate(WATER.coords, EvaluationRequest()).energy,
                WATER_ENERGY_EH * Units::HARTREE_TO_KCAL, 1e-6);
}

// Two threads create sessions on one engine that has not loaded libxtb yet, then evaluate
// concurrently: the load mutex and the process-wide libxtb mutex must keep both correct, and
// ThreadSanitizer checks the OEVibe side of every call.
TEST_F(XtbEngineTest, ConcurrentSessionsInTwoThreads) {
    XtbEngine engine(library_);
    const MolecularSystem ethanol = make_system(ETHANOL, 0, 1);
    const MolecularSystem water = make_system(WATER, 0, 1);
    struct Job {
        const MolecularSystem* system;
        const Geometry* geometry;
        double expected;
        std::vector<double> energies;
        std::string error;
    };
    Job jobs[2] = {{&ethanol, &ETHANOL, ETHANOL_ENERGY_EH * Units::HARTREE_TO_KCAL, {}, {}},
                   {&water, &WATER, WATER_ENERGY_EH * Units::HARTREE_TO_KCAL, {}, {}}};
    std::vector<std::thread> workers;
    for (Job& job : jobs) {
        workers.emplace_back([&engine, &job] {
            try {
                auto session = engine.CreateSession(*job.system, gfn0_settings());
                EvaluationRequest request;
                request.gradient = true;
                for (int round = 0; round < 10; ++round) {
                    job.energies.push_back(session->Evaluate(job.geometry->coords, request).energy);
                }
            } catch (const std::exception& e) {
                job.error = e.what();
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const Job& job : jobs) {
        EXPECT_EQ(job.error, "");
        ASSERT_EQ(job.energies.size(), 10u);
        for (double energy : job.energies) {
            EXPECT_NEAR(energy, job.expected, 1e-6);
        }
    }
}
