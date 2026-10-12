#include <gtest/gtest.h>

#include <oechem.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engines/RuntimeLibrary.h"
#include "engines/TbliteLibrary.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"
#include "oevibe/engines/TbliteEngine.h"
#include "test_runtime_libraries.h"

using namespace OEVIBE;

namespace {

// Reference energies in Hartree, computed with tblite 0.6.0 and dftd4 4.2.0 build 0 through the
// C API at accuracy 1 on exactly these geometries (single precision, as OpenEye stores them). The
// gas and ALPB water values agree with the xtb 6.7.1 binary to 2e-9 Eh.
constexpr double WATER_GFN2 = -5.0703707647;
constexpr double WATER_GFN2_ALPB_WATER = -5.0848282772;
constexpr double WATER_GFN2_GBSA_WATER = -5.0799674119;
constexpr double WATER_GFN1 = -5.7686405798;
constexpr double WATER_GFN1_ALPB_WATER = -5.7900673572;
constexpr double ETHANOL_GFN2 = -11.3904969382;
constexpr double HYDROXIDE_GFN2 = -4.6816117664;
constexpr double HYDROXYL_RADICAL_GFN2 = -4.4283391093;
constexpr double DIOXYGEN_SINGLET_GFN2 = -7.9067523569;
constexpr double DIOXYGEN_TRIPLET_GFN2 = -7.9041182796;
// The same library at the same accuracy from the same starting guess reproduces these to about
// 1e-10 Eh (the probe saw no change between accuracy 1 and 0.01), so 1e-6 Eh (6.3e-4 kcal/mol)
// only absorbs thread-order rounding, while any wrong unit, charge, spin, method, or parameter
// set moves the energy by more than 1e-3 Eh.
constexpr double ENERGY_TOLERANCE = 1e-6 * Units::HARTREE_TO_KCAL;

OEChem::OEGraphMol make_molecule(const std::vector<unsigned>& numbers,
                                 const std::vector<double>& coords,
                                 const std::vector<std::pair<unsigned, unsigned>>& bonds) {
    OEChem::OEGraphMol mol;
    std::vector<OEChem::OEAtomBase*> atoms;
    for (unsigned number : numbers) {
        atoms.push_back(mol.NewAtom(number));
    }
    for (const auto& bond : bonds) {
        mol.NewBond(atoms[bond.first], atoms[bond.second], 1);
    }
    mol.SetCoords(coords.data());
    mol.SetDimension(3);
    return mol;
}

OEChem::OEGraphMol water() {
    return make_molecule({8, 1, 1}, {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572, -0.4692},
                         {{0, 1}, {0, 2}});
}

// An approximate ethanol geometry (C, C, O, then the hydrogens), deliberately not a minimum so
// the gradient is large.
OEChem::OEGraphMol ethanol() {
    return make_molecule({6, 6, 8, 1, 1, 1, 1, 1, 1},
                         {-0.748, -0.015, 0.024,  0.558,  0.420,  -0.619, 0.599,  1.811,  -0.873,
                          -1.564, 0.226,  -0.657, -0.865, 0.495,  0.985,  -0.764, -1.094, 0.210,
                          1.398,  0.169,  0.035,  0.687,  -0.104, -1.572, 1.463,  2.034,  -1.247},
                         {{0, 1}, {1, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 6}, {1, 7}, {2, 8}});
}

OEChem::OEGraphMol hydroxyl() {
    return make_molecule({8, 1}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.97}, {{0, 1}});
}

OEChem::OEGraphMol dioxygen() {
    return make_molecule({8, 8}, {0.0, 0.0, 0.0, 0.0, 0.0, 1.21}, {{0, 1}});
}

EngineSettings settings_for(const char* method, const Solvent& solvent = Solvent()) {
    EngineSettings settings;
    settings.method = Method(method);
    settings.solvent = solvent;
    return settings;
}

double energy_of(TbliteEngine& engine, const MolecularSystem& system,
                 const EngineSettings& settings) {
    auto session = engine.CreateSession(system, settings);
    return session->Evaluate(system.GetConformerCoordinates(0), EvaluationRequest()).energy;
}

/// A scratch directory that is removed with the object.
class ScratchDirectory {
public:
    explicit ScratchDirectory(const std::string& name)
        : path_(std::filesystem::temp_directory_path() /
                (name + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::create_directories(path_);
    }
    ~ScratchDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;
    const std::filesystem::path& Path() const { return path_; }

private:
    std::filesystem::path path_;
};

/// Creates <prefix>/conda-meta with one empty record per file name; the guard reads names only.
void write_conda_meta(const std::filesystem::path& prefix, const std::vector<std::string>& files) {
    std::filesystem::create_directories(prefix / "conda-meta");
    for (const std::string& file : files) {
        std::ofstream(prefix / "conda-meta" / file) << "{}\n";
    }
}

/// Engine tests that need libtblite; each skips when ``OEVIBE_TBLITE_LIBRARY`` is unusable.
class TbliteEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        library_ = test::tblite_library_path();
        if (library_.empty()) {
            GTEST_SKIP() << test::TBLITE_SKIP_REASON;
        }
        engine_ = std::make_unique<TbliteEngine>(library_);
        ASSERT_TRUE(engine_->IsAvailable()) << engine_->GetUnavailableReason();
    }

    TbliteEngine& Tblite() { return *engine_; }

    std::string library_;
    std::unique_ptr<TbliteEngine> engine_;
};

}  // namespace

TEST(TbliteLibraryTest, OnlyThe06And07SeriesAreSupported) {
    for (int version : {600, 601, 699, 700, 712, 799}) {
        EXPECT_TRUE(detail::tblite_version_supported(version)) << version;
    }
    for (int version : {0, 500, 599, 800, 10000}) {
        EXPECT_FALSE(detail::tblite_version_supported(version)) << version;
    }
    EXPECT_EQ(detail::tblite_version_text(600), "0.6.0");
    EXPECT_EQ(detail::tblite_version_text(700), "0.7.0");
    EXPECT_EQ(detail::tblite_version_text(612), "0.6.12");
    EXPECT_EQ(detail::tblite_version_text(10203), "1.2.3");
}

TEST(TbliteLibraryTest, KnownBadPairingIsReadFromCondaMeta) {
    const ScratchDirectory scratch("oevibe-pairing");
    const auto library_in = [&scratch](const std::string& environment) {
        return (scratch.Path() / environment / "lib" / "libtblite.0.dylib").string();
    };
    // The record names of the environment that hangs, as conda-forge writes them.
    write_conda_meta(scratch.Path() / "bad", {"tblite-0.6.0-gfortran_hc17bbfb_0.json",
                                              "dftd4-4.2.0-gfortran_hc17bbfb_1.json",
                                              "xtb-6.7.1-gfortran_hc17bbfb_5.json"});
    const std::optional<std::string> reason = detail::known_bad_tblite_pairing(library_in("bad"));
    ASSERT_TRUE(reason.has_value());
    EXPECT_NE(reason->find("dftd4-4.2.0-gfortran_hc17bbfb_1.json"), std::string::npos) << *reason;
    EXPECT_NE(reason->find("tblite-0.6.0-gfortran_hc17bbfb_0.json"), std::string::npos) << *reason;

    // The working pin, tblite 0.7 with build 1, and a different package whose name starts with
    // "tblite" are all accepted, as is a directory without conda-meta.
    write_conda_meta(scratch.Path() / "pinned", {"tblite-0.6.0-gfortran_hc17bbfb_0.json",
                                                 "dftd4-4.2.0-gfortran_hc17bbfb_0.json"});
    write_conda_meta(scratch.Path() / "newer", {"tblite-0.7.0-gfortran_hc17bbfb_0.json",
                                                "dftd4-4.2.0-gfortran_hc17bbfb_1.json"});
    write_conda_meta(scratch.Path() / "python", {"tblite-python-0.6.0-py312h0_0.json",
                                                 "dftd4-4.2.0-gfortran_hc17bbfb_1.json"});
    std::filesystem::create_directories(scratch.Path() / "plain" / "lib");
    for (const char* environment : {"pinned", "newer", "python", "plain"}) {
        EXPECT_FALSE(detail::known_bad_tblite_pairing(library_in(environment)).has_value())
            << environment;
    }
    // A bare file name, as the system loader takes it, has no prefix to inspect.
    EXPECT_FALSE(detail::known_bad_tblite_pairing("libtblite.0.dylib").has_value());
}

TEST(TbliteEngineSchemaTest, MethodsAndKeywordsNeedNoLibrary) {
    const TbliteEngine engine("/nonexistent/lib/libtblite.0.dylib");
    EXPECT_EQ(engine.GetName(), "tblite");
    const std::vector<MethodCapabilities> methods = engine.GetMethods();
    ASSERT_EQ(methods.size(), 2u);
    EXPECT_EQ(methods[0].method, Method(GFN2_XTB));
    EXPECT_EQ(methods[1].method, Method(GFN1_XTB));
    for (const MethodCapabilities& caps : methods) {
        EXPECT_TRUE(caps.available);
        EXPECT_EQ(caps.solvent_models,
                  (std::vector<SolventModel>{SolventModel::ALPB, SolventModel::GBSA}));
        EXPECT_TRUE(caps.solvents.empty());
        EXPECT_TRUE(caps.analytic_gradient);
        EXPECT_FALSE(caps.analytic_hessian);
        EXPECT_TRUE(caps.open_shell);
        EXPECT_FALSE(caps.charge_from_topology);
        EXPECT_FALSE(caps.requires_partial_charges);
        EXPECT_FALSE(caps.force_field);
        EXPECT_EQ(caps.properties, ALL_PROPERTIES);
        ASSERT_EQ(caps.elements.size(), 86u);
        EXPECT_EQ(caps.elements.front(), 1u);
        EXPECT_EQ(caps.elements.back(), 86u);
    }

    const std::vector<KeywordSpec> specs = engine.GetKeywordSpecs();
    ASSERT_EQ(specs.size(), 3u);
    const auto expect_spec = [](const KeywordSpec& spec, const char* name, KeywordType type,
                                double min, double max) {
        EXPECT_EQ(spec.name, name);
        EXPECT_EQ(spec.type, type) << name;
        EXPECT_TRUE(spec.methods.empty()) << name;
        EXPECT_TRUE(spec.has_min && spec.has_max) << name;
        EXPECT_DOUBLE_EQ(spec.min, min) << name;
        EXPECT_DOUBLE_EQ(spec.max, max) << name;
    };
    expect_spec(specs[0], "accuracy", KeywordType::DOUBLE, 1e-4, 1e3);
    expect_spec(specs[1], "max_iterations", KeywordType::INT, 1, 10000);
    expect_spec(specs[2], "electronic_temperature", KeywordType::DOUBLE, 1, 10000);

    EngineSettings settings = settings_for(GFN1_XTB);
    settings.keywords = {
        {"accuracy", "0.5"}, {"max_iterations", "100"}, {"electronic_temperature", "500"}};
    EXPECT_NO_THROW(engine.ValidateSettings(settings));
    for (const auto& [name, value] :
         std::vector<std::pair<std::string, std::string>>{{"max_iterations", "0"},
                                                          {"max_iterations", "2.5"},
                                                          {"accuracy", "1e4"},
                                                          {"electronic_temperature", "0.5"},
                                                          {"unknown", "1"}}) {
        EngineSettings bad = settings_for(GFN2_XTB);
        bad.keywords = {{name, value}};
        EXPECT_THROW(engine.ValidateSettings(bad), OptionsError) << name << "=" << value;
    }
    EXPECT_THROW(engine.ValidateSettings(settings_for(GFN0_XTB)), OptionsError);
}

TEST(TbliteEngineSchemaTest, AMissingLibraryMakesTheEngineUnavailable) {
    const test::ScopedEnvironment variable("OEVIBE_TBLITE_LIBRARY", nullptr);
    const test::ScopedEnvironment conda("CONDA_PREFIX", nullptr);
    const test::ScopedPythonLibraryDirectory python("");
    // The bare file names still reach the system loader; a machine with a system-wide libtblite
    // cannot show a missing library.
    detail::LibrarySearch system;
    system.engine = "tblite";
    system.file_names = detail::library_file_names("tblite", "0");
    std::string ignored;
    if (detail::RuntimeLibrary::Load(system, ignored)) {
        GTEST_SKIP() << "the system loader finds a libtblite";
    }

    const TbliteEngine engine("/nonexistent/lib/libtblite.0.dylib");
    EXPECT_FALSE(engine.IsAvailable());
    EXPECT_EQ(engine.GetVersion(), "");
    EXPECT_NE(engine.GetUnavailableReason().find("/nonexistent/lib/libtblite.0.dylib"),
              std::string::npos)
        << engine.GetUnavailableReason();
    const OEChem::OEGraphMol mol = water();
    EXPECT_THROW(TbliteEngine("/nonexistent/lib/libtblite.0.dylib")
                     .CreateSession(MolecularSystem(mol, 0, 1), settings_for(GFN2_XTB)),
                 EngineUnavailableError);

    // Instances built without a path take SetLibraryPath's value when they first load.
    TbliteEngine::SetLibraryPath("/elsewhere/lib/libtblite.0.dylib");
    const TbliteEngine builtin;
    const std::string reason = builtin.GetUnavailableReason();
    TbliteEngine::SetLibraryPath("");
    EXPECT_NE(reason.find("/elsewhere/lib/libtblite.0.dylib"), std::string::npos) << reason;
}

TEST_F(TbliteEngineTest, ReportsTheLoadedVersion) {
    const std::string version = Tblite().GetVersion();
    EXPECT_TRUE(version.rfind("0.6.", 0) == 0 || version.rfind("0.7.", 0) == 0) << version;
    EXPECT_EQ(Tblite().GetUnavailableReason(), "");
}

TEST_F(TbliteEngineTest, EnergiesMatchTheReferenceValues) {
    const OEChem::OEGraphMol h2o = water();
    const OEChem::OEGraphMol etoh = ethanol();
    const MolecularSystem water_system(h2o, 0, 1);
    const MolecularSystem ethanol_system(etoh, 0, 1);
    EXPECT_NEAR(energy_of(Tblite(), water_system, settings_for(GFN2_XTB)),
                WATER_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
    EXPECT_NEAR(energy_of(Tblite(), ethanol_system, settings_for(GFN2_XTB)),
                ETHANOL_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
    EXPECT_NEAR(energy_of(Tblite(), water_system, settings_for(GFN1_XTB)),
                WATER_GFN1 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
}

// The optimizer and the finite-difference Hessian (spec 7.1, 7.2) need the gradient to be the
// derivative of the reported energy in kcal/mol/Angstrom, so it is checked against central
// differences of the engine's own energy.
TEST_F(TbliteEngineTest, GradientsMatchCentralDifferences) {
    const OEChem::OEGraphMol etoh = ethanol();
    const MolecularSystem system(etoh, 0, 1);
    for (const Solvent& solvent : {Solvent(), Solvent("water", SolventModel::ALPB)}) {
        SCOPED_TRACE(solvent.GetTag());
        EngineSettings settings = settings_for(GFN2_XTB, solvent);
        // Tight SCF convergence keeps the energy noise far below the difference quotient's.
        settings.accuracy = 0.01;
        auto session = Tblite().CreateSession(system, settings);
        const std::vector<double> x = system.GetConformerCoordinates(0);
        EvaluationRequest request;
        request.gradient = true;
        const Evaluation analytic = session->Evaluate(x, request);
        ASSERT_EQ(analytic.gradient.size(), x.size());
        // With this step and accuracy the probe found the two gradients within 8.8e-4
        // kcal/mol/Angstrom (gas and ALPB), while components reach 8 kcal/mol/Angstrom; a unit
        // slip or a swapped component would be off by more than 0.1.
        const double step = 1e-3;
        double largest = 0.0;
        for (std::size_t k = 0; k < x.size(); ++k) {
            std::vector<double> plus = x;
            std::vector<double> minus = x;
            plus[k] += step;
            minus[k] -= step;
            const double numeric = (session->Evaluate(plus, EvaluationRequest()).energy -
                                    session->Evaluate(minus, EvaluationRequest()).energy) /
                                   (2.0 * step);
            EXPECT_NEAR(analytic.gradient[k], numeric, 5e-3) << "component " << k;
            largest = std::max(largest, std::abs(analytic.gradient[k]));
        }
        EXPECT_GT(largest, 1.0);
    }
}

TEST_F(TbliteEngineTest, ImplicitSolventUsesTheMethodsParameterSets) {
    const OEChem::OEGraphMol h2o = water();
    const MolecularSystem system(h2o, 0, 1);
    const double gas = energy_of(Tblite(), system, settings_for(GFN2_XTB));
    const double alpb =
        energy_of(Tblite(), system, settings_for(GFN2_XTB, Solvent("water", SolventModel::ALPB)));
    const double gbsa =
        energy_of(Tblite(), system, settings_for(GFN2_XTB, Solvent("water", SolventModel::GBSA)));
    // Water is stabilized in water: by 9.07 kcal/mol with ALPB and 6.02 with GBSA here.
    EXPECT_LT(alpb, gas - 5.0);
    EXPECT_NEAR(alpb - gas, (WATER_GFN2_ALPB_WATER - WATER_GFN2) * Units::HARTREE_TO_KCAL,
                2.0 * ENERGY_TOLERANCE);
    EXPECT_NEAR(gbsa - gas, (WATER_GFN2_GBSA_WATER - WATER_GFN2) * Units::HARTREE_TO_KCAL,
                2.0 * ENERGY_TOLERANCE);
    // AUTO, which only a direct caller can pass, is the method's default model, ALPB.
    EXPECT_NEAR(energy_of(Tblite(), system, settings_for(GFN2_XTB, Solvent("water"))), alpb,
                ENERGY_TOLERANCE);
    // GFN1 takes its own ALPB parameters; the GFN2 set would give -5.7886588482 Eh (probe).
    EXPECT_NEAR(
        energy_of(Tblite(), system, settings_for(GFN1_XTB, Solvent("water", SolventModel::ALPB))),
        WATER_GFN1_ALPB_WATER * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
}

TEST_F(TbliteEngineTest, ChargeAndSpinReachTheLibrary) {
    const OEChem::OEGraphMol oh = hydroxyl();
    const OEChem::OEGraphMol o2 = dioxygen();
    EXPECT_NEAR(energy_of(Tblite(), MolecularSystem(oh, -1, 1), settings_for(GFN2_XTB)),
                HYDROXIDE_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
    EXPECT_NEAR(energy_of(Tblite(), MolecularSystem(oh, 0, 2), settings_for(GFN2_XTB)),
                HYDROXYL_RADICAL_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
    // Two unpaired electrons change the energy of the same 16-electron molecule.
    EXPECT_NEAR(energy_of(Tblite(), MolecularSystem(o2, 0, 1), settings_for(GFN2_XTB)),
                DIOXYGEN_SINGLET_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
    EXPECT_NEAR(energy_of(Tblite(), MolecularSystem(o2, 0, 3), settings_for(GFN2_XTB)),
                DIOXYGEN_TRIPLET_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
}

TEST_F(TbliteEngineTest, KeywordsAndAccuracyReachTheCalculator) {
    const OEChem::OEGraphMol oh = hydroxyl();
    const MolecularSystem radical(oh, 0, 2);
    const double default_temperature = energy_of(Tblite(), radical, settings_for(GFN2_XTB));
    EngineSettings settings = settings_for(GFN2_XTB);
    // tblite's default is 300 K with its own Boltzmann constant; the probe saw 8e-7 kcal/mol
    // between that and 300 K converted with CODATA 2018.
    settings.keywords = {{"electronic_temperature", "300"}};
    EXPECT_NEAR(energy_of(Tblite(), radical, settings), default_temperature, 1e-5);
    // Smearing over 3000 K lowers the radical's energy by 7.44 kcal/mol (probe).
    settings.keywords = {{"electronic_temperature", "3000"}};
    EXPECT_NEAR(energy_of(Tblite(), radical, settings) - default_temperature,
                (-4.4401937066 - HYDROXYL_RADICAL_GFN2) * Units::HARTREE_TO_KCAL, 1e-3);

    const OEChem::OEGraphMol etoh = ethanol();
    const MolecularSystem system(etoh, 0, 1);
    settings.keywords = {{"max_iterations", "1"}};
    auto session = Tblite().CreateSession(system, settings);
    try {
        session->Evaluate(system.GetConformerCoordinates(0), EvaluationRequest());
        FAIL() << "expected the SCF iteration limit to stop the evaluation";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("SCF not converged"), std::string::npos) << e.what();
    }

    // The loosest accuracy stops the SCF early: 0.040 kcal/mol above the converged energy
    // (probe), against no visible change between accuracy 1 and 0.01.
    EngineSettings loose = settings_for(GFN2_XTB);
    loose.accuracy = 1e3;
    EXPECT_GT(std::abs(energy_of(Tblite(), system, loose) - ETHANOL_GFN2 * Units::HARTREE_TO_KCAL),
              0.01);
}

TEST_F(TbliteEngineTest, AnUnknownSolventIsAnEngineErrorNamingIt) {
    const OEChem::OEGraphMol h2o = water();
    const MolecularSystem system(h2o, 0, 1);
    for (SolventModel model : {SolventModel::ALPB, SolventModel::GBSA}) {
        try {
            Tblite().CreateSession(system, settings_for(GFN2_XTB, Solvent("notasolvent", model)));
            FAIL() << "expected tblite to reject the solvent";
        } catch (const UnsupportedError& e) {
            FAIL() << "an unknown solvent is an engine error, not UNSUPPORTED: " << e.what();
        } catch (const std::runtime_error& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find("notasolvent"), std::string::npos) << message;
            // tblite's own text follows.
            EXPECT_NE(message.find("database of solvents"), std::string::npos) << message;
        }
    }
}

TEST_F(TbliteEngineTest, OneSessionFollowsItsGeometry) {
    const OEChem::OEGraphMol etoh = ethanol();
    const MolecularSystem system(etoh, 0, 1);
    const std::vector<double> first = system.GetConformerCoordinates(0);
    std::vector<double> second = first;
    for (std::size_t k = 0; k < second.size(); ++k) {
        second[k] += 0.02 * std::sin(1.7 * static_cast<double>(k) + 0.3);
    }
    auto session = Tblite().CreateSession(system, settings_for(GFN2_XTB));
    const double at_first = session->Evaluate(first, EvaluationRequest()).energy;
    const double at_second = session->Evaluate(second, EvaluationRequest()).energy;
    const double back_at_first = session->Evaluate(first, EvaluationRequest()).energy;
    auto fresh = Tblite().CreateSession(system, settings_for(GFN2_XTB));
    EXPECT_GT(std::abs(at_second - at_first), 0.1);
    EXPECT_NEAR(back_at_first, at_first, 1e-6);
    EXPECT_NEAR(at_second, fresh->Evaluate(second, EvaluationRequest()).energy, 1e-6);
}

TEST_F(TbliteEngineTest, RejectsWhatItCannotCompute) {
    const OEChem::OEGraphMol h2o = water();
    const MolecularSystem system(h2o, 0, 1);
    auto session = Tblite().CreateSession(system, settings_for(GFN2_XTB));
    EvaluationRequest hessian;
    hessian.hessian = true;
    EXPECT_THROW(session->Evaluate(system.GetConformerCoordinates(0), hessian), std::runtime_error);
    EXPECT_THROW(session->Evaluate(std::vector<double>(6, 0.0), EvaluationRequest()),
                 std::invalid_argument);
    EXPECT_THROW(Tblite().CreateSession(
                     system, settings_for(GFN2_XTB, Solvent("water", SolventModel::SHEFFIELD))),
                 OptionsError);
}

TEST_F(TbliteEngineTest, KnownBadPairingMakesTheEngineUnavailable) {
    const ScratchDirectory scratch("oevibe-bad-prefix");
    const std::filesystem::path prefix = scratch.Path() / "env";
    std::filesystem::create_directories(prefix / "lib");
    const std::filesystem::path link = prefix / "lib" / "libtblite.0.dylib";
    std::filesystem::create_symlink(std::filesystem::absolute(library_), link);
    write_conda_meta(
        prefix, {"tblite-0.6.0-gfortran_hc17bbfb_0.json", "dftd4-4.2.0-gfortran_hc17bbfb_1.json"});
    const TbliteEngine engine(link.string());
    EXPECT_FALSE(engine.IsAvailable());
    EXPECT_NE(engine.GetUnavailableReason().find("dftd4"), std::string::npos)
        << engine.GetUnavailableReason();
}

// Holding the process-wide mutex must stall another thread's session creation, evaluation, and
// destruction, which proves that each takes it.
TEST_F(TbliteEngineTest, LibraryCallsWaitForTheProcessWideMutex) {
    const OEChem::OEGraphMol h2o = water();
    const MolecularSystem system(h2o, 0, 1);
    std::shared_ptr<EngineSession> session;
    const auto expect_blocked = [](const char* what, const std::function<void()>& work) {
        SCOPED_TRACE(what);
        std::atomic<bool> done{false};
        std::string failure;
        std::unique_lock<std::mutex> hold(detail::tblite_mutex());
        std::thread worker([&]() {
            try {
                work();
            } catch (const std::exception& e) {
                failure = e.what();
            }
            done = true;
        });
        // Unblocked, each of these calls takes milliseconds on water.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        EXPECT_FALSE(done.load());
        hold.unlock();
        worker.join();
        EXPECT_TRUE(done.load());
        EXPECT_EQ(failure, "");
    };
    expect_blocked("CreateSession",
                   [&]() { session = Tblite().CreateSession(system, settings_for(GFN2_XTB)); });
    ASSERT_NE(session, nullptr);
    expect_blocked("Evaluate", [&]() {
        session->Evaluate(system.GetConformerCoordinates(0), EvaluationRequest());
    });
    expect_blocked("destruction", [&]() { session.reset(); });
}

// Two threads load one fresh engine and evaluate their own sessions at the same time; the
// mutex serializes the library, so both reproduce the sequential energy. TSan checks this test.
TEST_F(TbliteEngineTest, ConcurrentSessionsMatchSequentialResults) {
    const OEChem::OEGraphMol etoh = ethanol();
    const MolecularSystem system(etoh, 0, 1);
    TbliteEngine shared(library_);
    std::vector<double> energies(2, 0.0);
    std::vector<std::string> failures(2);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            try {
                if (!shared.IsAvailable()) {
                    failures[t] = shared.GetUnavailableReason();
                    return;
                }
                auto session = shared.CreateSession(system, settings_for(GFN2_XTB));
                for (int repeat = 0; repeat < 3; ++repeat) {
                    energies[t] =
                        session->Evaluate(system.GetConformerCoordinates(0), EvaluationRequest())
                            .energy;
                }
            } catch (const std::exception& e) {
                failures[t] = e.what();
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (std::size_t t = 0; t < 2; ++t) {
        EXPECT_EQ(failures[t], "") << "thread " << t;
        EXPECT_NEAR(energies[t], ETHANOL_GFN2 * Units::HARTREE_TO_KCAL, ENERGY_TOLERANCE);
    }
}

namespace {

EvaluationRequest all_properties() {
    EvaluationRequest request;
    request.properties = ALL_PROPERTIES;
    return request;
}

double sum_of(const std::vector<double>& values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum;
}

}  // namespace

// Reference values come from the same tblite 0.6.0 computation as the energies above, converted
// from e Bohr and Hartree with the Units constants.
TEST_F(TbliteEngineTest, WaterPropertiesAreValidAndPhysical) {
    const OEChem::OEGraphMol h2o = water();
    const MolecularSystem system(h2o, 0, 1);
    auto session = Tblite().CreateSession(system, settings_for(GFN2_XTB));
    const EvaluationRequest request = all_properties();
    const Evaluation evaluation = session->Evaluate(system.GetConformerCoordinates(0), request);
    ASSERT_EQ(validate_evaluation(evaluation, request, 3), "");

    // Charges add up to the neutral total, with the oxygen negative and the hydrogens equal.
    EXPECT_NEAR(sum_of(evaluation.charges), 0.0, 1e-6);
    EXPECT_NEAR(evaluation.charges[0], -0.563507, 1e-5);
    EXPECT_NEAR(evaluation.charges[1], evaluation.charges[2], 1e-6);

    // 0.893520 e Bohr is 2.2711 D, along -z from the oxygen toward the hydrogens. The measured
    // gas-phase dipole of water is 1.855 D, so the value is of the right size, and an
    // unconverted e Bohr value would be off by the factor 2.54.
    const std::vector<double>& dipole = evaluation.dipole;
    EXPECT_NEAR(std::hypot(dipole[0], dipole[1], dipole[2]), 2.271102, 1e-3);
    EXPECT_LT(dipole[2], 0.0);
    EXPECT_NEAR(dipole[0], 0.0, 1e-6);
    EXPECT_NEAR(dipole[1], 0.0, 1e-6);

    // Two O-H single bonds, each a little below 1; the H-H order (4e-4) is under the cutoff.
    ASSERT_EQ(evaluation.bond_orders.size(), 2u);
    for (std::size_t k = 0; k < 2; ++k) {
        EXPECT_EQ(evaluation.bond_orders[k].i, 0u);
        EXPECT_EQ(evaluation.bond_orders[k].j, k + 1);
        EXPECT_NEAR(evaluation.bond_orders[k].order, 0.9202, 1e-3);
    }

    // Six valence orbitals hold eight electrons, so the HOMO is orbital 3.
    ASSERT_EQ(evaluation.orbital_energies.size(), 6u);
    ASSERT_EQ(evaluation.homo_index, 3);
    const double homo = evaluation.orbital_energies[3];
    const double lumo = evaluation.orbital_energies[4];
    EXPECT_NEAR(homo, -12.1769, 1e-3);
    EXPECT_NEAR(lumo, 2.4528, 1e-3);
    EXPECT_LT(homo, lumo);
}

TEST_F(TbliteEngineTest, PropertiesArePresentOnlyWhenRequested) {
    const OEChem::OEGraphMol h2o = water();
    const MolecularSystem system(h2o, 0, 1);
    auto session = Tblite().CreateSession(system, settings_for(GFN2_XTB));
    for (const Property property :
         {Property::CHARGES, Property::DIPOLE, Property::ORBITAL_ENERGIES, Property::BOND_ORDERS}) {
        EvaluationRequest request;
        request.properties = static_cast<unsigned>(property);
        const Evaluation evaluation = session->Evaluate(system.GetConformerCoordinates(0), request);
        // validate_evaluation rejects any property that was returned without being requested.
        EXPECT_EQ(validate_evaluation(evaluation, request, 3), "")
            << property_names(request.properties);
        EXPECT_EQ(evaluation.charges.empty(), property != Property::CHARGES);
        EXPECT_EQ(evaluation.dipole.empty(), property != Property::DIPOLE);
        EXPECT_EQ(evaluation.orbital_energies.empty(), property != Property::ORBITAL_ENERGIES);
        EXPECT_EQ(evaluation.bond_orders.empty(), property != Property::BOND_ORDERS);
    }
}

// An open-shell result writes 2 num_atoms^2 bond orders, which the sanitizer builds would catch
// in a smaller buffer, and its HOMO comes from spin-summed occupations 2, 2, 1.5, 1.5, 0.
TEST_F(TbliteEngineTest, ChargedAndOpenShellPropertiesFollowTheElectrons) {
    const OEChem::OEGraphMol oh = hydroxyl();
    const EvaluationRequest request = all_properties();

    const MolecularSystem anion(oh, -1, 1);
    const Evaluation hydroxide = Tblite()
                                     .CreateSession(anion, settings_for(GFN2_XTB))
                                     ->Evaluate(anion.GetConformerCoordinates(0), request);
    ASSERT_EQ(validate_evaluation(hydroxide, request, 2), "");
    EXPECT_NEAR(sum_of(hydroxide.charges), -1.0, 1e-6);
    EXPECT_EQ(hydroxide.homo_index, 3);

    const MolecularSystem radical(oh, 0, 2);
    const Evaluation hydroxyl_radical = Tblite()
                                            .CreateSession(radical, settings_for(GFN2_XTB))
                                            ->Evaluate(radical.GetConformerCoordinates(0), request);
    ASSERT_EQ(validate_evaluation(hydroxyl_radical, request, 2), "");
    EXPECT_NEAR(sum_of(hydroxyl_radical.charges), 0.0, 1e-6);
    ASSERT_EQ(hydroxyl_radical.orbital_energies.size(), 5u);
    ASSERT_EQ(hydroxyl_radical.homo_index, 3);
    EXPECT_NEAR(hydroxyl_radical.orbital_energies[3], -13.5764, 1e-3);
    EXPECT_NEAR(hydroxyl_radical.orbital_energies[4], 2.9723, 1e-3);
    ASSERT_EQ(hydroxyl_radical.bond_orders.size(), 1u);
    EXPECT_NEAR(hydroxyl_radical.bond_orders[0].order, 0.8906, 1e-3);
}

TEST_F(TbliteEngineTest, BondOrdersFollowTheBonds) {
    const OEChem::OEGraphMol etoh = ethanol();
    const MolecularSystem system(etoh, 0, 1);
    EvaluationRequest request;
    request.properties = static_cast<unsigned>(Property::BOND_ORDERS);
    const Evaluation evaluation = Tblite()
                                      .CreateSession(system, settings_for(GFN2_XTB))
                                      ->Evaluate(system.GetConformerCoordinates(0), request);
    // Exactly the eight covalent bonds pass the 0.1 cutoff, each between 0.9 and 1.05.
    std::vector<std::pair<unsigned, unsigned>> pairs;
    for (const BondOrder& order : evaluation.bond_orders) {
        pairs.emplace_back(order.i, order.j);
        EXPECT_GT(order.order, 0.9);
        EXPECT_LT(order.order, 1.05);
    }
    EXPECT_EQ(pairs, (std::vector<std::pair<unsigned, unsigned>>{
                         {0, 1}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {1, 6}, {1, 7}, {2, 8}}));
}
