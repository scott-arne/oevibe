#include <gtest/gtest.h>

#include <oechem.h>
#include <sys/wait.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "engines/XtbLibrary.h"
#include "oevibe/Calculation.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "oevibe/Preset.h"
#include "oevibe/engines/TbliteEngine.h"
#include "oevibe/engines/XtbEngine.h"
#include "test_runtime_libraries.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions options_for(const char* method, const Solvent& solvent = Solvent()) {
    CalculationOptions options;
    options.SetMethod(Method(method));
    options.SetSolvent(solvent);
    return options;
}

/// Conformer 0 evaluated through an engine built here rather than the registry's built-in, with
/// the settings a STANDARD calculation hands its engine.
Evaluation direct_evaluation(Engine& engine, const OEChem::OEMolBase& mol, const char* method,
                             const Solvent& solvent, const EvaluationRequest& request) {
    const MolecularSystem system(mol, 0, 1);
    EngineSettings settings;
    settings.method = Method(method);
    settings.solvent = solvent;
    settings.accuracy = preset_values(Preset::STANDARD).engine_accuracy;
    settings.properties = request.properties;
    return engine.CreateSession(system, settings)
        ->Evaluate(system.GetConformerCoordinates(0), request);
}

bool has_warning(const ConformerResult& result, const std::string& text) {
    const std::vector<std::string> warnings = result.GetWarnings();
    return std::any_of(warnings.begin(), warnings.end(), [&text](const std::string& warning) {
        return warning.find(text) != std::string::npos;
    });
}

/// Energy, Gradient, and Optimize of ethanol through the registry's tblite engine, checked
/// against a directly built TbliteEngine. ``concrete`` is the solvent the direct session gets:
/// the model AUTO must resolve to.
void expect_tblite_tasks(const std::string& library, const char* method, const Solvent& requested,
                         const Solvent& concrete) {
    // GetTag() rejects a named solvent whose model is still AUTO, so the trace uses the name.
    SCOPED_TRACE(std::string(method) + " in " +
                 (requested.IsGasPhase() ? "gas" : requested.GetName()));
    const OEChem::OEMol mol = test::make_conformers("CCO");
    TbliteEngine direct(library);
    EvaluationRequest request;
    request.gradient = true;
    const Evaluation reference = direct_evaluation(direct, mol, method, concrete, request);
    Calculation calculation(options_for(method, requested));
    ASSERT_EQ(calculation.GetEngineName(), "tblite");

    OEChem::OEMol energy_input(mol);
    const ResultSet energies = calculation.Energy(energy_input);
    const ConformerResult& energy = energies.Get(0);
    ASSERT_EQ(energy.GetStatus(), ConformerStatus::OK) << energy.GetMessage();
    EXPECT_NEAR(energy.GetEnergy(), reference.energy, 1e-6);
    const Provenance& provenance = energy.GetProvenance();
    EXPECT_EQ(provenance.engine, "tblite");
    EXPECT_EQ(provenance.method, method);
    EXPECT_FALSE(provenance.engine_version.empty());
    EXPECT_EQ(provenance.engine_version, direct.GetVersion());

    OEChem::OEMol gradient_input(mol);
    const ResultSet gradients = calculation.Gradient(gradient_input);
    ASSERT_EQ(gradients.Get(0).GetStatus(), ConformerStatus::OK) << gradients.Get(0).GetMessage();
    const std::vector<double> gradient = gradients.Get(0).GetGradient();
    const std::vector<double> expected = MolecularSystem(mol, 0, 1).ToOELayout(reference.gradient);
    ASSERT_EQ(gradient.size(), expected.size());
    for (std::size_t k = 0; k < gradient.size(); ++k) {
        EXPECT_NEAR(gradient[k], expected[k], 1e-5) << "component " << k;
    }

    OEChem::OEMol optimized(mol);
    const ResultSet optimizations = calculation.Optimize(optimized, OptimizerOptions());
    const ConformerResult& result = optimizations.Get(0);
    ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
    EXPECT_TRUE(result.GetOptimizationStats().converged);
    EXPECT_LE(result.GetEnergy(), reference.energy + 1e-6);
    EXPECT_EQ(result.GetProvenance().resolved_options.at("optimizer.algorithm"), "anc_rfo");
}

/// Exit code of ``gfn0_after_an_early_custom_method_lookup`` when the failure it needs cannot be
/// made, because the system loader finds libxtb by its bare name.
constexpr int LOADER_FINDS_XTB = 77;

/// Resolves a custom method before the default xtb path is set, as a C++ caller can, and then
/// runs GFN0. Resolving asks every registered engine for its methods, the built-in xtb among
/// them, which loads its library from the default path, and here that path fails.
/// :returns: 0 when GFN0 then works, ``LOADER_FINDS_XTB`` when no early failure could be made,
///     and 1 with the reason on stderr otherwise.
int gfn0_after_an_early_custom_method_lookup(const std::string& library) {
    try {
        const test::ScopedEnvironment variable("OEVIBE_XTB_LIBRARY", nullptr);
        const test::ScopedEnvironment conda("CONDA_PREFIX", nullptr);
        const test::ScopedPythonLibraryDirectory python("");
        const test::ScopedDefaultLibraryPath<XtbEngine> default_path(
            "/nonexistent/lib/libxtb.6.dylib");
        EngineRegistry& registry = EngineRegistry::Instance();
        auto custom = std::make_shared<test::FakeEngine>(
            "fake-early-lookup",
            std::vector<MethodCapabilities>{test::fake_capabilities("early-lookup")});
        registry.Register(custom, true);
        if (registry.ResolveForMethod(Method("early-lookup")) != custom) {
            std::cerr << "the custom method did not resolve to its engine\n";
            return 1;
        }
        const std::shared_ptr<Engine> xtb = registry.Find("xtb");
        if (xtb->IsAvailable()) {
            return LOADER_FINDS_XTB;
        }
        {
            // Only a changed path reopens a failed load, so a source that now reaches the
            // library while the path stays the same does not.
            const test::ScopedEnvironment reachable("OEVIBE_XTB_LIBRARY", library.c_str());
            if (xtb->IsAvailable()) {
                std::cerr << "the failed load was retried although the path did not change\n";
                return 1;
            }
        }

        XtbEngine::SetLibraryPath(library);
        CalculationOptions options = options_for(GFN0_XTB);
        Calculation calculation(options);
        OEChem::OEMol input = test::make_conformers("CCO");
        const ResultSet results = calculation.Energy(input);
        if (results.Get(0).GetStatus() != ConformerStatus::OK ||
            results.Get(0).GetProvenance().engine != "xtb") {
            std::cerr << "GFN0 failed: " << results.Get(0).GetMessage() << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "GFN0 after the early lookup threw: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace

// The registry, its engines' load outcomes, and the default library path are process-wide, and
// the tests before this one have used them, so the check runs in a re-executed child process
// that starts from none of that.
TEST(EngineIntegrationTest, AnEarlyCustomMethodLookupDoesNotFreezeGfn0Unavailable) {
    const std::string library = test::xtb_library_path();
    if (library.empty()) {
        GTEST_SKIP() << test::XTB_SKIP_REASON;
    }
    const std::string previous_style = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    int status = -1;
    // The predicate runs in this process after the child exits, so it can keep the status.
    EXPECT_EXIT(
        std::exit(gfn0_after_an_early_custom_method_lookup(library)),
        [&status](int child_status) {
            status = child_status;
            return WIFEXITED(child_status) && (WEXITSTATUS(child_status) == 0 ||
                                               WEXITSTATUS(child_status) == LOADER_FINDS_XTB);
        },
        "");
    GTEST_FLAG_SET(death_test_style, previous_style);
    if (WIFEXITED(status) && WEXITSTATUS(status) == LOADER_FINDS_XTB) {
        GTEST_SKIP() << "the system loader finds a libxtb";
    }
}

TEST(EngineIntegrationTest, Gfn2InTheGasPhaseThroughTblite) {
    const std::string library = test::tblite_library_path();
    if (library.empty()) {
        GTEST_SKIP() << test::TBLITE_SKIP_REASON;
    }
    expect_tblite_tasks(library, GFN2_XTB, Solvent(), Solvent());
}

TEST(EngineIntegrationTest, Gfn2InAlpbWaterThroughTblite) {
    const std::string library = test::tblite_library_path();
    if (library.empty()) {
        GTEST_SKIP() << test::TBLITE_SKIP_REASON;
    }
    expect_tblite_tasks(library, GFN2_XTB, Solvent("water"), Solvent("water", SolventModel::ALPB));
}

// Spec 10 records the solvent model AUTO chose on every result. ALPB water lowers the GFN2
// energy of the spike S4 ethanol by 4.65 kcal/mol (probe 2026-10-11).
TEST(EngineIntegrationTest, AutoWaterResolvesToAlpbAndIsRecorded) {
    if (test::tblite_library_path().empty()) {
        GTEST_SKIP() << test::TBLITE_SKIP_REASON;
    }
    const OEChem::OEMol mol = test::make_conformers("CCO");
    OEChem::OEMol gas_input(mol);
    OEChem::OEMol water_input(mol);
    const ResultSet gas = Calculation(options_for(GFN2_XTB)).Energy(gas_input);
    const ResultSet water =
        Calculation(options_for(GFN2_XTB, Solvent("water"))).Energy(water_input);
    ASSERT_EQ(gas.Get(0).GetStatus(), ConformerStatus::OK) << gas.Get(0).GetMessage();
    ASSERT_EQ(water.Get(0).GetStatus(), ConformerStatus::OK) << water.Get(0).GetMessage();
    EXPECT_LT(water.Get(0).GetEnergy(), gas.Get(0).GetEnergy() - 1.0);
    EXPECT_EQ(water.Get(0).GetProvenance().solvent_model, "alpb");
    EXPECT_EQ(water.Get(0).GetProvenance().solvent, "water");
    EXPECT_TRUE(has_warning(water.Get(0), "resolved to 'alpb'"));
    EXPECT_FALSE(has_warning(gas.Get(0), "resolved to"));
    EXPECT_TRUE(OEChem::OEHasSDData(*water_input.GetActive(), "OEVibe.gfn2-xtb.alpb-water.Energy"));
}

TEST(EngineIntegrationTest, Gfn1EnergyThroughTblite) {
    const std::string library = test::tblite_library_path();
    if (library.empty()) {
        GTEST_SKIP() << test::TBLITE_SKIP_REASON;
    }
    const OEChem::OEMol mol = test::make_conformers("CCO");
    TbliteEngine direct(library);
    const double gfn1 =
        direct_evaluation(direct, mol, GFN1_XTB, Solvent(), EvaluationRequest()).energy;
    const double gfn2 =
        direct_evaluation(direct, mol, GFN2_XTB, Solvent(), EvaluationRequest()).energy;
    OEChem::OEMol input(mol);
    const ResultSet results = Calculation(options_for(GFN1_XTB)).Energy(input);
    const ConformerResult& result = results.Get(0);
    ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
    EXPECT_NEAR(result.GetEnergy(), gfn1, 1e-6);
    EXPECT_EQ(result.GetProvenance().engine, "tblite");
    EXPECT_EQ(result.GetProvenance().method, "gfn1-xtb");
    // The two parametrizations' total energies differ by 0.77 Eh (483 kcal/mol) for the spike
    // S4 ethanol (probe 2026-10-11), so routing GFN1 to a GFN2 calculator cannot pass.
    EXPECT_GT(std::abs(gfn1 - gfn2), 100.0);
}

// Spec 12.1: tblite takes the total charge from the options (it does not declare
// charge_from_topology), so an override that differs from the formal-charge sum is accepted and
// changes the energy; OEFF methods reject the same override (Plan A covers that side).
TEST(EngineIntegrationTest, TbliteAcceptsAChargeOverride) {
    if (test::tblite_library_path().empty()) {
        GTEST_SKIP() << test::TBLITE_SKIP_REASON;
    }
    const OEChem::OEMol mol = test::make_conformers("CCO");
    OEChem::OEMol neutral_input(mol);
    OEChem::OEMol cation_input(mol);
    CalculationOptions cation = options_for(GFN2_XTB);
    cation.SetCharge(1);
    cation.SetMultiplicity(2);
    const ResultSet neutral = Calculation(options_for(GFN2_XTB)).Energy(neutral_input);
    const ResultSet radical_cation = Calculation(cation).Energy(cation_input);
    ASSERT_EQ(neutral.Get(0).GetStatus(), ConformerStatus::OK) << neutral.Get(0).GetMessage();
    ASSERT_EQ(radical_cation.Get(0).GetStatus(), ConformerStatus::OK)
        << radical_cation.Get(0).GetMessage();
    // Removing an electron costs roughly the ionization energy (several eV), far more than
    // 50 kcal/mol, so an ignored override cannot pass.
    EXPECT_GT(radical_cation.Get(0).GetEnergy(), neutral.Get(0).GetEnergy() + 50.0);
    EXPECT_EQ(radical_cation.Get(0).GetProvenance().resolved_options.at("charge"), "1");
}

TEST(EngineIntegrationTest, Gfn0EnergyAndPropertiesThroughXtb) {
    const std::string library = test::xtb_library_path();
    if (library.empty()) {
        GTEST_SKIP() << test::XTB_SKIP_REASON;
    }
    const OEChem::OEMol mol = test::make_conformers("CCO");
    XtbEngine direct(library);
    EvaluationRequest request;
    request.properties = ALL_PROPERTIES;
    const Evaluation reference = direct_evaluation(direct, mol, GFN0_XTB, Solvent(), request);
    CalculationOptions options = options_for(GFN0_XTB);
    options.SetProperties(ALL_PROPERTIES);
    OEChem::OEMol input(mol);
    const ResultSet results = Calculation(options).Energy(input);
    const ConformerResult& result = results.Get(0);
    ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
    EXPECT_NEAR(result.GetEnergy(), reference.energy, 1e-6);
    EXPECT_EQ(result.GetProvenance().engine, "xtb");
    EXPECT_EQ(result.GetProvenance().engine_version, "C API 1.0.0");

    const MolecularSystem system(mol, 0, 1);
    const std::vector<double> charges = result.GetCharges();
    const std::vector<double> expected_charges = system.ToOEScalarLayout(reference.charges);
    ASSERT_EQ(charges.size(), expected_charges.size());
    double total = 0.0;
    for (std::size_t k = 0; k < charges.size(); ++k) {
        EXPECT_NEAR(charges[k], expected_charges[k], 1e-8);
        total += charges[k];
    }
    EXPECT_NEAR(total, 0.0, 1e-6);
    const std::vector<double> dipole = result.GetDipole();
    ASSERT_EQ(dipole.size(), 3u);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(dipole[axis], reference.dipole[axis], 1e-8);
    }
    ASSERT_GE(reference.homo_index, 0);
    ASSERT_TRUE(result.HasHOMO());
    ASSERT_TRUE(result.HasLUMO());
    EXPECT_NEAR(result.GetHOMO(), reference.orbital_energies[reference.homo_index], 1e-8);
    EXPECT_LT(result.GetHOMO(), result.GetLUMO());
    ASSERT_FALSE(reference.bond_orders.empty());
    for (const BondOrder& bond : reference.bond_orders) {
        EXPECT_NEAR(
            result.GetBondOrder(system.GetOEAtomIndex(bond.i), system.GetOEAtomIndex(bond.j)),
            bond.order, 1e-8);
    }
    // Every bond of ethanol is single; GFN0 gives 0.977 to 1.026 at the spike S4 geometry.
    for (OESystem::OEIter<OEChem::OEBondBase> bond = input.GetBonds(); bond; ++bond) {
        EXPECT_GT(result.GetBondOrder(*bond), 0.9);
        EXPECT_LT(result.GetBondOrder(*bond), 1.1);
    }
}

TEST(EngineIntegrationTest, GfnFfCalculationsAreUnavailableWithTheReason) {
    if (test::xtb_library_path().empty()) {
        GTEST_SKIP() << test::XTB_SKIP_REASON;
    }
    try {
        Calculation calculation(options_for(GFN_FF));
        FAIL() << "expected EngineUnavailableError";
    } catch (const EngineUnavailableError& e) {
        EXPECT_NE(std::string(e.what()).find(detail::XTB_GFNFF_UNAVAILABLE_REASON),
                  std::string::npos)
            << e.what();
    }
}

TEST(EngineIntegrationTest, Gfn0RejectsASolventAtConstruction) {
    if (test::xtb_library_path().empty()) {
        GTEST_SKIP() << test::XTB_SKIP_REASON;
    }
    EXPECT_THROW(Calculation calculation(options_for(GFN0_XTB, Solvent("water"))), OptionsError);
}

// Two Calculations on two threads (one per thread, spec 5.6) share the registry's one tblite
// engine. Its process-wide mutex serializes the libtblite calls, and the tsan preset checks
// everything around them.
TEST(EngineIntegrationTest, CalculationsOnTwoThreadsRunTbliteConcurrently) {
    if (test::tblite_library_path().empty()) {
        GTEST_SKIP() << test::TBLITE_SKIP_REASON;
    }
    constexpr std::size_t ROUNDS = 3;
    const OEChem::OEMol mol = test::make_conformers("CCO");
    OEChem::OEMol reference_input(mol);
    const ResultSet reference = Calculation(options_for(GFN2_XTB)).Energy(reference_input);
    ASSERT_EQ(reference.Get(0).GetStatus(), ConformerStatus::OK) << reference.Get(0).GetMessage();
    const double expected = reference.Get(0).GetEnergy();

    struct Job {
        std::vector<OEChem::OEMol> inputs;  ///< Copied up front: each calculation tags its input.
        std::vector<double> energies;
        std::string error;
    };
    std::vector<Job> jobs(2);
    for (Job& job : jobs) {
        job.inputs.assign(ROUNDS, mol);
    }
    std::vector<std::thread> workers;
    for (Job& job : jobs) {
        workers.emplace_back([&job] {
            try {
                Calculation calculation(options_for(GFN2_XTB));
                for (OEChem::OEMol& input : job.inputs) {
                    const ResultSet results = calculation.Energy(input);
                    if (results.Get(0).GetStatus() != ConformerStatus::OK) {
                        job.error = results.Get(0).GetMessage();
                        return;
                    }
                    job.energies.push_back(results.Get(0).GetEnergy());
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
        ASSERT_EQ(job.energies.size(), ROUNDS);
        for (double energy : job.energies) {
            EXPECT_NEAR(energy, expected, 1e-6);
        }
    }
}
