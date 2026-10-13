#include <gtest/gtest.h>

#include <oechem.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/Frequencies.h"
#include "core/WriteBack.h"
#include "oevibe/Calculation.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "oevibe/engines/OEFFEngine.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions mmff94() {
    CalculationOptions options;
    options.SetMethod(Method(MMFF94));
    return options;
}

CalculationOptions fake_options(const std::string& engine) {
    CalculationOptions options;
    options.SetMethod(Method("fake-method"));
    options.SetEngine(engine);
    return options;
}

ThermoOptions without_optimization() {
    ThermoOptions options;
    options.SetOptimizeFirst(false);
    return options;
}

std::vector<double> coordinates_of(const OEChem::OEMolBase& conf) {
    std::vector<double> coords(3 * conf.GetMaxAtomIdx());
    conf.GetCoords(coords.data());
    return coords;
}

void jitter(OEChem::OEMol& mol) {
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf) {
        conf->SetCoords(test::perturbed(coordinates_of(*conf), 0.08).data());
    }
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

bool any_warning_contains(const ConformerResult& result, const std::string& part) {
    for (const std::string& warning : result.GetWarnings()) {
        if (contains(warning, part)) {
            return true;
        }
    }
    return false;
}

/// :returns: The resolved-option keys starting with ``prefix``, sorted, so that a test can pin a
///     whole group of provenance keys. The keys are hashed into the options digest, so a group
///     that gains or loses one is a digest change.
std::vector<std::string> keys_with_prefix(const std::map<std::string, std::string>& resolved,
                                          const std::string& prefix) {
    std::vector<std::string> keys;
    for (const auto& [key, value] : resolved) {
        if (key.rfind(prefix, 0) == 0) {
            keys.push_back(key);
        }
    }
    return keys;
}

/// Re-exposes the OpenEye force fields without their analytic Hessian, so the
/// finite-difference path runs on the same potential as the analytic one.
class NoHessianEngine : public Engine {
public:
    std::string GetName() const override { return "oeff-no-hessian"; }
    std::string GetVersion() const override { return inner_.GetVersion(); }
    bool IsAvailable() const override { return true; }
    std::string GetUnavailableReason() const override { return ""; }
    std::vector<MethodCapabilities> GetMethods() const override {
        std::vector<MethodCapabilities> methods = inner_.GetMethods();
        for (MethodCapabilities& caps : methods) {
            caps.analytic_hessian = false;
        }
        return methods;
    }
    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override {
        return inner_.CreateSession(system, settings);
    }

private:
    OEFFEngine inner_;
};

}  // namespace

TEST(CalculationFrequenciesTest, Mmff94GivesModesAndThermochemistry) {
    OEChem::OEMol mol = test::make_conformers("CCO", 2);
    jitter(mol);
    const unsigned num_atoms = mol.NumAtoms();
    const ResultSet results = Calculation(mmff94()).Frequencies(mol, ThermoOptions());
    ASSERT_EQ(results.Size(), mol.NumConfs());
    unsigned position = 0;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf, ++position) {
        const ConformerResult& result = results.Get(position);
        ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
        EXPECT_TRUE(result.GetOptimizationStats().converged);
        EXPECT_EQ(result.GetHessian().size(), 9u * num_atoms * num_atoms);
        const std::vector<double> frequencies = result.GetFrequencies();
        ASSERT_EQ(frequencies.size(), 3u * num_atoms - 6);
        EXPECT_EQ(result.GetNormalModes().size(), frequencies.size() * 3 * num_atoms);
        // The optimized geometry is a minimum, so every mode is real.
        EXPECT_EQ(result.GetImaginaryModeCount(), 0u);
        EXPECT_GT(frequencies.front(), 0.0);
        EXPECT_FALSE(any_warning_contains(result, "imaginary"));

        const Thermochemistry& thermo = result.GetThermochemistry();
        EXPECT_DOUBLE_EQ(thermo.temperature, 298.15);
        EXPECT_DOUBLE_EQ(thermo.pressure, 1.0);
        EXPECT_DOUBLE_EQ(thermo.electronic_energy, result.GetEnergy());
        EXPECT_EQ(thermo.frequencies, frequencies);
        EXPECT_NEAR(thermo.total_mass, 2 * 12.0107 + 6 * 1.00794 + 15.9994, 1e-9);
        EXPECT_FALSE(thermo.linear);
        EXPECT_EQ(thermo.symmetry_number, 1u);
        EXPECT_EQ(thermo.multiplicity, 1u);
        EXPECT_GT(thermo.zpe, 0.0);
        EXPECT_GT(thermo.enthalpy, thermo.electronic_energy + thermo.zpe);
        EXPECT_LT(thermo.free_energy, thermo.enthalpy);
        EXPECT_DOUBLE_EQ(thermo.free_energy_correction,
                         thermo.free_energy - thermo.electronic_energy);

        // Optimize-first writes the optimized geometry back; OpenEye stores single precision.
        const std::vector<double> written = coordinates_of(*conf);
        const std::vector<double> optimized = result.GetCoordinates();
        ASSERT_EQ(written.size(), optimized.size());
        for (std::size_t k = 0; k < written.size(); ++k) {
            EXPECT_NEAR(written[k], optimized[k], 1e-4);
        }
    }
    const std::map<std::string, std::string>& resolved =
        results.Get(0).GetProvenance().resolved_options;
    EXPECT_EQ(resolved.at("task"), "frequencies");
    EXPECT_EQ(resolved.at("hessian"), "analytic");
    EXPECT_EQ(resolved.count("hessian.step"), 0u);
    EXPECT_EQ(resolved.at("thermo.temperature"), "298.150000");
    EXPECT_EQ(resolved.at("thermo.pressure"), "1.000000");
    EXPECT_EQ(resolved.at("thermo.quasi_rrho_cutoff"), "100.000000");
    EXPECT_EQ(resolved.at("thermo.symmetry_number"), "1");
    EXPECT_EQ(resolved.at("thermo.optimize_first"), "true");
    // Spec 7.5: the six optimizer keys of a task that optimizes, and no more. ThermoOptions
    // forbids constraints, so the three constraint keys Optimize and Scan also write would be
    // permanently empty here, and adding them would move the frequency digest.
    EXPECT_EQ(keys_with_prefix(resolved, "optimizer."),
              (std::vector<std::string>{"optimizer.algorithm", "optimizer.energy_change",
                                        "optimizer.gradient_norm", "optimizer.initial_trust_radius",
                                        "optimizer.max_iterations", "optimizer.max_trust_radius"}));
    EXPECT_EQ(resolved.at("optimizer.algorithm"), "lbfgs");
    EXPECT_EQ(resolved.at("optimizer.max_iterations"), "200");
    EXPECT_EQ(resolved.at("optimizer.energy_change"), "0.003138");
    EXPECT_EQ(resolved.at("optimizer.gradient_norm"), "1.185821");
    EXPECT_EQ(resolved.at("optimizer.initial_trust_radius"), "0.100000");
    EXPECT_EQ(resolved.at("optimizer.max_trust_radius"), "0.300000");
}

TEST(CalculationFrequenciesTest, ThermoSettingsReachTheResultsAndProvenance) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    ThermoOptions options = without_optimization();
    options.SetTemperature(350.0);
    options.SetPressure(2.0);
    options.SetQuasiRRHOCutoff(50.0);
    options.SetSymmetryNumber(3);
    const ResultSet results = Calculation(mmff94()).Frequencies(mol, options);
    const ConformerResult& result = results.Get(0);
    ASSERT_TRUE(result.HasThermochemistry()) << result.GetMessage();
    const Thermochemistry& thermo = result.GetThermochemistry();
    EXPECT_DOUBLE_EQ(thermo.temperature, 350.0);
    EXPECT_DOUBLE_EQ(thermo.pressure, 2.0);
    EXPECT_DOUBLE_EQ(thermo.quasi_rrho_cutoff, 50.0);
    EXPECT_EQ(thermo.symmetry_number, 3u);
    const std::map<std::string, std::string>& resolved = result.GetProvenance().resolved_options;
    EXPECT_EQ(resolved.at("thermo.temperature"), "350.000000");
    EXPECT_EQ(resolved.at("thermo.pressure"), "2.000000");
    EXPECT_EQ(resolved.at("thermo.quasi_rrho_cutoff"), "50.000000");
    EXPECT_EQ(resolved.at("thermo.symmetry_number"), "3");
    EXPECT_EQ(resolved.at("thermo.optimize_first"), "false");
    // Without an optimization the optimizer settings cannot change the result, so none of them
    // is recorded.
    EXPECT_EQ(keys_with_prefix(resolved, "optimizer."), std::vector<std::string>());
}

TEST(CalculationFrequenciesTest, WithoutOptimizationTheInputGeometryIsAnalyzed) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    OEChem::OEMol reference(mol);
    const ResultSet single = Calculation(mmff94()).Energy(reference);
    const ResultSet results = Calculation(mmff94()).Frequencies(mol, without_optimization());
    const ConformerResult& result = results.Get(0);
    ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
    EXPECT_EQ(coordinates_of(*mol.GetActive()), before);
    EXPECT_FALSE(result.HasCoordinates());
    EXPECT_FALSE(result.HasOptimizationStats());
    EXPECT_DOUBLE_EQ(result.GetEnergy(), single.Get(0).GetEnergy());
    EXPECT_EQ(result.GetFrequencies().size(), 3u * mol.NumAtoms() - 6);
}

// Eclipsed ethane, held there by a dihedral constraint, is the saddle point of methyl rotation:
// exactly one imaginary mode, the torsion.
TEST(CalculationFrequenciesTest, ASaddlePointHasOneImaginaryModeAndAWarning) {
    OEChem::OEMol mol = test::make_conformers("CC");
    std::vector<unsigned> carbons;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(OEChem::OEIsHeavy()); atom;
         ++atom) {
        carbons.push_back(atom->GetIdx());
    }
    ASSERT_EQ(carbons.size(), 2u);
    const auto hydrogen_on = [&mol](unsigned carbon) {
        const OEChem::OEAtomBase* atom = mol.GetAtom(OEChem::OEHasAtomIdx(carbon));
        for (OESystem::OEIter<OEChem::OEAtomBase> neighbor = atom->GetAtoms(); neighbor;
             ++neighbor) {
            if (neighbor->GetAtomicNum() == 1) {
                return neighbor->GetIdx();
            }
        }
        return 0u;
    };
    ConstraintSet eclipsed;
    eclipsed.AddDihedral(hydrogen_on(carbons[0]), carbons[0], carbons[1], hydrogen_on(carbons[1]),
                         0.0);
    OptimizerOptions constrained;
    constrained.SetConstraints(eclipsed);
    const ResultSet saddle = Calculation(mmff94()).Optimize(mol, constrained);
    ASSERT_EQ(saddle.Get(0).GetStatus(), ConformerStatus::OK) << saddle.Get(0).GetMessage();

    const ResultSet results = Calculation(mmff94()).Frequencies(mol, without_optimization());
    const ConformerResult& result = results.Get(0);
    ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
    EXPECT_EQ(result.GetImaginaryModeCount(), 1u);
    const std::vector<double> frequencies = result.GetFrequencies();
    EXPECT_LT(frequencies[0], 0.0);
    EXPECT_GT(frequencies[1], 0.0);
    EXPECT_TRUE(any_warning_contains(result, "1 imaginary mode")) << result.GetWarnings().size();
    EXPECT_EQ(result.GetThermochemistry().frequencies, frequencies);
}

TEST(CalculationFrequenciesTest, OptimizationThatDoesNotConvergeStopsTheLaterStages) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    OptimizerOptions one_step;
    one_step.SetMaxIterations(1);
    ThermoOptions options;
    options.SetOptimizerOptions(one_step);
    const ResultSet results = Calculation(mmff94()).Frequencies(mol, options);
    const ConformerResult& result = results.Get(0);
    EXPECT_EQ(result.GetStatus(), ConformerStatus::NOT_CONVERGED);
    EXPECT_EQ(result.GetMessage().rfind("optimization stage: ", 0), 0u) << result.GetMessage();
    EXPECT_TRUE(result.HasEnergy());
    EXPECT_TRUE(result.HasCoordinates());
    EXPECT_FALSE(result.GetOptimizationStats().converged);
    EXPECT_FALSE(result.HasHessian());
    EXPECT_FALSE(result.HasFrequencies());
    EXPECT_FALSE(result.HasThermochemistry());
    EXPECT_TRUE(any_warning_contains(result, "skipped: Hessian, analysis"));
    // NOT_CONVERGED conformers still receive their last geometry (spec 6.5).
    EXPECT_NE(coordinates_of(*mol.GetActive()), before);
    const std::string prefix = "OEVibe.mmff94.gas.";
    EXPECT_EQ(OEChem::OEGetSDData(*mol.GetActive(), prefix + "Status"), "NOT_CONVERGED");
    EXPECT_FALSE(OEChem::OEHasSDData(*mol.GetActive(), prefix + "ZPE"));
}

TEST(CalculationFrequenciesTest, TagsCarryTheThermochemistry) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    const ResultSet results = Calculation(mmff94()).Frequencies(mol, ThermoOptions());
    const Thermochemistry& thermo = results.Get(0).GetThermochemistry();
    const OEChem::OEConfBase& conf = *mol.GetActive();
    const std::string prefix = "OEVibe.mmff94.gas.";
    EXPECT_EQ(OEChem::OEGetSDData(conf, prefix + "ZPE"), detail::format_value(thermo.zpe));
    EXPECT_EQ(OEChem::OEGetSDData(conf, prefix + "Enthalpy"),
              detail::format_value(thermo.enthalpy));
    EXPECT_EQ(OEChem::OEGetSDData(conf, prefix + "FreeEnergy"),
              detail::format_value(thermo.free_energy));
    EXPECT_EQ(OEChem::OEGetSDData(conf, prefix + "Status"), "OK");

    // A later call without thermochemistry removes the tags, so they never describe another run.
    Calculation(mmff94()).Energy(mol);
    EXPECT_FALSE(OEChem::OEHasSDData(*mol.GetActive(), prefix + "ZPE"));
    EXPECT_FALSE(OEChem::OEHasSDData(*mol.GetActive(), prefix + "Enthalpy"));
    EXPECT_FALSE(OEChem::OEHasSDData(*mol.GetActive(), prefix + "FreeEnergy"));
}

TEST(CalculationFrequenciesTest, ConstraintsAreRejectedBeforeAnythingRuns) {
    auto engine = test::register_fake("freq-options");
    OEChem::OEMol mol = test::make_conformers("CCO");
    ConstraintSet frozen;
    frozen.FreezeAtomIndices({0});
    OptimizerOptions optimizer;
    optimizer.SetConstraints(frozen);
    ThermoOptions options;
    options.SetOptimizerOptions(optimizer);
    EXPECT_THROW(Calculation(fake_options("freq-options")).Frequencies(mol, options), OptionsError);
    OptimizerOptions radii;
    radii.SetInitialTrustRadius(0.5);
    radii.SetMaxTrustRadius(0.2);
    options.SetOptimizerOptions(radii);
    EXPECT_THROW(Calculation(fake_options("freq-options")).Frequencies(mol, options), OptionsError);
    EXPECT_EQ(engine->sessions_created, 0);
    EXPECT_FALSE(OEChem::OEHasSDData(*mol.GetActive(), "OEVibe.fake-method.gas.Status"));
}

TEST(CalculationFrequenciesTest, AnalyticAndFiniteDifferenceHessiansAgree) {
    EngineRegistry::Instance().Register(std::make_shared<NoHessianEngine>(), true);
    OEChem::OEMol mol = test::make_conformers("CCO");
    OEChem::OEMol copy(mol);
    CalculationOptions numeric = mmff94();
    numeric.SetEngine("oeff-no-hessian");
    const ResultSet analytic = Calculation(mmff94()).Frequencies(mol, without_optimization());
    const ResultSet finite = Calculation(numeric).Frequencies(copy, without_optimization());
    ASSERT_EQ(analytic.Get(0).GetStatus(), ConformerStatus::OK) << analytic.Get(0).GetMessage();
    ASSERT_EQ(finite.Get(0).GetStatus(), ConformerStatus::OK) << finite.Get(0).GetMessage();
    const std::vector<double> a = analytic.Get(0).GetFrequencies();
    const std::vector<double> f = finite.Get(0).GetFrequencies();
    ASSERT_EQ(a.size(), f.size());
    // Compare curvatures, lambda = (nu / FREQUENCY_FACTOR)^2 in kcal/mol/Angstrom^2/amu: central
    // differences at the preset step move each Hessian entry by at most about 0.13
    // kcal/mol/Angstrom^2 (Task 7), so after weighting by hydrogen's 1.008 amu an eigenvalue
    // moves by a small multiple of that. A probe measured at most 0.03 for this molecule (0.55
    // cm^-1 on the 299 cm^-1 methyl torsion, where a fixed wavenumber tolerance would be unfair).
    const auto curvature = [](double nu) {
        return std::copysign(nu * nu, nu) / (detail::FREQUENCY_FACTOR * detail::FREQUENCY_FACTOR);
    };
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(curvature(f[i]), curvature(a[i]), 0.15) << "mode " << i;
    }
    const auto& analytic_options = analytic.Get(0).GetProvenance().resolved_options;
    const auto& finite_options = finite.Get(0).GetProvenance().resolved_options;
    EXPECT_EQ(analytic_options.at("hessian"), "analytic");
    EXPECT_EQ(finite_options.at("hessian"), "finite_difference");
    EXPECT_EQ(finite_options.at("hessian.step"), "0.005000");
    EXPECT_NE(analytic.Get(0).GetProvenance().options_digest,
              finite.Get(0).GetProvenance().options_digest);
}

TEST(CalculationFrequenciesTest, OneSessionPerConformerThroughEveryStage) {
    auto engine = test::register_fake("freq-sessions");
    OEChem::OEMol mol = test::make_conformers("CCCCO", 2);
    ASSERT_GE(mol.NumConfs(), 2u);
    const ResultSet results =
        Calculation(fake_options("freq-sessions")).Frequencies(mol, ThermoOptions());
    EXPECT_EQ(engine->sessions_created, static_cast<int>(mol.NumConfs()));
    for (std::size_t i = 0; i < results.Size(); ++i) {
        const ConformerResult& result = results.Get(i);
        ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
        // The spring network is rigid, so every vibration of its minimum is real.
        EXPECT_EQ(result.GetImaginaryModeCount(), 0u);
        EXPECT_EQ(result.GetFrequencies().size(), 3u * mol.NumAtoms() - 6);
    }
    const auto& resolved = results.Get(0).GetProvenance().resolved_options;
    EXPECT_EQ(resolved.at("hessian"), "finite_difference");
    EXPECT_EQ(resolved.at("optimizer.algorithm"), "anc_rfo");
}

TEST(CalculationFrequenciesTest, EngineFailuresNameTheStage) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    auto failing = test::register_fake("freq-failing");
    failing->mode = test::FakeMode::THROW_RUNTIME;
    failing->healthy_evaluations = 1;  // the energy at the input geometry, then the Hessian fails
    const ResultSet broken =
        Calculation(fake_options("freq-failing")).Frequencies(mol, without_optimization());
    const ConformerResult& result = broken.Get(0);
    EXPECT_EQ(result.GetStatus(), ConformerStatus::ENGINE_ERROR);
    EXPECT_EQ(result.GetMessage(), "Hessian stage: fake engine failure");
    EXPECT_TRUE(result.HasEnergy());
    EXPECT_FALSE(result.HasHessian());
    EXPECT_FALSE(result.HasFrequencies());
    EXPECT_TRUE(any_warning_contains(result, "skipped: analysis"));

    auto unsupported = test::register_fake("freq-unsupported");
    unsupported->mode = test::FakeMode::THROW_UNSUPPORTED;
    const ResultSet refused =
        Calculation(fake_options("freq-unsupported")).Frequencies(mol, ThermoOptions());
    const ConformerResult& refusal = refused.Get(0);
    EXPECT_EQ(refusal.GetStatus(), ConformerStatus::UNSUPPORTED);
    EXPECT_EQ(refusal.GetMessage(), "optimization stage: fake engine cannot handle this system");
    EXPECT_FALSE(refusal.HasEnergy());
    EXPECT_FALSE(refusal.HasCoordinates());
}

// Spec 7.5: an optimization stage whose first evaluation is malformed has computed that single
// point, so the conformer keeps its finite energy, exactly as Optimize does (Task 1).
TEST(CalculationFrequenciesTest, AMalformedFirstEvaluationKeepsItsEnergy) {
    auto engine = test::register_fake("freq-malformed-start");
    engine->mode = test::FakeMode::MALFORMED_GRADIENT;
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    ASSERT_GE(mol.NumConfs(), 2u);
    OEChem::OEMol copy(mol);
    const ResultSet gradients = Calculation(fake_options("freq-malformed-start")).Gradient(copy);
    const ResultSet results =
        Calculation(fake_options("freq-malformed-start")).Frequencies(mol, ThermoOptions());
    ASSERT_EQ(results.Size(), gradients.Size());
    for (std::size_t i = 0; i < results.Size(); ++i) {
        SCOPED_TRACE(i);
        const ConformerResult& result = results.Get(i);
        EXPECT_EQ(result.GetStatus(), ConformerStatus::ENGINE_ERROR);
        EXPECT_EQ(result.GetMessage().rfind("optimization stage: ", 0), 0u) << result.GetMessage();
        EXPECT_TRUE(contains(result.GetMessage(), "malformed Evaluation")) << result.GetMessage();
        ASSERT_TRUE(gradients.Get(i).HasEnergy());
        ASSERT_TRUE(result.HasEnergy());
        EXPECT_DOUBLE_EQ(result.GetEnergy(), gradients.Get(i).GetEnergy());
        EXPECT_FALSE(result.HasCoordinates());
        EXPECT_FALSE(result.HasHessian());
    }
    // Conformer 0 is the fake engine's reference, so only a later conformer shows that the kept
    // energy is the computed one rather than a default zero.
    EXPECT_GT(results.Get(1).GetEnergy(), 0.0);
}

// Spec 7.5: the frequency-stage rule governs over the single-stage one, so an UNSUPPORTED later
// stage keeps what the completed stages computed. Optimize, a single-stage task, keeps nothing
// from the same engine refusal (test_calculation_optimize.cpp).
TEST(CalculationFrequenciesTest, UnsupportedPropertiesKeepTheOptimizationStageValues) {
    const std::shared_ptr<test::FakeEngine> engine =
        test::register_fake("freq-unsupported-properties");
    engine->mode = test::FakeMode::UNSUPPORTED_PROPERTIES;
    CalculationOptions options = fake_options("freq-unsupported-properties");
    options.SetProperties(static_cast<unsigned>(Property::DIPOLE));
    OEChem::OEMol mol = test::make_conformers("CCCCO");
    const ResultSet results = Calculation(options).Frequencies(mol, ThermoOptions());
    const ConformerResult& result = results.Get(0);
    EXPECT_EQ(result.GetStatus(), ConformerStatus::UNSUPPORTED);
    EXPECT_EQ(result.GetMessage().rfind("Hessian stage: ", 0), 0u) << result.GetMessage();
    EXPECT_TRUE(contains(result.GetMessage(), "cannot compute properties")) << result.GetMessage();
    // The optimization stage completed, so everything it computed survives the refusal.
    EXPECT_TRUE(result.HasEnergy());
    EXPECT_TRUE(result.HasCoordinates());
    EXPECT_TRUE(result.HasOptimizationStats());
    // The refusing stage and the stages after it contribute nothing.
    EXPECT_FALSE(result.HasDipole());
    EXPECT_FALSE(result.HasHessian());
    EXPECT_FALSE(result.HasFrequencies());
    EXPECT_FALSE(result.HasThermochemistry());
    EXPECT_TRUE(any_warning_contains(result, "skipped: analysis"));
}

// Spec 7.2: the Hessian stage evaluates the requested properties at the geometry it analyzes,
// which after an optimization is the optimized one, not the input.
TEST(CalculationFrequenciesTest, RequestedPropertiesComeFromTheAnalyzedGeometry) {
    test::register_fake("freq-properties");
    CalculationOptions options = fake_options("freq-properties");
    options.SetProperties(static_cast<unsigned>(Property::DIPOLE) |
                          static_cast<unsigned>(Property::CHARGES));
    OEChem::OEMol mol = test::make_conformers("CCCCO", 2);
    ASSERT_GE(mol.NumConfs(), 2u);
    std::vector<std::vector<double>> before;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf) {
        before.push_back(coordinates_of(*conf));
    }
    const ResultSet results = Calculation(options).Frequencies(mol, ThermoOptions());
    ASSERT_EQ(results.Size(), mol.NumConfs());
    // The fake engine's springs hold conformer 0's geometry, so only a later conformer moves far
    // enough for the charges to tell the optimized geometry from the input one.
    const ConformerResult& result = results.Get(1);
    ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
    EXPECT_TRUE(result.HasFrequencies());
    EXPECT_TRUE(result.HasThermochemistry());
    ASSERT_TRUE(result.HasDipole());
    EXPECT_EQ(result.GetDipole(), (std::vector<double>{0.0, 0.0, 1.0}));
    // The fake engine's charge on an atom is 0.001 times its x coordinate.
    ASSERT_TRUE(result.HasCharges());
    const std::vector<double> charges = result.GetCharges();
    const std::vector<double> optimized = result.GetCoordinates();
    ASSERT_EQ(charges.size(), static_cast<std::size_t>(mol.NumAtoms()));
    ASSERT_EQ(optimized.size(), 3 * charges.size());
    EXPECT_NE(optimized, before[1]);
    for (std::size_t i = 0; i < charges.size(); ++i) {
        EXPECT_DOUBLE_EQ(charges[i], 0.001 * optimized[3 * i]) << "atom " << i;
    }
}
