#include <gtest/gtest.h>

#include <oechem.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "core/Geometry.h"
#include "core/Optimizer.h"
#include "oevibe/engines/OEFFEngine.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

ResolvedOptimizerSettings settings_for(Algorithm algorithm) {
    ResolvedOptimizerSettings settings;
    settings.algorithm = algorithm;
    settings.max_iterations = 500;
    settings.energy_change = preset_values(Preset::STANDARD).energy_change;
    settings.gradient_norm = preset_values(Preset::STANDARD).gradient_norm;
    return settings;
}

}  // namespace

TEST(OptimizerTest, FakeEngineReachesItsReferenceShape) {
    const OEChem::OEMol mol = test::make_conformers("CCCCO", 2);
    ASSERT_GE(mol.NumConfs(), 2u);
    const MolecularSystem system(mol, 0, 1);
    test::FakeEngine engine("optimizer-fake");
    auto session = engine.CreateSession(system, EngineSettings());
    detail::OptimizationProblem problem;
    problem.x0 = system.GetConformerCoordinates(1);
    problem.atomic_numbers = system.GetAtomicNumbers();
    problem.settings = settings_for(Algorithm::ANC_RFO);
    const detail::OptimizationOutcome outcome = detail::run_optimization(*session, problem);
    ASSERT_TRUE(outcome.converged) << outcome.message;
    // At the STANDARD gradient-norm threshold the residual spring energy is a few hundredths.
    EXPECT_LT(outcome.engine_energy, 0.05);
    EXPECT_TRUE(outcome.stats.converged);
    EXPECT_GE(outcome.stats.gradient_evaluations, outcome.stats.iterations);
}

TEST(OptimizerTest, BothAlgorithmsAgreeOnMmff94) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    OEFFEngine engine;
    EngineSettings settings;
    settings.method = Method(MMFF94);
    double energies[2];
    int slot = 0;
    for (Algorithm algorithm : {Algorithm::LBFGS, Algorithm::ANC_RFO}) {
        auto session = engine.CreateSession(system, settings);
        detail::OptimizationProblem problem;
        problem.x0 = test::perturbed(system.GetConformerCoordinates(0), 0.05);
        problem.atomic_numbers = system.GetAtomicNumbers();
        problem.settings = settings_for(algorithm);
        const detail::OptimizationOutcome outcome = detail::run_optimization(*session, problem);
        ASSERT_TRUE(outcome.converged) << outcome.message;
        energies[slot++] = outcome.engine_energy;
    }
    // Both stop within the STANDARD thresholds of the same minimum, not at it exactly.
    EXPECT_NEAR(energies[0], energies[1], 0.05);
}

TEST(OptimizerTest, RestraintEnergyIsReportedAndExcluded) {
    const OEChem::OEMol mol = test::make_conformers("CCCCO", 1);
    const MolecularSystem system(mol, 0, 1);
    test::FakeEngine engine("optimizer-restraint");
    auto session = engine.CreateSession(system, EngineSettings());
    const std::vector<double>& x0 = system.GetConformerCoordinates(0);
    detail::OptimizationProblem problem;
    problem.x0 = x0;
    problem.atomic_numbers = system.GetAtomicNumbers();
    problem.settings = settings_for(Algorithm::ANC_RFO);
    detail::ResolvedRestraint stretch;
    stretch.kind = RestraintKind::DISTANCE;
    stretch.atoms = {0, 1};
    stretch.force_constant = 20.0;
    stretch.target = detail::distance(x0.data(), 0, 1, nullptr) + 0.5;
    problem.constraints.restraints = {stretch};
    const detail::OptimizationOutcome outcome = detail::run_optimization(*session, problem);
    ASSERT_TRUE(outcome.converged) << outcome.message;
    EXPECT_GT(outcome.stats.restraint_energy, 1e-3);
    EXPECT_GT(outcome.engine_energy, 1e-3);
}

TEST(OptimizerTest, FailuresCarryTheLastAcceptedIterate) {
    const OEChem::OEMol mol = test::make_conformers("CCCCO");
    const MolecularSystem system(mol, 0, 1);
    const std::vector<double> reference = system.GetConformerCoordinates(0);
    const std::vector<double> start = test::perturbed(reference, 0.08);
    for (Algorithm algorithm : {Algorithm::LBFGS, Algorithm::ANC_RFO}) {
        SCOPED_TRACE(algorithm_name(algorithm));
        test::FakeEngine engine("optimizer-late-failure");
        engine.mode = test::FakeMode::THROW_RUNTIME;
        engine.healthy_evaluations = 5;
        auto session = engine.CreateSession(system, EngineSettings());
        detail::OptimizationProblem problem;
        problem.x0 = start;
        problem.atomic_numbers = system.GetAtomicNumbers();
        problem.settings = settings_for(algorithm);
        try {
            detail::run_optimization(*session, problem);
            FAIL() << "expected detail::OptimizationFailure";
        } catch (const detail::OptimizationFailure& e) {
            EXPECT_STREQ(e.what(), "fake engine failure");
            ASSERT_TRUE(e.HasPartialOutcome());
            const detail::OptimizationOutcome& partial = e.GetPartialOutcome();
            EXPECT_FALSE(partial.converged);
            EXPECT_FALSE(partial.stats.converged);
            EXPECT_EQ(partial.stats.gradient_evaluations, 5u);
            EXPECT_GE(partial.stats.iterations, 1u);
            EXPECT_NEAR(partial.engine_energy,
                        test::spring_energy(reference, partial.x, engine.force_constant, nullptr),
                        1e-9);
            EXPECT_LT(partial.engine_energy,
                      test::spring_energy(reference, start, engine.force_constant, nullptr));
        }
    }

    // A failure before any evaluation succeeds has nothing to report.
    test::FakeEngine broken("optimizer-immediate-failure");
    broken.mode = test::FakeMode::THROW_RUNTIME;
    auto session = broken.CreateSession(system, EngineSettings());
    detail::OptimizationProblem problem;
    problem.x0 = start;
    problem.atomic_numbers = system.GetAtomicNumbers();
    problem.settings = settings_for(Algorithm::ANC_RFO);
    try {
        detail::run_optimization(*session, problem);
        FAIL() << "expected detail::OptimizationFailure";
    } catch (const detail::OptimizationFailure& e) {
        EXPECT_FALSE(e.HasPartialOutcome());
    }
}

TEST(OptimizerTest, MalformedEvaluationsThrow) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    test::FakeEngine engine("optimizer-malformed");
    engine.mode = test::FakeMode::MALFORMED_GRADIENT;
    auto session = engine.CreateSession(system, EngineSettings());
    detail::OptimizationProblem problem;
    problem.x0 = system.GetConformerCoordinates(0);
    problem.atomic_numbers = system.GetAtomicNumbers();
    problem.settings = settings_for(Algorithm::ANC_RFO);
    try {
        detail::run_optimization(*session, problem);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("gradient"), std::string::npos);
    }
}
