#include <gtest/gtest.h>

#include <oechem.h>
#include <oeff.h>
#include <oemmff.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "oevibe/Calculation.h"
#include "oevibe/Error.h"
#include "oevibe/OEAdapters.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions mmff_options() {
    CalculationOptions options;
    options.SetMethod(Method("mmff94s"));
    return options;
}

CalculationOptions fake_options(const std::string& engine) {
    CalculationOptions options;
    options.SetEngine(engine);
    options.SetMethod(Method("fake-method"));
    return options;
}

std::vector<double> coordinates_of(const OEChem::OEMolBase& mol) {
    std::vector<double> coords(3 * mol.GetMaxAtomIdx(), 0.0);
    mol.GetCoords(coords.data());
    return coords;
}

}  // namespace

TEST(EngineMolFuncTest, NumVarCoversTheOpenEyeLayoutIncludingGaps) {
    OEChem::OEMol mol = test::make_gapped_methane();
    EngineMolFunc function(fake_options(test::register_fake("molfunc-gap")->GetName()));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    EXPECT_EQ(function.NumVar(), 3 * mol.GetMaxAtomIdx());
    // The gap is real, and the private copy compacts it away: the map must come from the caller.
    EXPECT_EQ(mol.NumAtoms() + 1, mol.GetMaxAtomIdx());
}

TEST(EngineMolFuncTest, EnergyMatchesACalculationWithTheSameOptions) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(mmff_options());
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const std::vector<double> x = coordinates_of(mol);
    const double adapter_energy = function(x.data());

    OEChem::OEMol copy(mol);
    const ResultSet results = Calculation(mmff_options()).Energy(copy);
    EXPECT_NEAR(adapter_energy, results.Get(0).GetEnergy(), 1e-8);
}

TEST(EngineMolFuncTest, GradientAccumulatesAndLeavesGapsAlone) {
    OEChem::OEMol mol = test::make_gapped_methane();
    EngineMolFunc function(fake_options(test::register_fake("molfunc-grad")->GetName()));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    std::vector<double> x = test::perturbed(coordinates_of(mol), 0.05);

    std::vector<double> clean(function.NumVar(), 0.0);
    function(x.data(), clean.data());
    std::vector<double> prefilled(function.NumVar(), 7.0);
    function(x.data(), prefilled.data());

    for (unsigned k = 0; k < function.NumVar(); ++k) {
        EXPECT_NEAR(prefilled[k], 7.0 + clean[k], 1e-9) << "variable " << k;
    }
    // Atom index 1 was deleted, so its three variables carry no force.
    for (unsigned axis = 0; axis < 3; ++axis) {
        EXPECT_DOUBLE_EQ(clean[3 * 1 + axis], 0.0);
    }
}

TEST(EngineMolFuncTest, HessianFillsThePackedLowerTriangleAndMatchesFiniteDifferences) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(mmff_options());
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const unsigned nvar = function.NumVar();
    std::vector<double> x = test::perturbed(coordinates_of(mol), 0.02);

    std::vector<double> packed(static_cast<std::size_t>(nvar) * (nvar + 1) / 2, 0.0);
    std::vector<double> gradient(nvar, 0.0);
    ASSERT_TRUE(function(x.data(), packed.data(), gradient.data()));

    const double step = 1e-4;
    for (unsigned row = 0; row < nvar; row += 7) {  // a sample keeps the test quick
        for (unsigned col = 0; col <= row; col += 5) {
            std::vector<double> plus = x;
            std::vector<double> minus = x;
            plus[col] += step;
            minus[col] -= step;
            std::vector<double> gp(nvar, 0.0);
            std::vector<double> gm(nvar, 0.0);
            function(plus.data(), gp.data());
            function(minus.data(), gm.data());
            const double expected = (gp[row] - gm[row]) / (2.0 * step);
            EXPECT_NEAR(packed[static_cast<std::size_t>(row) * (row + 1) / 2 + col], expected, 5e-2)
                << "element (" << row << ", " << col << ")";
        }
    }
}

TEST(EngineMolFuncTest, GetFComponentsIsNull) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(mmff_options());
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const std::vector<double> x = coordinates_of(mol);
    EXPECT_EQ(function.GetFComponents(x.data()), nullptr);
}

TEST(EngineMolFuncTest, CreateCopyReturnsAnUnconfiguredFunctionOfTheSameOptions) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(mmff_options());
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    std::unique_ptr<EngineMolFunc> copy(function.CreateCopy());
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->NumVar(), 0u);
    ASSERT_TRUE(copy->Setup(mol)) << copy->GetSetupError();
    const std::vector<double> x = coordinates_of(mol);
    EXPECT_NEAR((*copy)(x.data()), function(x.data()), 1e-10);
}

TEST(EngineMolFuncTest, SetupFailsWhenPreparationWouldChangeTheAtoms) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));  // implicit hydrogens, no coordinates
    EngineMolFunc function(mmff_options());
    EXPECT_FALSE(function.Setup(mol));
    EXPECT_FALSE(function.GetSetupError().empty());
    EXPECT_EQ(function.NumVar(), 0u);
}

TEST(EngineMolFuncTest, SetupFailsWhenTheEngineRejectsTheSystem) {
    auto fake = test::register_fake("molfunc-unsupported");
    fake->mode = test::FakeMode::THROW_UNSUPPORTED;
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(fake_options("molfunc-unsupported"));
    EXPECT_FALSE(function.Setup(mol));
    EXPECT_NE(function.GetSetupError().find("fake engine cannot handle this system"),
              std::string::npos);
}

TEST(EngineMolFuncTest, DrivesOEBFGSOptToTheSameMinimumAsCalculationOptimize) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(mmff_options());
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    std::vector<double> x = coordinates_of(mol);
    std::vector<double> out(function.NumVar(), 0.0);
    const double adapter_minimum = OEOpt::OEBFGSOpt()(function, x.data(), out.data());

    OEChem::OEMol copy(mol);
    const ResultSet results = Calculation(mmff_options()).Optimize(copy, OptimizerOptions());
    EXPECT_NEAR(adapter_minimum, results.Get(0).GetEnergy(), 1e-2);
}

TEST(EngineMolFuncTest, WorksAsAnOEGenericFF2Component) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(mmff_options());
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    OEMolPotential::OEGenericFF2 composite;
    ASSERT_TRUE(composite.AddMolFunc(function));
    ASSERT_TRUE(composite.Setup(mol));
    const std::vector<double> x = coordinates_of(mol);
    EXPECT_NEAR(composite(x.data()), function(x.data()), 1e-8);
}

TEST(EngineMolFuncTest, EngineErrorPropagatesOutOfTheOptimizer) {
    auto fake = test::register_fake("molfunc-error");
    fake->mode = test::FakeMode::THROW_RUNTIME;
    fake->healthy_evaluations = 1;
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(fake_options("molfunc-error"));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    // Away from the fake's reference geometry, where the energy and the gradient are exactly
    // zero and the optimizer would converge on the one healthy evaluation without ever reaching
    // the failing one.
    std::vector<double> x = test::perturbed(coordinates_of(mol), 0.1);
    std::vector<double> out(function.NumVar(), 0.0);
    EXPECT_THROW(OEOpt::OEBFGSOpt()(function, x.data(), out.data()), std::runtime_error);
}

TEST(EngineMolFuncTest, EvaluateAssignsTheSameValuesOperatorAccumulates) {
    OEChem::OEMol mol = test::make_gapped_methane();
    EngineMolFunc function(fake_options(test::register_fake("molfunc-evaluate")->GetName()));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const std::vector<double> x = test::perturbed(coordinates_of(mol), 0.05);

    std::vector<double> accumulated(function.NumVar(), 0.0);
    const double energy = function(x.data(), accumulated.data());
    const EngineMolFunc::FunctionValue value = function.Evaluate(x, true);

    EXPECT_DOUBLE_EQ(value.energy, energy);
    ASSERT_EQ(value.gradient.size(), accumulated.size());
    for (std::size_t k = 0; k < accumulated.size(); ++k) {
        EXPECT_NEAR(value.gradient[k], accumulated[k], 1e-12) << "variable " << k;
    }
    EXPECT_TRUE(function.Evaluate(x, false).gradient.empty());
    EXPECT_THROW(function.Evaluate(std::vector<double>(3, 0.0), false), OEVibeError);
}

TEST(EngineMolFuncTest, RepeatedCallAtTheSameGeometryReusesTheCachedEnergy) {
    // A session that counts, because the engine's sessions_created would read 1 whether the
    // second call hit the cache or re-evaluated.
    struct CountingSession : public EngineSession {
        int evaluations = 0;
        Evaluation Evaluate(const std::vector<double>& coords,
                            const EvaluationRequest& request) override {
            ++evaluations;
            Evaluation evaluation;
            evaluation.energy = 1.0;
            if (request.gradient) {
                evaluation.gradient.assign(coords.size(), 0.5);
            }
            return evaluation;
        }
    };
    struct CountingEngine : public test::FakeEngine {
        using test::FakeEngine::FakeEngine;
        std::shared_ptr<CountingSession> session = std::make_shared<CountingSession>();
        std::shared_ptr<EngineSession> CreateSession(const MolecularSystem&,
                                                     const EngineSettings&) override {
            return session;
        }
    };
    auto engine = std::make_shared<CountingEngine>("molfunc-cache");
    EngineRegistry::Instance().Register(engine, true);

    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(fake_options("molfunc-cache"));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const std::vector<double> x = coordinates_of(mol);

    EXPECT_DOUBLE_EQ(function(x.data()), function(x.data()));
    EXPECT_EQ(engine->session->evaluations, 1) << "the second energy call must hit the cache";

    // An energy-only cache entry cannot answer a gradient request.
    std::vector<double> gradient(function.NumVar(), 0.0);
    function(x.data(), gradient.data());
    EXPECT_EQ(engine->session->evaluations, 2);
    std::fill(gradient.begin(), gradient.end(), 0.0);
    function(x.data(), gradient.data());
    EXPECT_EQ(engine->session->evaluations, 2) << "the gradient must come from the cache";

    const std::vector<double> moved = test::perturbed(x, 0.01);
    function(moved.data());
    EXPECT_EQ(engine->session->evaluations, 3);
}
