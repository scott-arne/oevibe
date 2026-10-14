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

TEST(EngineMolFuncTest, GradientValuesOnAGappedMoleculeFollowTheCallerIndexMap) {
    OEChem::OEMol mol = test::make_gapped_methane();
    auto fake = test::register_fake("molfunc-map");
    EngineMolFunc function(fake_options(fake->GetName()));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const std::vector<double> reference = coordinates_of(mol);

    // The caller's atom indices in iteration order. Dense position k is the k-th of these, so
    // the expected values below are built without asking the adapter what its map is.
    std::vector<unsigned> caller_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        caller_indices.push_back(atom->GetIdx());
    }
    ASSERT_EQ(caller_indices.size(), 5u);

    // The fake engine's spring reference is the prepared copy's conformer 0, which is this
    // geometry gathered: at it the energy and every force are exactly zero. A gather that read
    // dense position k from variable 3 * k would feed the engine the gap instead of an atom,
    // and neither would be zero.
    std::vector<double> at_reference(function.NumVar(), 0.0);
    EXPECT_DOUBLE_EQ(function(reference.data(), at_reference.data()), 0.0);
    for (unsigned k = 0; k < function.NumVar(); ++k) {
        EXPECT_DOUBLE_EQ(at_reference[k], 0.0) << "variable " << k;
    }

    const std::vector<double> x = test::perturbed(reference, 0.05);
    std::vector<double> dense_reference;
    std::vector<double> dense_x;
    for (unsigned index : caller_indices) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            dense_reference.push_back(reference[3 * index + axis]);
            dense_x.push_back(x[3 * index + axis]);
        }
    }
    std::vector<double> expected_dense;
    const double expected_energy =
        test::spring_energy(dense_reference, dense_x, fake->force_constant, &expected_dense);

    std::vector<double> gradient(function.NumVar(), 0.0);
    EXPECT_DOUBLE_EQ(function(x.data(), gradient.data()), expected_energy);
    for (std::size_t k = 0; k < caller_indices.size(); ++k) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            EXPECT_DOUBLE_EQ(gradient[3 * caller_indices[k] + axis], expected_dense[3 * k + axis])
                << "atom " << caller_indices[k] << " axis " << axis;
        }
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

TEST(EngineMolFuncTest, HessianOnAGappedMoleculeLeavesTheGapRowsAndColumnsAlone) {
    OEChem::OEMol mol = test::make_gapped_methane();
    EngineMolFunc function(fake_options(test::register_fake("molfunc-gap-hessian")->GetName()));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();
    const unsigned nvar = function.NumVar();
    const std::vector<double> x = test::perturbed(coordinates_of(mol), 0.05);

    // A nonzero fill, so an untouched element is distinguishable from one written as zero and
    // every written element has to be read back as the fill plus the value.
    const double fill = 11.0;
    std::vector<double> packed(static_cast<std::size_t>(nvar) * (nvar + 1) / 2, fill);
    std::vector<double> gradient(nvar, 0.0);
    ASSERT_TRUE(function(x.data(), packed.data(), gradient.data())) << function.GetSetupError();

    // Atom index 1 was deleted, so nothing may be written in its rows or in its columns.
    const unsigned gap_atom = 1;
    for (unsigned row = 0; row < nvar; ++row) {
        for (unsigned col = 0; col <= row; ++col) {
            if (row / 3 != gap_atom && col / 3 != gap_atom) {
                continue;
            }
            EXPECT_DOUBLE_EQ(packed[static_cast<std::size_t>(row) * (row + 1) / 2 + col], fill)
                << "element (" << row << ", " << col << ")";
        }
    }

    // The elements that were written are the right ones: each still matches the finite
    // difference of the adapter's own gradient, which the caller index map also produced.
    const double step = 1e-4;
    for (unsigned row = 0; row < nvar; ++row) {
        if (row / 3 == gap_atom) {
            continue;
        }
        for (unsigned col = 0; col <= row; col += 4) {
            if (col / 3 == gap_atom) {
                continue;
            }
            std::vector<double> plus = x;
            std::vector<double> minus = x;
            plus[col] += step;
            minus[col] -= step;
            std::vector<double> gp(nvar, 0.0);
            std::vector<double> gm(nvar, 0.0);
            function(plus.data(), gp.data());
            function(minus.data(), gm.data());
            const double expected = (gp[row] - gm[row]) / (2.0 * step);
            EXPECT_NEAR(packed[static_cast<std::size_t>(row) * (row + 1) / 2 + col] - fill,
                        expected, 1e-2)
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

TEST(EngineMolFuncTest, CreateCopyKeepsTheEngineTheOriginalResolved) {
    auto original = test::register_fake("molfunc-replaced");
    original->force_constant = 50.0;
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(fake_options("molfunc-replaced"));
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();

    // The registry is process-wide and mutable, so another engine can take the name after the
    // function was built. A copy that re-resolved would compute with this stiffer spring.
    auto replacement = test::register_fake("molfunc-replaced");
    replacement->force_constant = 500.0;

    std::unique_ptr<EngineMolFunc> copy(function.CreateCopy());
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->NumVar(), 0u);
    ASSERT_TRUE(copy->Setup(mol)) << copy->GetSetupError();
    const std::vector<double> x = test::perturbed(coordinates_of(mol), 0.05);
    EXPECT_DOUBLE_EQ((*copy)(x.data()), function(x.data()));
    EXPECT_EQ(replacement->sessions_created, 0) << "the copy must not use the replacement engine";
}

TEST(EngineMolFuncTest, CreateCopyDoesNotResolveTheMethodAgainstTheRegistryAgain) {
    const std::vector<MethodCapabilities> methods = {test::fake_capabilities("molfunc-one-method")};
    auto provider = std::make_shared<test::FakeEngine>("molfunc-sole-provider", methods);
    EngineRegistry::Instance().Register(provider, true);
    // No engine named, so the registry resolves the method to its only provider.
    CalculationOptions options;
    options.SetMethod(Method("molfunc-one-method"));
    OEChem::OEMol mol = test::make_conformers("CCO");
    EngineMolFunc function(options);
    ASSERT_TRUE(function.Setup(mol)) << function.GetSetupError();

    auto competitor = std::make_shared<test::FakeEngine>("molfunc-rival-provider", methods);
    EngineRegistry::Instance().Register(competitor, true);

    // Resolving the method now would be ambiguous and throw, which an OpenEye caller asking for
    // a copy has no way to handle; the copy reuses what the original resolved.
    std::unique_ptr<EngineMolFunc> copy(function.CreateCopy());
    ASSERT_NE(copy, nullptr);
    ASSERT_TRUE(copy->Setup(mol)) << copy->GetSetupError();
    EXPECT_EQ(competitor->sessions_created, 0) << "the copy must not use the competing provider";
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
