#include <gtest/gtest.h>

#include <oechem.h>
#include <oemmff.h>

#include <cmath>
#include <vector>

#include "oevibe/Error.h"
#include "oevibe/OEAdapters.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

EngineSettings settings_for(const char* method) {
    EngineSettings settings;
    settings.method = Method(method);
    return settings;
}

std::vector<double> gradient_at(EngineSession& session, std::vector<double> coords) {
    EvaluationRequest request;
    request.gradient = true;
    return session.Evaluate(coords, request).gradient;
}

}  // namespace

TEST(MolFuncEngineTest, DeclaresOneGasPhaseMethod) {
    const MolFuncEngine engine("molfunc-declare", "1.0", Method("my-mmff"),
                               OEMolMMFF::OEMMFF(OEMolMMFF::OEMMFF94Params()));
    EXPECT_EQ(engine.GetName(), "molfunc-declare");
    EXPECT_TRUE(engine.IsAvailable());
    const std::vector<MethodCapabilities> methods = engine.GetMethods();
    ASSERT_EQ(methods.size(), 1u);
    EXPECT_EQ(methods[0].method.GetName(), "my-mmff");
    EXPECT_TRUE(methods[0].solvent_models.empty());
    EXPECT_EQ(methods[0].properties, 0u);
    EXPECT_TRUE(methods[0].analytic_hessian);
    EXPECT_TRUE(methods[0].charge_from_topology);
    EXPECT_TRUE(methods[0].force_field);
    EXPECT_FALSE(methods[0].requires_partial_charges);
}

TEST(MolFuncEngineTest, EnergyMatchesDirectOpenEyeEvaluation) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    MolFuncEngine engine("molfunc-energy", "1.0", Method("my-mmff"),
                         OEMolMMFF::OEMMFF(OEMolMMFF::OEMMFF94Params()));
    auto session = engine.CreateSession(system, settings_for("my-mmff"));
    const Evaluation evaluation =
        session->Evaluate(system.GetConformerCoordinates(0), EvaluationRequest());

    OEMolMMFF::OEMMFF direct{OEMolMMFF::OEMMFF94Params()};
    OEChem::OEMol copy(mol);
    ASSERT_TRUE(direct.PrepMol(copy));
    ASSERT_TRUE(direct.Setup(copy));
    std::vector<double> coords(3 * copy.GetMaxAtomIdx());
    copy.GetCoords(coords.data());
    EXPECT_NEAR(evaluation.energy, direct(coords.data()), 1e-8);
}

TEST(MolFuncEngineTest, GradientMatchesFiniteDifferences) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    MolFuncEngine engine("molfunc-gradient", "1.0", Method("my-mmff"),
                         OEMolMMFF::OEMMFF(OEMolMMFF::OEMMFF94Params()));
    auto session = engine.CreateSession(system, settings_for("my-mmff"));
    std::vector<double> x = system.GetConformerCoordinates(0);
    const std::vector<double> analytic = gradient_at(*session, x);
    const double step = 1e-5;
    for (std::size_t k = 0; k < 6; ++k) {
        std::vector<double> plus = x;
        std::vector<double> minus = x;
        plus[k] += step;
        minus[k] -= step;
        const double numeric = (session->Evaluate(plus, EvaluationRequest()).energy -
                                session->Evaluate(minus, EvaluationRequest()).energy) /
                               (2.0 * step);
        EXPECT_NEAR(analytic[k], numeric, 1e-4 * std::max(1.0, std::abs(numeric)));
    }
}

TEST(MolFuncEngineTest, AnalyticHessianMatchesFiniteDifferences) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    MolFuncEngine engine("molfunc-hessian", "1.0", Method("my-mmff"),
                         OEMolMMFF::OEMMFF(OEMolMMFF::OEMMFF94Params()));
    auto session = engine.CreateSession(system, settings_for("my-mmff"));
    std::vector<double> x = system.GetConformerCoordinates(0);
    EvaluationRequest request;
    request.hessian = true;
    const Evaluation evaluation = session->Evaluate(x, request);
    const std::size_t dim = x.size();
    ASSERT_EQ(evaluation.hessian.size(), dim * dim);
    const double step = 1e-4;
    for (std::size_t k = 0; k < 6; ++k) {
        std::vector<double> plus = x;
        std::vector<double> minus = x;
        plus[k] += step;
        minus[k] -= step;
        const std::vector<double> g_plus = gradient_at(*session, plus);
        const std::vector<double> g_minus = gradient_at(*session, minus);
        for (std::size_t j = 0; j < dim; ++j) {
            const double numeric = (g_plus[j] - g_minus[j]) / (2.0 * step);
            EXPECT_NEAR(evaluation.hessian[k * dim + j], numeric, 1e-2 + 1e-3 * std::abs(numeric));
        }
    }
}

TEST(MolFuncEngineTest, FailedSetupIsUnsupported) {
    OEChem::OEGraphMol xenon;
    xenon.NewAtom(54);
    const double origin[3] = {0.0, 0.0, 0.0};
    xenon.SetCoords(origin);
    xenon.SetDimension(3);
    const MolecularSystem system(xenon, 0, 1);
    MolFuncEngine engine("molfunc-xenon", "1.0", Method("my-mmff"),
                         OEMolMMFF::OEMMFF(OEMolMMFF::OEMMFF94Params()));
    EXPECT_THROW(engine.CreateSession(system, settings_for("my-mmff")), UnsupportedError);
}
