#include <gtest/gtest.h>

#include <oechem.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Hessian.h"
#include "oevibe/Error.h"
#include "oevibe/Preset.h"
#include "oevibe/engines/OEFFEngine.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

/// Forwards to another session and records every request, so tests can count evaluations and
/// see where they were made.
class RecordingSession : public EngineSession {
public:
    explicit RecordingSession(std::shared_ptr<EngineSession> inner) : inner_(std::move(inner)) {}

    Evaluation Evaluate(const std::vector<double>& coords,
                        const EvaluationRequest& request) override {
        points.push_back(coords);
        requests.push_back(request);
        return inner_->Evaluate(coords, request);
    }

    std::vector<std::vector<double>> points;
    std::vector<EvaluationRequest> requests;

private:
    std::shared_ptr<EngineSession> inner_;
};

std::shared_ptr<EngineSession> mmff94_session(const MolecularSystem& system) {
    OEFFEngine engine;
    EngineSettings settings;
    settings.method = Method(MMFF94);
    return engine.CreateSession(system, settings);
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

double preset_step() { return preset_values(Preset::STANDARD).fd_step; }

}  // namespace

TEST(HessianTest, FiniteDifferencesMatchTheMmff94AnalyticHessian) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const std::vector<double>& x = system.GetConformerCoordinates(0);
    const std::vector<double> numeric =
        detail::finite_difference_hessian(*session, x, preset_step());
    const std::vector<double> analytic = detail::compute_hessian(*session, x, true, preset_step());
    const std::size_t dim = x.size();
    ASSERT_EQ(numeric.size(), dim * dim);
    ASSERT_EQ(analytic.size(), dim * dim);
    // Central differences err by about h^2/6 times the fourth energy derivative. MMFF94's stiffest
    // term here is the O-H stretch, 143.9325 kb/2 dr^2 (1 + cs dr + 7/12 cs^2 dr^2) with
    // kb = 7.788 mdyn/Angstrom and cs = -2/Angstrom, whose fourth derivative 7 * 143.9325 kb cs^2
    // is 3.14e4 kcal/mol/Angstrom^4, so h = 0.005 Angstrom gives at most 0.13 kcal/mol/Angstrom^2.
    // A probe against step 1e-4 measured 0.058 for this molecule.
    for (std::size_t k = 0; k < dim * dim; ++k) {
        EXPECT_NEAR(numeric[k], analytic[k], 0.3) << "entry " << k;
    }
}

TEST(HessianTest, FiniteDifferenceHessianIsExactlySymmetric) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const std::vector<double>& x = system.GetConformerCoordinates(0);
    const std::vector<double> hessian =
        detail::finite_difference_hessian(*session, x, preset_step());
    const std::size_t dim = x.size();
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            EXPECT_EQ(hessian[i * dim + j], hessian[j * dim + i]) << i << ", " << j;
        }
    }
}

TEST(HessianTest, UsesSixNGradientEvaluationsOnePerDisplacement) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    test::FakeEngine engine("hessian-count");
    RecordingSession session(engine.CreateSession(system, EngineSettings()));
    const std::vector<double>& x = system.GetConformerCoordinates(0);
    const double step = preset_step();
    detail::finite_difference_hessian(session, x, step);

    ASSERT_EQ(session.points.size(), 2 * x.size());
    std::set<std::pair<std::size_t, int>> displacements;
    for (std::size_t e = 0; e < session.points.size(); ++e) {
        EXPECT_TRUE(session.requests[e].gradient);
        EXPECT_FALSE(session.requests[e].hessian);
        EXPECT_EQ(session.requests[e].properties, 0u);
        std::size_t moved = x.size();
        unsigned changed = 0;
        for (std::size_t k = 0; k < x.size(); ++k) {
            if (session.points[e][k] != x[k]) {
                moved = k;
                ++changed;
            }
        }
        ASSERT_EQ(changed, 1u) << "evaluation " << e;
        const double shift = session.points[e][moved] - x[moved];
        EXPECT_NEAR(std::abs(shift), step, 1e-12);
        displacements.insert({moved, shift > 0.0 ? 1 : -1});
    }
    // Every coordinate is displaced once in each direction.
    EXPECT_EQ(displacements.size(), 2 * x.size());
}

TEST(HessianTest, AnalyticPathAsksTheEngineOnce) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    auto inner = mmff94_session(system);
    RecordingSession session(inner);
    const std::vector<double>& x = system.GetConformerCoordinates(0);

    const std::vector<double> analytic = detail::compute_hessian(session, x, true, preset_step());
    ASSERT_EQ(session.requests.size(), 1u);
    EXPECT_TRUE(session.requests[0].hessian);
    EXPECT_FALSE(session.requests[0].gradient);
    EvaluationRequest request;
    request.hessian = true;
    EXPECT_EQ(analytic, inner->Evaluate(x, request).hessian);

    const std::vector<double> numeric = detail::compute_hessian(session, x, false, preset_step());
    EXPECT_EQ(session.requests.size(), 1u + 2 * x.size());
    EXPECT_EQ(numeric, detail::finite_difference_hessian(*inner, x, preset_step()));
}

TEST(HessianTest, MalformedEvaluationsNameTheField) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    const std::vector<double>& x = system.GetConformerCoordinates(0);

    test::FakeEngine malformed("hessian-malformed");
    malformed.mode = test::FakeMode::MALFORMED_GRADIENT;
    auto short_gradient = malformed.CreateSession(system, EngineSettings());
    try {
        detail::finite_difference_hessian(*short_gradient, x, preset_step());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(contains(e.what(), "gradient")) << e.what();
    }

    // The fake engine never returns a Hessian, so an analytic request comes back without one.
    test::FakeEngine no_hessian("hessian-absent");
    auto session = no_hessian.CreateSession(system, EngineSettings());
    try {
        detail::compute_hessian(*session, x, true, preset_step());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(contains(e.what(), "hessian")) << e.what();
    }
}

TEST(HessianTest, EngineExceptionsPropagate) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    test::FakeEngine engine("hessian-throws");
    engine.mode = test::FakeMode::THROW_RUNTIME;
    auto session = engine.CreateSession(system, EngineSettings());
    try {
        detail::finite_difference_hessian(*session, system.GetConformerCoordinates(0),
                                          preset_step());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), "fake engine failure");
    }
}

TEST(HessianTest, InvalidStepOrGeometryIsRejected) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    test::FakeEngine engine("hessian-invalid");
    auto session = engine.CreateSession(system, EngineSettings());
    const std::vector<double>& x = system.GetConformerCoordinates(0);
    EXPECT_THROW(detail::finite_difference_hessian(*session, x, 0.0), OEVibeError);
    EXPECT_THROW(detail::finite_difference_hessian(*session, x, -0.005), OEVibeError);
    EXPECT_THROW(detail::finite_difference_hessian(*session, x, std::nan("")), OEVibeError);
    EXPECT_THROW(detail::compute_hessian(*session, {0.0, 0.0}, false, 0.005), OEVibeError);
    EXPECT_THROW(detail::compute_hessian(*session, {}, true, 0.005), OEVibeError);
}
