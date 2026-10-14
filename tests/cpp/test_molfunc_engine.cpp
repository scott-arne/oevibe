#include <gtest/gtest.h>

#include <oechem.h>
#include <oemmff.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "oevibe/Error.h"
#include "oevibe/OEAdapters.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

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

/// Minimal potential for adapter edge cases. Like OpenEye force fields it keeps a pointer to the
/// molecule it was set up on, so a test can check that molecule is still alive when the potential
/// is destroyed.
class StubPotential : public OEMolPotential::OEMolFunc1 {
public:
    struct Probe {
        /// ``CreateCopy`` returns null once this reaches zero; some OpenEye potentials always do.
        int copies_allowed = 1000;
        unsigned atoms_at_destruction = 0;
    };

    explicit StubPotential(Probe& probe) : probe_(&probe) {}

    ~StubPotential() override {
        if (molecule_ != nullptr) {
            probe_->atoms_at_destruction = molecule_->NumAtoms();
        }
    }

    bool Setup(const OEChem::OEMolBase& mol) override {
        molecule_ = &mol;
        return true;
    }

    unsigned NumVar() const override {
        return molecule_ == nullptr ? 0u : 3u * molecule_->NumAtoms();
    }

    double operator()(const double*) override { return 0.0; }
    double operator()(const double*, double*) override { return 0.0; }

    OESystem::OEIterBase<OEOpt::OEFComponent>* GetFComponents(const double*) override {
        return nullptr;
    }

    OEMolPotential::OEMolFunc1* CreateCopy() const override {
        if (probe_->copies_allowed <= 0) {
            return nullptr;
        }
        --probe_->copies_allowed;
        return new StubPotential(*probe_);
    }

private:
    Probe* probe_;
    const OEChem::OEMolBase* molecule_ = nullptr;
};

/// A gradient-only potential with a real energy surface: ``E = 1/2 sum_i k_i x_i^2`` in
/// kcal/mol over the OpenEye coordinates in Angstrom. It supplies the ``OEMolFunc1`` half of
/// the spec 12.1 round trip, which ``StubPotential``'s zero energy cannot. OpenEye's own
/// gradient-only potential, ``OEHarmonicPotential``, is no substitute: it evaluates flat after
/// its ``CreateCopy()``, which is the copy every session gets.
class QuadraticPotential : public OEMolPotential::OEMolFunc1 {
public:
    /// :returns: The force constant for variable ``k``, in kcal/mol/Angstrom^2. It differs per
    ///     variable so that a transposed or shifted index mapping cannot cancel out.
    static double StiffnessAt(std::size_t k) { return 1.0 + static_cast<double>(k); }

    static double EnergyAt(const std::vector<double>& x) {
        double energy = 0.0;
        for (std::size_t k = 0; k < x.size(); ++k) {
            energy += 0.5 * StiffnessAt(k) * x[k] * x[k];
        }
        return energy;
    }

    bool Setup(const OEChem::OEMolBase& mol) override {
        num_var_ = 3u * mol.GetMaxAtomIdx();
        return true;
    }

    unsigned NumVar() const override { return num_var_; }

    double operator()(const double* x) override {
        double energy = 0.0;
        for (std::size_t k = 0; k < num_var_; ++k) {
            energy += 0.5 * StiffnessAt(k) * x[k] * x[k];
        }
        return energy;
    }

    double operator()(const double* x, double* g) override {
        double energy = 0.0;
        for (std::size_t k = 0; k < num_var_; ++k) {
            energy += 0.5 * StiffnessAt(k) * x[k] * x[k];
            // OpenEye sums the components of a composite potential into one buffer, so its
            // contract is that a potential adds to the gradient it is handed. Conforming to it
            // here is a matter of being a faithful stand-in, not something the tests below
            // discriminate: MolFuncSession hands over a freshly zeroed buffer
            // (src/core/MolFuncSession.cpp), so an overwriting potential would pass identically.
            g[k] += StiffnessAt(k) * x[k];
        }
        return energy;
    }

    OESystem::OEIterBase<OEOpt::OEFComponent>* GetFComponents(const double*) override {
        return nullptr;
    }

    OEMolPotential::OEMolFunc1* CreateCopy() const override {
        return new QuadraticPotential(*this);
    }

private:
    unsigned num_var_ = 0;
};

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
    for (std::size_t k = 0; k < dim; ++k) {
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

TEST(MolFuncEngineTest, GradientOnlyPrototypeDeclaresNoAnalyticHessian) {
    const MolFuncEngine engine("molfunc-quadratic-declare", "1.0", Method("my-quadratic"),
                               QuadraticPotential());
    const std::vector<MethodCapabilities> methods = engine.GetMethods();
    ASSERT_EQ(methods.size(), 1u);
    EXPECT_TRUE(methods[0].analytic_gradient);
    // An OEMolFunc1 has no second-derivative call, so the capability has to say so; the engine
    // reads that off the prototype's dynamic type, not off a flag the caller passes.
    EXPECT_FALSE(methods[0].analytic_hessian);
}

TEST(MolFuncEngineTest, GradientOnlyPrototypeRoundTripsEnergyAndGradient) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    MolFuncEngine engine("molfunc-quadratic", "1.0", Method("my-quadratic"), QuadraticPotential());
    auto session = engine.CreateSession(system, settings_for("my-quadratic"));
    const std::vector<double> x = system.GetConformerCoordinates(0);

    const double energy = session->Evaluate(x, EvaluationRequest()).energy;
    EXPECT_NEAR(energy, QuadraticPotential::EnergyAt(x), 1e-9 * std::abs(energy));

    // The gradient the session reports comes from OEMolFunc1::operator()(x, g), which
    // accumulates into a buffer the session owns; finite differences of the energy are what
    // pin that it was neither dropped nor double-counted.
    const std::vector<double> analytic = gradient_at(*session, x);
    ASSERT_EQ(analytic.size(), x.size());
    // Evaluating twice pins the other half of that arrangement: every call starts from a
    // zeroed buffer. One carried over between calls would hand this accumulating potential a
    // non-zero gradient, and the second result would come back doubled.
    EXPECT_EQ(gradient_at(*session, x), analytic);
    const double step = 1e-5;
    for (std::size_t k = 0; k < x.size(); ++k) {
        std::vector<double> plus = x;
        std::vector<double> minus = x;
        plus[k] += step;
        minus[k] -= step;
        const double numeric = (session->Evaluate(plus, EvaluationRequest()).energy -
                                session->Evaluate(minus, EvaluationRequest()).energy) /
                               (2.0 * step);
        EXPECT_NEAR(analytic[k], numeric, 1e-6 * std::max(1.0, std::abs(numeric))) << k;
    }

    // The capability and the session agree: there is no Hessian to ask for. Every OEVibe error
    // derives from std::runtime_error (include/oevibe/Error.h), so only the text separates this
    // from an unsupported system or a Hessian evaluation that ran and failed.
    EvaluationRequest request;
    request.hessian = true;
    try {
        session->Evaluate(x, request);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(contains(e.what(), "no analytic Hessian")) << e.what();
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

TEST(MolFuncEngineTest, UncopyablePrototypeIsRejected) {
    StubPotential::Probe probe;
    probe.copies_allowed = 0;
    EXPECT_THROW(
        MolFuncEngine("molfunc-uncopyable", "1.0", Method("my-mmff"), StubPotential(probe)),
        OptionsError);
}

TEST(MolFuncEngineTest, UncopyableSessionPotentialIsUnsupported) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    StubPotential::Probe probe;
    // The constructor takes the only copy, so the session's own copy comes back null.
    probe.copies_allowed = 1;
    MolFuncEngine engine("molfunc-one-copy", "1.0", Method("my-mmff"), StubPotential(probe));
    EXPECT_THROW(engine.CreateSession(system, settings_for("my-mmff")), UnsupportedError);
}

TEST(MolFuncEngineTest, SessionDestroysPotentialBeforeItsMolecule) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    StubPotential::Probe probe;
    {
        MolFuncEngine engine("molfunc-lifetime", "1.0", Method("my-mmff"), StubPotential(probe));
        auto session = engine.CreateSession(system, settings_for("my-mmff"));
    }
    // The potential reads the session's molecule while it is destroyed. A wrong member order reads
    // freed memory, which AddressSanitizer reports and a plain debug build may hang on.
    EXPECT_EQ(probe.atoms_at_destruction, system.NumAtoms());
}
