#include <gtest/gtest.h>

#include <oechem.h>
#include <oemmff.h>

#include <cmath>
#include <vector>

#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "oevibe/engines/OEFFEngine.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

double energy_of(OEFFEngine& engine, const MolecularSystem& system, const char* method,
                 const Solvent& solvent = Solvent()) {
    EngineSettings settings;
    settings.method = Method(method);
    settings.solvent = solvent;
    auto session = engine.CreateSession(system, settings);
    return session->Evaluate(system.GetConformerCoordinates(0), EvaluationRequest()).energy;
}

std::vector<double> gradient_at(EngineSession& session, const std::vector<double>& coords) {
    EvaluationRequest request;
    request.gradient = true;
    return session.Evaluate(coords, request).gradient;
}

/// Compares the analytic Hessian of ``method`` on ethanol with central differences of its
/// analytic gradient over every coordinate. Each OpenEye potential packs its Hessian itself, so
/// the layout has to be checked per method.
void expect_hessian_matches_finite_differences(const char* method, const Solvent& solvent,
                                               PartialChargePolicy policy) {
    SCOPED_TRACE(method);
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1, policy);
    OEFFEngine engine;
    EngineSettings settings;
    settings.method = Method(method);
    settings.solvent = solvent;
    auto session = engine.CreateSession(system, settings);

    const std::vector<double> x = system.GetConformerCoordinates(0);
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
            EXPECT_NEAR(evaluation.hessian[k * dim + j], numeric, 1e-2 + 1e-3 * std::abs(numeric))
                << "row " << k << " col " << j;
        }
    }
}

}  // namespace

TEST(OEFFEngineTest, IsTheBuiltinForForceFieldMethods) {
    ASSERT_NE(EngineRegistry::Instance().Find("oeff"), nullptr);
    EXPECT_EQ(EngineRegistry::Instance().ResolveForMethod(Method(MMFF94))->GetName(), "oeff");
    EXPECT_EQ(EngineRegistry::Instance().ResolveForMethod(Method(SAGE))->GetName(), "oeff");
}

TEST(OEFFEngineTest, DeclaresFourMethods) {
    const OEFFEngine engine;
    EXPECT_FALSE(engine.GetVersion().empty());
    const std::vector<MethodCapabilities> methods = engine.GetMethods();
    ASSERT_EQ(methods.size(), 4u);
    for (const MethodCapabilities& caps : methods) {
        EXPECT_TRUE(caps.force_field);
        EXPECT_TRUE(caps.charge_from_topology);
        EXPECT_TRUE(caps.analytic_hessian);
        EXPECT_FALSE(caps.open_shell);
        EXPECT_EQ(caps.properties, 0u);
        const bool mmff = caps.method == Method(MMFF94) || caps.method == Method(MMFF94S);
        EXPECT_EQ(caps.requires_partial_charges, !mmff);
        if (mmff) {
            EXPECT_EQ(caps.solvent_models, (std::vector<SolventModel>{SolventModel::SHEFFIELD}));
            EXPECT_EQ(caps.solvents, (std::vector<std::string>{"water"}));
        } else {
            EXPECT_TRUE(caps.solvent_models.empty());
        }
    }
}

TEST(OEFFEngineTest, Mmff94EnergyMatchesOpenEye) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    OEFFEngine engine;
    OEMolMMFF::OEMMFF direct{OEMolMMFF::OEMMFF94Params()};
    OEChem::OEMol copy(mol);
    ASSERT_TRUE(direct.PrepMol(copy));
    ASSERT_TRUE(direct.Setup(copy));
    std::vector<double> coords(3 * copy.GetMaxAtomIdx());
    copy.GetCoords(coords.data());
    EXPECT_NEAR(energy_of(engine, system, MMFF94), direct(coords.data()), 1e-8);
}

// The engine selects the variant with a bare flag, so each method is pinned to OpenEye's own
// parameter set on a delocalized nitrogen, where MMFF94 and MMFF94s differ.
TEST(OEFFEngineTest, MmffVariantsMatchTheirOpenEyeParameterSets) {
    const OEChem::OEMol mol = test::make_conformers("Nc1ccccc1");
    const MolecularSystem system(mol, 0, 1);
    OEFFEngine engine;
    const auto direct_energy = [&mol](const OEMolMMFF::OEMMFFParams& params) {
        OEMolMMFF::OEMMFF direct(params);
        OEChem::OEMol copy(mol);
        EXPECT_TRUE(direct.PrepMol(copy));
        EXPECT_TRUE(direct.Setup(copy));
        std::vector<double> coords(3 * copy.GetMaxAtomIdx());
        copy.GetCoords(coords.data());
        return direct(coords.data());
    };
    EXPECT_NEAR(energy_of(engine, system, MMFF94), direct_energy(OEMolMMFF::OEMMFF94Params()),
                1e-8);
    EXPECT_NEAR(energy_of(engine, system, MMFF94S), direct_energy(OEMolMMFF::OEMMFF94sParams()),
                1e-8);
}

TEST(OEFFEngineTest, Mmff94sDiffersForADelocalizedNitrogen) {
    const OEChem::OEMol mol = test::make_conformers("Nc1ccccc1");
    const MolecularSystem system(mol, 0, 1);
    OEFFEngine engine;
    EXPECT_GT(std::abs(energy_of(engine, system, MMFF94) - energy_of(engine, system, MMFF94S)),
              1e-6);
}

// Every MMFF partial charge of an alkane is zero, which OpenEye's charge check reads as absent.
TEST(OEFFEngineTest, MmffRunsOnHydrocarbons) {
    const OEChem::OEMol mol = test::make_conformers("CCCC");
    const MolecularSystem system(mol, 0, 1);
    OEFFEngine engine;
    EXPECT_TRUE(std::isfinite(energy_of(engine, system, MMFF94)));
    EXPECT_TRUE(std::isfinite(energy_of(engine, system, MMFF94S)));
}

TEST(OEFFEngineTest, SheffieldWaterChangesTheEnergy) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    OEFFEngine engine;
    const double gas = energy_of(engine, system, MMFF94);
    const double water =
        energy_of(engine, system, MMFF94, Solvent("water", SolventModel::SHEFFIELD));
    EXPECT_GT(std::abs(water - gas), 0.1);
}

TEST(OEFFEngineTest, SageAndParsleyRunOnAm1BccCharges) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1, PartialChargePolicy::ASSIGN_AM1BCC_ELF10);
    OEFFEngine engine;
    EXPECT_TRUE(std::isfinite(energy_of(engine, system, SAGE)));
    EXPECT_TRUE(std::isfinite(energy_of(engine, system, PARSLEY)));
}

TEST(OEFFEngineTest, UnparameterizedAtomIsUnsupported) {
    OEChem::OEGraphMol xenon;
    xenon.NewAtom(54);
    const double origin[3] = {0.0, 0.0, 0.0};
    xenon.SetCoords(origin);
    xenon.SetDimension(3);
    const MolecularSystem system(xenon, 0, 1);
    OEFFEngine engine;
    EngineSettings settings;
    settings.method = Method(MMFF94);
    EXPECT_THROW(engine.CreateSession(system, settings), UnsupportedError);
}

TEST(OEFFEngineTest, AnalyticHessiansMatchFiniteDifferences) {
    expect_hessian_matches_finite_differences(MMFF94S, Solvent(), PartialChargePolicy::NONE);
    expect_hessian_matches_finite_differences(MMFF94, Solvent("water", SolventModel::SHEFFIELD),
                                              PartialChargePolicy::NONE);
    expect_hessian_matches_finite_differences(SAGE, Solvent(),
                                              PartialChargePolicy::ASSIGN_AM1BCC_ELF10);
    expect_hessian_matches_finite_differences(PARSLEY, Solvent(),
                                              PartialChargePolicy::ASSIGN_AM1BCC_ELF10);
}
