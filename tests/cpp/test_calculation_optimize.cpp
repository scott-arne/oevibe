#include <gtest/gtest.h>

#include <oechem.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "oevibe/Calculation.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

CalculationOptions mmff94() {
    CalculationOptions options;
    options.SetMethod(Method(MMFF94));
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

// OpenEye stores coordinates in single precision, so written-back coordinates match the
// double-precision result only to about 1e-6 Angstrom.
void expect_close(const std::vector<double>& a, const std::vector<double>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        EXPECT_NEAR(a[k], b[k], 1e-4);
    }
}

}  // namespace

TEST(CalculationOptimizeTest, Mmff94LowersEveryConformerAndWritesCoordinates) {
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    jitter(mol);
    OEChem::OEMol reference(mol);
    const ResultSet singles = Calculation(mmff94()).Energy(reference);
    std::vector<std::vector<double>> before;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf) {
        before.push_back(coordinates_of(*conf));
    }
    const ResultSet results = Calculation(mmff94()).Optimize(mol, OptimizerOptions());
    unsigned position = 0;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = mol.GetConfs(); conf; ++conf, ++position) {
        const ConformerResult& result = results.Get(position);
        ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
        EXPECT_LE(result.GetEnergy(), singles.Get(position).GetEnergy() + 1e-6);
        EXPECT_TRUE(result.GetOptimizationStats().converged);
        expect_close(coordinates_of(*conf), result.GetCoordinates());
        EXPECT_NE(coordinates_of(*conf), before[position]);
    }
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("optimizer.algorithm"), "lbfgs");
}

TEST(CalculationOptimizeTest, UpdateCoordinatesFalseKeepsTheInput) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    CalculationOptions options = mmff94();
    options.SetUpdateCoordinates(false);
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    EXPECT_EQ(coordinates_of(*mol.GetActive()), before);
    EXPECT_TRUE(results.Get(0).HasCoordinates());
    EXPECT_NE(results.Get(0).GetCoordinates(), before);
}

TEST(CalculationOptimizeTest, FrozenAtomsStayPut) {
    OEChem::OEMol mol = test::make_conformers("CCCCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    ConstraintSet constraints;
    constraints.FreezeAtoms(mol, OEChem::OEIsHeavy());
    OptimizerOptions options;
    options.SetConstraints(constraints);
    const ResultSet results = Calculation(mmff94()).Optimize(mol, options);
    // A failed conformer writes nothing back, which would leave the frozen atoms trivially put.
    ASSERT_EQ(results.Get(0).GetStatus(), ConformerStatus::OK) << results.Get(0).GetMessage();
    const std::vector<double> after = coordinates_of(*mol.GetActive());
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(OEChem::OEIsHeavy()); atom;
         ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(after[3 * atom->GetIdx() + axis], before[3 * atom->GetIdx() + axis], 1e-6);
        }
    }
}

TEST(CalculationOptimizeTest, DihedralConstraintIsHeld) {
    OEChem::OEMol mol = test::make_conformers("CCCC");
    std::vector<unsigned> carbons;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(OEChem::OEIsHeavy()); atom;
         ++atom) {
        carbons.push_back(atom->GetIdx());
    }
    ConstraintSet constraints;
    constraints.AddDihedral(carbons[0], carbons[1], carbons[2], carbons[3], 60.0);
    OptimizerOptions options;
    options.SetConstraints(constraints);
    const ResultSet results = Calculation(mmff94()).Optimize(mol, options);
    ASSERT_EQ(results.Get(0).GetStatus(), ConformerStatus::OK) << results.Get(0).GetMessage();
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("optimizer.algorithm"), "anc_rfo");
    OEChem::OEConfBase* conf = mol.GetActive();
    const double torsion =
        OEChem::OEGetTorsion(*conf, conf->GetAtom(OEChem::OEHasAtomIdx(carbons[0])),
                             conf->GetAtom(OEChem::OEHasAtomIdx(carbons[1])),
                             conf->GetAtom(OEChem::OEHasAtomIdx(carbons[2])),
                             conf->GetAtom(OEChem::OEHasAtomIdx(carbons[3])));
    EXPECT_NEAR(torsion * Units::RADIANS_TO_DEGREES, 60.0, 1e-3);
}

TEST(CalculationOptimizeTest, ConstraintContentEntersProvenanceAndDigest) {
    // SMILES parsing numbers the carbons 0-3; OMEGA appends the hydrogens after them.
    const OEChem::OEMol mol = test::make_conformers("CCCC");
    const auto provenance_of = [&mol](const ConstraintSet& constraints) {
        OEChem::OEMol copy(mol);
        OptimizerOptions options;
        options.SetConstraints(constraints);
        return Calculation(mmff94()).Optimize(copy, options).Get(0).GetProvenance();
    };

    ConstraintSet sixty;
    sixty.AddDihedral(0, 1, 2, 3, 60.0);
    sixty.AddDistance(1, 0);
    ConstraintSet one_twenty;
    one_twenty.AddDihedral(0, 1, 2, 3, 120.0);
    one_twenty.AddDistance(1, 0);
    const Provenance at_sixty = provenance_of(sixty);
    const Provenance at_one_twenty = provenance_of(one_twenty);
    EXPECT_EQ(at_sixty.resolved_options.at("optimizer.constraints"),
              "dihedral:0,1,2,3:60.000000;distance:1,0:start");
    EXPECT_EQ(at_one_twenty.resolved_options.at("optimizer.constraints"),
              "dihedral:0,1,2,3:120.000000;distance:1,0:start");
    EXPECT_NE(at_sixty.options_digest, at_one_twenty.options_digest);

    ConstraintSet low;
    low.FreezeAtomIndices({1, 0});
    ConstraintSet high;
    high.FreezeAtomIndices({3, 2});
    const Provenance frozen_low = provenance_of(low);
    const Provenance frozen_high = provenance_of(high);
    EXPECT_EQ(frozen_low.resolved_options.at("optimizer.frozen_atoms"), "0,1");
    EXPECT_EQ(frozen_high.resolved_options.at("optimizer.frozen_atoms"), "2,3");
    EXPECT_NE(frozen_low.options_digest, frozen_high.options_digest);

    ConstraintSet restrained;
    restrained.AddPositionRestraint(0, 10.0);
    restrained.AddDistanceRestraint(0, 3, 5.0, 3.5);
    const Provenance with_restraints = provenance_of(restrained);
    EXPECT_EQ(with_restraints.resolved_options.at("optimizer.restraints"),
              "position:0:10.000000:start;distance:0,3:5.000000:3.500000");
    EXPECT_EQ(with_restraints.resolved_options.at("optimizer.frozen_atoms"), "");
    EXPECT_EQ(with_restraints.resolved_options.at("optimizer.constraints"), "");
}

// A position restraint breaks rigid-body invariance; ANC_RFO used to project translations and
// rotations out anyway and stopped short of the restrained minimum LBFGS finds.
TEST(CalculationOptimizeTest, PositionRestraintGivesTheSameMinimumWithBothAlgorithms) {
    OEChem::OEMol mol = test::make_conformers("CCCCO");
    jitter(mol);
    ConstraintSet anchored;
    anchored.AddPositionRestraint(0, 10.0);
    double totals[2];
    double restraint_energies[2];
    int slot = 0;
    for (Algorithm algorithm : {Algorithm::LBFGS, Algorithm::ANC_RFO}) {
        SCOPED_TRACE(algorithm_name(algorithm));
        OEChem::OEMol copy(mol);
        OptimizerOptions options;
        options.SetAlgorithm(algorithm);
        options.SetConstraints(anchored);
        options.SetGradientNormThreshold(0.01);
        options.SetEnergyChangeThreshold(1e-7);
        const ResultSet results = Calculation(mmff94()).Optimize(copy, options);
        const ConformerResult& result = results.Get(0);
        ASSERT_EQ(result.GetStatus(), ConformerStatus::OK) << result.GetMessage();
        restraint_energies[slot] = result.GetOptimizationStats().restraint_energy;
        totals[slot++] = result.GetEnergy() + result.GetOptimizationStats().restraint_energy;
    }
    EXPECT_LT(restraint_energies[0], 1e-4);
    EXPECT_LT(restraint_energies[1], 1e-4);
    EXPECT_NEAR(totals[0], totals[1], 1e-3);
}

TEST(CalculationOptimizeTest, OptionAndInputErrorsComeFirst) {
    OEChem::OEMol mol = test::make_conformers("CCO");
    OptimizerOptions lbfgs;
    ConstraintSet exact;
    exact.AddDistance(0, 1);
    lbfgs.SetConstraints(exact);
    lbfgs.SetAlgorithm(Algorithm::LBFGS);
    EXPECT_THROW(Calculation(mmff94()).Optimize(mol, lbfgs), OptionsError);
    OptimizerOptions missing;
    ConstraintSet far;
    far.AddDistance(0, 999);
    missing.SetConstraints(far);
    EXPECT_THROW(Calculation(mmff94()).Optimize(mol, missing), InputError);
}

TEST(CalculationOptimizeTest, RejectedCallsLeaveTheMoleculeUnchanged) {
    // Input preparation would add the six hydrogens to the caller's molecule.
    const auto implicit_ethanol = []() {
        OEChem::OEMol mol = test::make_conformers("CCO");
        OEChem::OESuppressHydrogens(mol);
        return mol;
    };

    OEChem::OEMol missing_atom = implicit_ethanol();
    OptimizerOptions far_reference;
    ConstraintSet far;
    far.AddDistance(0, 999);
    far_reference.SetConstraints(far);
    EXPECT_THROW(Calculation(mmff94()).Optimize(missing_atom, far_reference), InputError);
    EXPECT_EQ(missing_atom.NumAtoms(), 3u);
    EXPECT_TRUE(OEChem::OEHasImplicitHydrogens(missing_atom));

    // Validate() accepts a lone initial trust radius; only resolution against the preset's
    // maximum rejects it.
    OEChem::OEMol wide_trust = implicit_ethanol();
    OptimizerOptions wide_start;
    wide_start.SetInitialTrustRadius(2.0 * preset_values(Preset::STANDARD).max_trust_radius);
    EXPECT_THROW(Calculation(mmff94()).Optimize(wide_trust, wide_start), OptionsError);
    EXPECT_EQ(wide_trust.NumAtoms(), 3u);
    EXPECT_TRUE(OEChem::OEHasImplicitHydrogens(wide_trust));
}

TEST(CalculationOptimizeTest, LateEngineFailuresKeepTheLastAcceptedGeometry) {
    for (const test::FakeMode mode :
         {test::FakeMode::THROW_RUNTIME, test::FakeMode::MALFORMED_GRADIENT}) {
        SCOPED_TRACE(mode == test::FakeMode::THROW_RUNTIME ? "exception" : "malformed");
        OEChem::OEMol mol = test::make_conformers("CCCCO");
        const std::shared_ptr<test::FakeEngine> engine =
            test::register_fake("optimize-late-failure");
        // OMEGA leaves no index gaps, so the OpenEye coordinates are also the dense ones.
        const std::vector<double> reference = coordinates_of(*mol.GetActive());
        engine->reference = reference;
        engine->mode = mode;
        engine->healthy_evaluations = 5;
        jitter(mol);
        const std::vector<double> start = coordinates_of(*mol.GetActive());
        CalculationOptions options;
        options.SetMethod(Method("fake-method"));
        options.SetEngine("optimize-late-failure");

        const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
        const ConformerResult& result = results.Get(0);
        ASSERT_EQ(result.GetStatus(), ConformerStatus::ENGINE_ERROR);
        EXPECT_NE(result.GetMessage().find(mode == test::FakeMode::THROW_RUNTIME
                                               ? "fake engine failure"
                                               : "malformed Evaluation"),
                  std::string::npos)
            << result.GetMessage();
        ASSERT_TRUE(result.HasEnergy());
        ASSERT_TRUE(result.HasCoordinates());
        ASSERT_TRUE(result.HasOptimizationStats());
        EXPECT_FALSE(result.GetOptimizationStats().converged);
        EXPECT_GE(result.GetOptimizationStats().iterations, 1u);
        EXPECT_GE(result.GetOptimizationStats().gradient_evaluations, 5u);
        // The energy is the engine's at the reported geometry, which improves on the start.
        EXPECT_NEAR(result.GetEnergy(),
                    test::spring_energy(reference, result.GetCoordinates(), engine->force_constant,
                                        nullptr),
                    1e-9);
        EXPECT_LT(result.GetEnergy(),
                  test::spring_energy(reference, start, engine->force_constant, nullptr));
        // ENGINE_ERROR conformers are not written back (spec 6.5).
        EXPECT_EQ(coordinates_of(*mol.GetActive()), start);
    }
}

TEST(CalculationOptimizeTest, ImmediateEngineFailuresKeepNoValues) {
    const std::shared_ptr<test::FakeEngine> engine =
        test::register_fake("optimize-immediate-failure");
    engine->mode = test::FakeMode::THROW_RUNTIME;
    CalculationOptions options;
    options.SetMethod(Method("fake-method"));
    options.SetEngine("optimize-immediate-failure");
    OEChem::OEMol mol = test::make_conformers("CCO");
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    const ConformerResult& result = results.Get(0);
    EXPECT_EQ(result.GetStatus(), ConformerStatus::ENGINE_ERROR);
    EXPECT_FALSE(result.HasEnergy());
    EXPECT_FALSE(result.HasCoordinates());
    EXPECT_FALSE(result.HasOptimizationStats());
}

TEST(CalculationOptimizeTest, NullSessionsBecomeEngineErrors) {
    const std::shared_ptr<test::FakeEngine> engine = test::register_fake("optimize-null-session");
    engine->null_session = true;
    CalculationOptions options;
    options.SetMethod(Method("fake-method"));
    options.SetEngine("optimize-null-session");
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    ASSERT_GE(mol.NumConfs(), 2u);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    ASSERT_EQ(results.Size(), mol.NumConfs());
    EXPECT_EQ(engine->sessions_created, static_cast<int>(mol.NumConfs()));
    for (std::size_t i = 0; i < results.Size(); ++i) {
        const ConformerResult& result = results.Get(i);
        EXPECT_EQ(result.GetStatus(), ConformerStatus::ENGINE_ERROR);
        EXPECT_NE(result.GetMessage().find("null session"), std::string::npos)
            << result.GetMessage();
        EXPECT_FALSE(result.HasEnergy());
        EXPECT_FALSE(result.HasCoordinates());
        EXPECT_FALSE(result.HasOptimizationStats());
    }
    EXPECT_EQ(coordinates_of(*mol.GetActive()), before);
}

TEST(CalculationOptimizeTest, AutoSolventResolutionIsRecordedAsAWarning) {
    OEChem::OEMol mol = test::make_conformers("CCCCO", 2);
    CalculationOptions options = mmff94();
    options.SetSolvent(Solvent("water"));
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    ASSERT_EQ(results.Size(), mol.NumConfs());
    for (std::size_t i = 0; i < results.Size(); ++i) {
        const std::vector<std::string> warnings = results.Get(i).GetWarnings();
        EXPECT_TRUE(std::any_of(warnings.begin(), warnings.end(), [](const std::string& warning) {
            return warning.find("water") != std::string::npos &&
                   warning.find("sheffield") != std::string::npos;
        }));
    }
}

TEST(CalculationOptimizeTest, IterationLimitGivesNotConverged) {
    OEChem::OEMol mol = test::make_conformers("CCCCO");
    jitter(mol);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    OptimizerOptions options;
    options.SetMaxIterations(1);
    const ResultSet results = Calculation(mmff94()).Optimize(mol, options);
    EXPECT_EQ(results.Get(0).GetStatus(), ConformerStatus::NOT_CONVERGED);
    EXPECT_FALSE(results.Get(0).HasRelativeEnergy());
    EXPECT_TRUE(results.Get(0).HasEnergy());
    EXPECT_NE(coordinates_of(*mol.GetActive()), before);
}

TEST(CalculationOptimizeTest, NonForceFieldEnginesUseAncRfo) {
    test::register_fake("optimize-fake");
    CalculationOptions options;
    options.SetMethod(Method("fake-method"));
    options.SetEngine("optimize-fake");
    OEChem::OEMol mol = test::make_conformers("CCCCO", 3);
    const ResultSet results = Calculation(options).Optimize(mol, OptimizerOptions());
    // At the STANDARD gradient-norm threshold the residual spring energy is a few hundredths.
    for (std::size_t i = 0; i < results.Size(); ++i) {
        ASSERT_EQ(results.Get(i).GetStatus(), ConformerStatus::OK) << results.Get(i).GetMessage();
        EXPECT_LT(results.Get(i).GetEnergy(), 0.05);
    }
    EXPECT_EQ(results.Get(0).GetProvenance().resolved_options.at("optimizer.algorithm"), "anc_rfo");
}
