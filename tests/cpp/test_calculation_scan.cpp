#include <gtest/gtest.h>

#include <oechem.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Geometry.h"
#include "core/WriteBack.h"
#include "oevibe/Calculation.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"
#include "oevibe/engines/OEFFEngine.h"
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

// OMEGA renumbers atoms, so butane's backbone is not atoms 0-1-2-3: this walks the heavy-atom
// chain from its lowest-index end.
std::vector<unsigned> heavy_chain(const OEChem::OEMolBase& mol) {
    const OEChem::OEAtomBase* current = nullptr;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(OEChem::OEIsHeavy()); atom;
         ++atom) {
        if (atom->GetHvyDegree() == 1 &&
            (current == nullptr || atom->GetIdx() < current->GetIdx())) {
            current = atom;
        }
    }
    std::vector<unsigned> chain = {current->GetIdx()};
    const OEChem::OEAtomBase* previous = nullptr;
    while (true) {
        const OEChem::OEAtomBase* next = nullptr;
        for (OESystem::OEIter<OEChem::OEAtomBase> nbr = current->GetAtoms(OEChem::OEIsHeavy()); nbr;
             ++nbr) {
            if (previous == nullptr || nbr->GetIdx() != previous->GetIdx()) {
                next = nbr;
                break;
            }
        }
        if (next == nullptr) {
            return chain;
        }
        chain.push_back(next->GetIdx());
        previous = current;
        current = next;
    }
}

double torsion_degrees(const OEChem::OEConfBase& conf, const std::vector<unsigned>& atoms,
                       std::size_t first) {
    return OEChem::OEGetTorsion(conf, conf.GetAtom(OEChem::OEHasAtomIdx(atoms[first])),
                                conf.GetAtom(OEChem::OEHasAtomIdx(atoms[first + 1])),
                                conf.GetAtom(OEChem::OEHasAtomIdx(atoms[first + 2])),
                                conf.GetAtom(OEChem::OEHasAtomIdx(atoms[first + 3]))) *
           Units::RADIANS_TO_DEGREES;
}

// One OMEGA conformer of ``smiles`` with every backbone dihedral set to anti, so tests know
// which grid point the input seeds.
OEChem::OEMol all_anti(const std::string& smiles, std::vector<unsigned>& chain) {
    OEChem::OEMol mol = test::make_conformers(smiles);
    chain = heavy_chain(mol);
    OEChem::OEConfBase& conf = *mol.GetActive();
    for (std::size_t first = 0; first + 3 < chain.size(); ++first) {
        OEChem::OESetTorsion(conf, conf.GetAtom(OEChem::OEHasAtomIdx(chain[first])),
                             conf.GetAtom(OEChem::OEHasAtomIdx(chain[first + 1])),
                             conf.GetAtom(OEChem::OEHasAtomIdx(chain[first + 2])),
                             conf.GetAtom(OEChem::OEHasAtomIdx(chain[first + 3])), Units::PI);
    }
    return mol;
}

ScanOptions backbone_scan(const std::vector<unsigned>& chain, double step) {
    ScanOptions options;
    options.AddDihedral(chain[0], chain[1], chain[2], chain[3]);
    options.SetStep(step);
    return options;
}

// Relative energies keyed by the rounded angle of a 1D scan.
std::map<long, double> profile_of(const ScanResult& scan) {
    std::map<long, double> profile;
    for (std::size_t i = 0; i < scan.GetNumPoints(); ++i) {
        const ScanPoint& point = scan.GetPoint(i);
        if (point.HasRelativeEnergy()) {
            profile[std::lround(point.GetAngles()[0])] = point.GetRelativeEnergy();
        }
    }
    return profile;
}

const ScanPoint& point_at(const ScanResult& scan, double angle) {
    for (std::size_t i = 0; i < scan.GetNumPoints(); ++i) {
        if (std::abs(scan.GetPoint(i).GetAngles()[0] - angle) < 1e-9) {
            return scan.GetPoint(i);
        }
    }
    throw std::runtime_error("no scan point at " + std::to_string(angle));
}

// MMFF94 through OEFFEngine, except near two values of one dihedral: within 15 degrees of +60
// it reports the system unsupported, and within 15 degrees of -60 it fails. An optimization
// constrained to either grid value must end inside its window, so those two points fail
// whichever neighbor seeds them. run_optimization propagates UnsupportedError from a session
// unchanged (src/core/Optimizer.h), so an evaluation can report it.
class GatedSession : public EngineSession {
public:
    GatedSession(std::shared_ptr<EngineSession> inner, std::vector<unsigned> atoms)
        : inner_(std::move(inner)), atoms_(std::move(atoms)) {}

    Evaluation Evaluate(const std::vector<double>& coords,
                        const EvaluationRequest& request) override {
        const double angle =
            detail::dihedral(coords.data(), atoms_[0], atoms_[1], atoms_[2], atoms_[3], nullptr) *
            Units::RADIANS_TO_DEGREES;
        if (std::abs(angle - 60.0) < 15.0) {
            throw UnsupportedError("gated engine: unsupported near +60 degrees");
        }
        if (std::abs(angle + 60.0) < 15.0) {
            throw std::runtime_error("gated engine: failure near -60 degrees");
        }
        return inner_->Evaluate(coords, request);
    }

private:
    std::shared_ptr<EngineSession> inner_;
    std::vector<unsigned> atoms_;
};

class GatedEngine : public OEFFEngine {
public:
    /// :param atoms: Dense indices of the gated dihedral.
    explicit GatedEngine(std::vector<unsigned> atoms) : atoms_(std::move(atoms)) {}

    std::string GetName() const override { return "scan-gated"; }

    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override {
        ++sessions_created;
        return std::make_shared<GatedSession>(OEFFEngine::CreateSession(system, settings), atoms_);
    }

    int sessions_created = 0;

private:
    std::vector<unsigned> atoms_;
};

// MMFF94 through OEFFEngine, except that the ``failing_session``-th session created throws a
// runtime error once it has served ``healthy_evaluations`` evaluations. The optimizer has then
// accepted iterates, which a scan point must not keep when its attempt fails.
class FlakySession : public EngineSession {
public:
    FlakySession(std::shared_ptr<EngineSession> inner, bool flaky, int healthy_evaluations,
                 int* served)
        : inner_(std::move(inner)),
          flaky_(flaky),
          healthy_evaluations_(healthy_evaluations),
          served_(served) {}

    Evaluation Evaluate(const std::vector<double>& coords,
                        const EvaluationRequest& request) override {
        if (flaky_) {
            if (evaluations_ >= healthy_evaluations_) {
                throw std::runtime_error("flaky engine: failure after accepted iterates");
            }
            ++evaluations_;
            ++*served_;
        }
        return inner_->Evaluate(coords, request);
    }

private:
    std::shared_ptr<EngineSession> inner_;
    bool flaky_;
    int healthy_evaluations_;
    int evaluations_ = 0;
    int* served_;
};

class FlakyEngine : public OEFFEngine {
public:
    FlakyEngine(int failing_session, int healthy_evaluations)
        : failing_session_(failing_session), healthy_evaluations_(healthy_evaluations) {}

    std::string GetName() const override { return "scan-flaky"; }

    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override {
        ++sessions_created;
        return std::make_shared<FlakySession>(OEFFEngine::CreateSession(system, settings),
                                              sessions_created == failing_session_,
                                              healthy_evaluations_, &evaluations_before_failure);
    }

    int sessions_created = 0;
    int evaluations_before_failure = 0;

private:
    int failing_session_;
    int healthy_evaluations_;
};

double distance_between(const std::vector<double>& first, unsigned a,
                        const std::vector<double>& second, unsigned b) {
    double sum = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double delta = first[3 * a + axis] - second[3 * b + axis];
        sum += delta * delta;
    }
    return std::sqrt(sum);
}

double distance_between(const std::vector<double>& coords, unsigned a, unsigned b) {
    return distance_between(coords, a, coords, b);
}

// ``atom`` and the atoms bonded to it other than ``keep``: for the end of a chain, the terminal
// group that moves when the bond to ``keep`` stretches.
std::vector<unsigned> terminal_group(const OEChem::OEMolBase& mol, unsigned atom, unsigned keep) {
    std::vector<unsigned> group = {atom};
    const OEChem::OEAtomBase* center = mol.GetAtom(OEChem::OEHasAtomIdx(atom));
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = center->GetAtoms(); nbr; ++nbr) {
        if (nbr->GetIdx() != keep) {
            group.push_back(nbr->GetIdx());
        }
    }
    return group;
}

void translate(std::vector<double>& coords, const std::vector<unsigned>& atoms,
               const std::array<double, 3>& shift) {
    for (const unsigned atom : atoms) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            coords[3 * atom + axis] += shift[axis];
        }
    }
}

}  // namespace

TEST(CalculationScanTest, ButaneMmff94ProfileHasTheExpectedShape) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    const std::vector<double> before = coordinates_of(*mol.GetActive());
    const ScanResult scan = Calculation(mmff94()).Scan(mol, backbone_scan(chain, 30.0));

    ASSERT_EQ(scan.GetNumPoints(), 12u);
    EXPECT_FALSE(scan.IsTruncated());
    ASSERT_EQ(scan.GetNumDihedrals(), 1u);
    EXPECT_EQ(scan.GetDihedral(0), std::vector<unsigned>(chain.begin(), chain.end()));
    for (std::size_t i = 0; i < scan.GetNumPoints(); ++i) {
        ASSERT_EQ(scan.GetPoint(i).GetStatus(), ScanPointStatus::OK)
            << i << ": " << scan.GetPoint(i).GetMessage();
    }
    const std::map<long, double> e = profile_of(scan);
    ASSERT_EQ(e.size(), 12u);
    // Anti is the global minimum.
    EXPECT_DOUBLE_EQ(e.at(180), 0.0);
    for (const auto& [angle, energy] : e) {
        if (angle != 180) {
            EXPECT_GT(energy, 0.3) << angle;
        }
    }
    // Gauche local minima above anti, eclipsed local maxima, and syn the global maximum.
    for (const long sign : {1L, -1L}) {
        EXPECT_LT(e.at(60 * sign), e.at(30 * sign));
        EXPECT_LT(e.at(60 * sign), e.at(90 * sign));
        EXPECT_GT(e.at(120 * sign), e.at(90 * sign));
        EXPECT_GT(e.at(120 * sign), e.at(150 * sign));
    }
    for (const auto& [angle, energy] : e) {
        if (angle != 0) {
            EXPECT_LT(energy, e.at(0)) << angle;
        }
    }
    EXPECT_NEAR(e.at(60), e.at(-60), 0.05);
    EXPECT_NEAR(e.at(120), e.at(-120), 0.05);
    // Sequential MMFF94 constrained optimizations (Plan B Task 15 probe) gave 0.84 (gauche),
    // 3.96 (eclipsed), and 5.21 (syn) kcal/mol above anti.
    EXPECT_NEAR(e.at(60), 0.84, 0.15);
    EXPECT_NEAR(e.at(120), 3.96, 0.15);
    EXPECT_NEAR(e.at(0), 5.21, 0.15);

    const std::map<std::string, std::string>& resolved = scan.GetProvenance().resolved_options;
    EXPECT_EQ(resolved.at("task"), "scan");
    EXPECT_EQ(resolved.at("scan.max_optimizations"), "240");
    EXPECT_EQ(resolved.at("optimizer.algorithm"), "anc_rfo");
    EXPECT_EQ(scan.GetProvenance().method, "mmff94");
    // A scan writes nothing back.
    EXPECT_EQ(coordinates_of(*mol.GetActive()), before);
    EXPECT_FALSE(OEChem::OEHasSDData(*mol.GetActive(), "OEVibe.mmff94.gas.Status"));
}

TEST(CalculationScanTest, PartialRangesIncludeBothEnds) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    ScanOptions options = backbone_scan(chain, 30.0);
    options.SetRange(0.0, 90.0);
    const ScanResult scan = Calculation(mmff94()).Scan(mol, options);
    ASSERT_EQ(scan.GetNumPoints(), 4u);
    const std::vector<double> expected = {0.0, 30.0, 60.0, 90.0};
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(scan.GetPoint(i).GetAngles()[0], expected[i]);
        EXPECT_EQ(scan.GetPoint(i).GetStatus(), ScanPointStatus::OK)
            << scan.GetPoint(i).GetMessage();
    }
    // Over 0-90 degrees gauche is the lowest point.
    const std::map<long, double> e = profile_of(scan);
    EXPECT_DOUBLE_EQ(e.at(60), 0.0);
    EXPECT_GT(e.at(0), e.at(30));
}

TEST(CalculationScanTest, TwoDimensionalScansAreRowMajor) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCCC", chain);
    ASSERT_EQ(chain.size(), 5u);
    ScanOptions options = backbone_scan(chain, 120.0);
    options.AddDihedral(chain[1], chain[2], chain[3], chain[4]);
    const ScanResult scan = Calculation(mmff94()).Scan(mol, options);
    ASSERT_EQ(scan.GetNumPoints(), 9u);
    ASSERT_EQ(scan.GetNumDihedrals(), 2u);
    EXPECT_EQ(scan.GetDihedral(1), std::vector<unsigned>(chain.begin() + 1, chain.end()));
    const std::vector<double> values = {-60.0, 60.0, 180.0};
    for (std::size_t i = 0; i < 9; ++i) {
        const ScanPoint& point = scan.GetPoint(i);
        EXPECT_EQ(point.GetAngles(), (std::vector<double>{values[i / 3], values[i % 3]})) << i;
        EXPECT_TRUE(point.HasEnergy()) << i << ": " << point.GetMessage();
    }
    // All-anti is the global minimum, and gauche+/gauche- (points 1 and 3) carries the
    // syn-pentane clash that gauche+/gauche+ (point 4) avoids.
    EXPECT_DOUBLE_EQ(scan.GetPoint(8).GetRelativeEnergy(), 0.0);
    EXPECT_GT(scan.GetPoint(1).GetRelativeEnergy(), scan.GetPoint(4).GetRelativeEnergy() + 2.0);
    EXPECT_GT(scan.GetPoint(3).GetRelativeEnergy(), scan.GetPoint(4).GetRelativeEnergy() + 2.0);
    const OEChem::OEMol profile = scan.GetMolecule();
    ASSERT_EQ(profile.NumConfs(), 9u);
    const OEChem::OEConfBase* second = profile.GetConf(OEChem::OEHasConfIdx(1));
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(OEChem::OEGetSDData(*second, "OEVibe.mmff94.gas.ScanAngles"), "-60.000000,60.000000");
}

TEST(CalculationScanTest, CapTruncatesAndGetMoleculeHoldsThePointsWithEnergies) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    ScanOptions options = backbone_scan(chain, 30.0);
    // The seed at 180 and its two neighbors, then the cap.
    options.SetMaxOptimizations(3);
    const ScanResult scan = Calculation(mmff94()).Scan(mol, options);
    EXPECT_TRUE(scan.IsTruncated());
    const std::vector<std::string> warnings = scan.GetWarnings();
    EXPECT_TRUE(std::any_of(warnings.begin(), warnings.end(), [](const std::string& warning) {
        return warning.find("cap of 3") != std::string::npos;
    }));
    std::vector<double> attempted;
    for (std::size_t i = 0; i < scan.GetNumPoints(); ++i) {
        const ScanPoint& point = scan.GetPoint(i);
        if (point.GetStatus() == ScanPointStatus::NOT_ATTEMPTED) {
            EXPECT_FALSE(point.HasEnergy());
            EXPECT_FALSE(point.HasRelativeEnergy());
            EXPECT_NE(point.GetMessage().find("cap"), std::string::npos);
        } else {
            EXPECT_EQ(point.GetStatus(), ScanPointStatus::OK) << point.GetMessage();
            attempted.push_back(point.GetAngles()[0]);
        }
    }
    EXPECT_EQ(attempted, (std::vector<double>{-150.0, 150.0, 180.0}));

    const OEChem::OEMol profile = scan.GetMolecule();
    ASSERT_EQ(profile.NumConfs(), 3u);
    const std::string prefix = "OEVibe.mmff94.gas.";
    std::size_t k = 0;
    for (OESystem::OEIter<OEChem::OEConfBase> conf = profile.GetConfs(); conf; ++conf, ++k) {
        const ScanPoint& point = point_at(scan, attempted[k]);
        EXPECT_EQ(OEChem::OEGetSDData(*conf, prefix + "ScanAngles"),
                  detail::format_value(attempted[k]));
        EXPECT_EQ(OEChem::OEGetSDData(*conf, prefix + "Energy"),
                  detail::format_value(point.GetEnergy()));
        EXPECT_EQ(OEChem::OEGetSDData(*conf, prefix + "RelativeEnergy"),
                  detail::format_value(point.GetRelativeEnergy()));
        EXPECT_EQ(OEChem::OEGetSDData(*conf, prefix + "Status"), "OK");
        // OpenEye stores coordinates in single precision.
        EXPECT_NEAR(
            std::abs(normalize_dihedral_degrees(torsion_degrees(*conf, chain, 0) - attempted[k])),
            0.0, 0.01);
        const std::vector<double> stored = coordinates_of(*conf);
        const std::vector<double> expected = point.GetCoordinates();
        ASSERT_EQ(stored.size(), expected.size());
        for (std::size_t c = 0; c < stored.size(); ++c) {
            EXPECT_NEAR(stored[c], expected[c], 1e-4);
        }
    }
}

TEST(CalculationScanTest, OptionErrorsComeFirst) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    const auto with_constraints = [&](const ConstraintSet& constraints) {
        ScanOptions options = backbone_scan(chain, 30.0);
        OptimizerOptions optimizer;
        optimizer.SetConstraints(constraints);
        options.SetOptimizerOptions(optimizer);
        return options;
    };
    ConstraintSet reversed;
    reversed.AddDihedral(chain[3], chain[2], chain[1], chain[0], 60.0);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, with_constraints(reversed)), OptionsError);
    ConstraintSet restrained;
    restrained.AddDihedralRestraint(chain[0], chain[1], chain[2], chain[3], 5.0);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, with_constraints(restrained)), OptionsError);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, ScanOptions()), OptionsError);
    ScanOptions lbfgs = backbone_scan(chain, 30.0);
    OptimizerOptions optimizer;
    optimizer.SetAlgorithm(Algorithm::LBFGS);
    lbfgs.SetOptimizerOptions(optimizer);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, lbfgs), OptionsError);

    // A scan computes no properties, so requesting them is an options error, not a silent no-op.
    test::register_fake("scan-properties");
    CalculationOptions properties;
    properties.SetMethod(Method("fake-method"));
    properties.SetEngine("scan-properties");
    properties.SetProperties(static_cast<unsigned>(Property::CHARGES));
    EXPECT_THROW(Calculation(properties).Scan(mol, backbone_scan(chain, 30.0)), OptionsError);
}

TEST(CalculationScanTest, MissingAtomsAreInputErrorsAndLeaveTheMoleculeUnchanged) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    // Input preparation would add the ten hydrogens to the caller's molecule.
    OEChem::OESuppressHydrogens(mol);
    ASSERT_EQ(mol.NumAtoms(), 4u);
    ScanOptions missing;
    missing.AddDihedral(chain[0], chain[1], chain[2], 99);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, missing), InputError);
    ConstraintSet far;
    far.AddDistance(chain[0], 99);
    OptimizerOptions optimizer;
    optimizer.SetConstraints(far);
    ScanOptions far_constraint = backbone_scan(chain, 30.0);
    far_constraint.SetOptimizerOptions(optimizer);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, far_constraint), InputError);
    EXPECT_EQ(mol.NumAtoms(), 4u);
    EXPECT_TRUE(OEChem::OEHasImplicitHydrogens(mol));
}

TEST(CalculationScanTest, PointStatusesFollowSpec75) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    // OMEGA leaves no index gaps, so the OpenEye indices are also the dense ones.
    const auto gated = std::make_shared<GatedEngine>(chain);
    EngineRegistry::Instance().Register(gated, true);
    CalculationOptions options = mmff94();
    options.SetEngine("scan-gated");
    // Grid -120, -60, 0, 60, 120, 180; the seed is 180.
    const ScanResult scan = Calculation(options).Scan(mol, backbone_scan(chain, 60.0));
    EXPECT_FALSE(scan.IsTruncated());
    for (const double angle : {180.0, 120.0, -120.0}) {
        EXPECT_EQ(point_at(scan, angle).GetStatus(), ScanPointStatus::OK) << angle;
        EXPECT_TRUE(point_at(scan, angle).HasRelativeEnergy()) << angle;
    }
    const ScanPoint& unsupported = point_at(scan, 60.0);
    EXPECT_EQ(unsupported.GetStatus(), ScanPointStatus::UNSUPPORTED);
    EXPECT_NE(unsupported.GetMessage().find("unsupported near +60"), std::string::npos);
    const ScanPoint& failed = point_at(scan, -60.0);
    EXPECT_EQ(failed.GetStatus(), ScanPointStatus::ENGINE_ERROR);
    EXPECT_NE(failed.GetMessage().find("failure near -60"), std::string::npos);
    // Both neighbors of 0 failed, so nothing ever seeded it.
    const ScanPoint& unreached = point_at(scan, 0.0);
    EXPECT_EQ(unreached.GetStatus(), ScanPointStatus::NOT_ATTEMPTED);
    EXPECT_NE(unreached.GetMessage().find("never reached"), std::string::npos);
    for (const ScanPoint* point : {&unsupported, &failed, &unreached}) {
        EXPECT_FALSE(point->HasEnergy());
        EXPECT_FALSE(point->HasRelativeEnergy());
        EXPECT_FALSE(point->HasCoordinates());
    }
    EXPECT_EQ(scan.GetMolecule().NumConfs(), 3u);
    EXPECT_EQ(scan.GetProvenance().engine, "scan-gated");
    // The seed, pass 1 (-120, 120), and pass 2 (-60, 180, 60, 180) at least, each in its own
    // session (spec 6.4).
    EXPECT_GE(gated->sessions_created, 7);

    // Every attempt stops at the iteration limit, so every point keeps a NOT_CONVERGED energy.
    ScanOptions one_step = backbone_scan(chain, 60.0);
    OptimizerOptions optimizer;
    optimizer.SetMaxIterations(1);
    one_step.SetOptimizerOptions(optimizer);
    const ScanResult limited = Calculation(mmff94()).Scan(mol, one_step);
    for (std::size_t i = 0; i < limited.GetNumPoints(); ++i) {
        const ScanPoint& point = limited.GetPoint(i);
        EXPECT_EQ(point.GetStatus(), ScanPointStatus::NOT_CONVERGED) << i;
        EXPECT_FALSE(point.GetMessage().empty());
        EXPECT_TRUE(point.HasEnergy());
        EXPECT_TRUE(point.HasRelativeEnergy());
    }
    const OEChem::OEMol limited_profile = limited.GetMolecule();
    ASSERT_EQ(limited_profile.NumConfs(), 6u);
    EXPECT_EQ(OEChem::OEGetSDData(*limited_profile.GetActive(), "OEVibe.mmff94.gas.Status"),
              "NOT_CONVERGED");
}

TEST(CalculationScanTest, UniformEngineFailuresStopAtTheSeed) {
    for (const bool null_session : {false, true}) {
        SCOPED_TRACE(null_session ? "null session" : "unsupported");
        const std::shared_ptr<test::FakeEngine> engine = test::register_fake("scan-fake-failure");
        engine->null_session = null_session;
        engine->mode = null_session ? test::FakeMode::HARMONIC : test::FakeMode::THROW_UNSUPPORTED;
        CalculationOptions options;
        options.SetMethod(Method("fake-method"));
        options.SetEngine("scan-fake-failure");
        std::vector<unsigned> chain;
        OEChem::OEMol mol = all_anti("CCCC", chain);
        const ScanResult scan = Calculation(options).Scan(mol, backbone_scan(chain, 30.0));
        // The seed's attempt fails, so it seeds nothing further.
        EXPECT_EQ(engine->sessions_created, 1);
        const ScanPoint& seed = point_at(scan, 180.0);
        EXPECT_EQ(seed.GetStatus(),
                  null_session ? ScanPointStatus::ENGINE_ERROR : ScanPointStatus::UNSUPPORTED);
        EXPECT_NE(seed.GetMessage().find(null_session ? "null session" : "cannot handle"),
                  std::string::npos)
            << seed.GetMessage();
        for (std::size_t i = 0; i + 1 < scan.GetNumPoints(); ++i) {
            EXPECT_EQ(scan.GetPoint(i).GetStatus(), ScanPointStatus::NOT_ATTEMPTED) << i;
        }
        EXPECT_EQ(scan.GetMolecule().NumConfs(), 0u);
    }
}

TEST(CalculationScanTest, AFailedAttemptKeepsNoEnergyEvenAfterTheOptimizerAcceptedIterates) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    // Grid -120, -60, 0, 60, 120, 180; the seed at 180 is session 1, and its first neighbor,
    // -120, session 2. That attempt starts 60 degrees from its constraint, so it has accepted
    // iterates when the engine fails.
    const auto flaky = std::make_shared<FlakyEngine>(2, 6);
    EngineRegistry::Instance().Register(flaky, true);
    CalculationOptions options = mmff94();
    options.SetEngine("scan-flaky");
    ScanOptions scan_options = backbone_scan(chain, 60.0);
    scan_options.SetMaxOptimizations(2);
    const ScanResult scan = Calculation(options).Scan(mol, scan_options);

    EXPECT_EQ(flaky->sessions_created, 2);
    EXPECT_EQ(flaky->evaluations_before_failure, 6);
    EXPECT_EQ(point_at(scan, 180.0).GetStatus(), ScanPointStatus::OK);
    // The failed attempt's last geometry need not satisfy the grid constraint, so the point
    // keeps neither its energy nor its geometry.
    const ScanPoint& failed = point_at(scan, -120.0);
    EXPECT_EQ(failed.GetStatus(), ScanPointStatus::ENGINE_ERROR);
    EXPECT_NE(failed.GetMessage().find("flaky engine"), std::string::npos) << failed.GetMessage();
    EXPECT_FALSE(failed.HasEnergy());
    EXPECT_FALSE(failed.HasRelativeEnergy());
    EXPECT_FALSE(failed.HasCoordinates());
    for (const double angle : {-60.0, 0.0, 60.0, 120.0}) {
        EXPECT_EQ(point_at(scan, angle).GetStatus(), ScanPointStatus::NOT_ATTEMPTED) << angle;
    }
    EXPECT_TRUE(scan.IsTruncated());
    const OEChem::OEMol profile = scan.GetMolecule();
    ASSERT_EQ(profile.NumConfs(), 1u);
    EXPECT_EQ(OEChem::OEGetSDData(*profile.GetActive(), "OEVibe.mmff94.gas.ScanAngles"),
              detail::format_value(180.0));
}

TEST(CalculationScanTest, HeldBondLengthsComeFromTheFirstConformerAtEveryGridPoint) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    const std::vector<double> relaxed = coordinates_of(*mol.GetActive());
    // The first conformer has its terminal bond stretched, which MMFF94 penalizes by a few
    // kcal/mol, so a length taken from the relaxed second conformer would win its grid point
    // and the held length would drift to the relaxed value.
    const double bond = distance_between(relaxed, chain[0], chain[1]);
    std::array<double, 3> stretch{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        stretch[axis] = 0.1 * (relaxed[3 * chain[0] + axis] - relaxed[3 * chain[1] + axis]) / bond;
    }
    std::vector<double> stretched = relaxed;
    translate(stretched, terminal_group(mol, chain[0], chain[1]), stretch);
    const double held = distance_between(stretched, chain[0], chain[1]);
    ASSERT_NEAR(held, bond + 0.1, 1e-9);
    OEChem::OEConfBase* second = mol.NewConf(mol.GetActive());
    ASSERT_EQ(mol.NumConfs(), 2u);
    mol.GetConf(OEChem::OEHasConfIdx(0))->SetCoords(stretched.data());
    second->SetCoords(relaxed.data());

    ConstraintSet hold;
    hold.AddDistance(chain[0], chain[1]);
    OptimizerOptions optimizer;
    optimizer.SetConstraints(hold);
    ScanOptions scan_options = backbone_scan(chain, 120.0);
    scan_options.SetOptimizerOptions(optimizer);
    const ScanResult scan = Calculation(mmff94()).Scan(mol, scan_options);
    ASSERT_EQ(scan.GetNumPoints(), 3u);
    for (std::size_t i = 0; i < scan.GetNumPoints(); ++i) {
        const ScanPoint& point = scan.GetPoint(i);
        ASSERT_TRUE(point.HasCoordinates()) << i << ": " << point.GetMessage();
        EXPECT_NEAR(distance_between(point.GetCoordinates(), chain[0], chain[1]), held, 5e-3) << i;
    }
}

TEST(CalculationScanTest, PositionRestraintsReferTheFirstConformerAtEveryGridPoint) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    std::vector<double> displaced = coordinates_of(*mol.GetActive());
    // The first conformer is gauche (grid point 60) and the second is anti and 3 Angstrom away
    // (grid point 180), so the second seed starts far from the first conformer's position.
    OEChem::OEConfBase* second = mol.NewConf(mol.GetActive());
    ASSERT_EQ(mol.NumConfs(), 2u);
    OEChem::OEConfBase* first = mol.GetConf(OEChem::OEHasConfIdx(0));
    OEChem::OESetTorsion(*first, first->GetAtom(OEChem::OEHasAtomIdx(chain[0])),
                         first->GetAtom(OEChem::OEHasAtomIdx(chain[1])),
                         first->GetAtom(OEChem::OEHasAtomIdx(chain[2])),
                         first->GetAtom(OEChem::OEHasAtomIdx(chain[3])), Units::PI / 3.0);
    std::vector<unsigned> everything;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        everything.push_back(atom->GetIdx());
    }
    translate(displaced, everything, {3.0, 0.0, 0.0});
    second->SetCoords(displaced.data());
    const std::vector<double> reference = coordinates_of(*first);

    ConstraintSet pin;
    pin.AddPositionRestraint(chain[0], 100.0);
    OptimizerOptions optimizer;
    optimizer.SetConstraints(pin);
    ScanOptions scan_options = backbone_scan(chain, 120.0);
    scan_options.SetOptimizerOptions(optimizer);
    const ScanResult scan = Calculation(mmff94()).Scan(mol, scan_options);
    ASSERT_EQ(scan.GetNumPoints(), 3u);
    for (std::size_t i = 0; i < scan.GetNumPoints(); ++i) {
        const ScanPoint& point = scan.GetPoint(i);
        ASSERT_TRUE(point.HasCoordinates()) << i << ": " << point.GetMessage();
        EXPECT_LT(distance_between(point.GetCoordinates(), chain[0], reference, chain[0]), 0.1)
            << i;
    }
}

TEST(CalculationScanTest, RejectedCallsLeaveTheMoleculeUnchanged) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    // Input preparation would add the ten hydrogens to the caller's molecule.
    OEChem::OESuppressHydrogens(mol);
    ASSERT_EQ(mol.NumAtoms(), 4u);
    const auto expect_unchanged = [&mol] {
        EXPECT_EQ(mol.NumAtoms(), 4u);
        EXPECT_TRUE(OEChem::OEHasImplicitHydrogens(mol));
    };
    // The scan options are validated before anything else, so none of these reaches Prepare.
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, ScanOptions()), OptionsError);
    expect_unchanged();
    ScanOptions lbfgs = backbone_scan(chain, 30.0);
    OptimizerOptions optimizer;
    optimizer.SetAlgorithm(Algorithm::LBFGS);
    lbfgs.SetOptimizerOptions(optimizer);
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, lbfgs), OptionsError);
    expect_unchanged();

    // Scan points carry no properties, whether requested or implied by partial-charge write-back.
    test::register_fake("scan-properties");
    CalculationOptions requested;
    requested.SetMethod(Method("fake-method"));
    requested.SetEngine("scan-properties");
    requested.SetProperties(static_cast<unsigned>(Property::DIPOLE));
    EXPECT_THROW(Calculation(requested).Scan(mol, backbone_scan(chain, 30.0)), OptionsError);
    expect_unchanged();
    CalculationOptions assigned;
    assigned.SetMethod(Method("fake-method"));
    assigned.SetEngine("scan-properties");
    assigned.SetAssignPartialCharges(true);
    EXPECT_THROW(Calculation(assigned).Scan(mol, backbone_scan(chain, 30.0)), OptionsError);
    expect_unchanged();
}

TEST(CalculationScanTest, AMoleculeWithoutConformersIsAnInputError) {
    std::vector<unsigned> chain;
    OEChem::OEMol mol = all_anti("CCCC", chain);
    mol.DeleteConfs();
    EXPECT_THROW(Calculation(mmff94()).Scan(mol, backbone_scan(chain, 30.0)), InputError);
}

TEST(ScanResultTest, FinalizeAndAccessorsFollowTheOptionalValueRules) {
    ScanResult scan;
    EXPECT_THROW(scan.GetDihedral(0), OEVibeError);
    EXPECT_THROW(scan.GetPoint(0), OEVibeError);
    EXPECT_EQ(scan.GetMolecule().NumConfs(), 0u);
    for (const double energy : {5.0, 2.0}) {
        ScanPoint point;
        point.SetAngles({energy * 10.0});
        point.SetEnergy(energy);
        point.SetStatus(ScanPointStatus::OK, "");
        scan.AddPoint(point);
    }
    ScanPoint empty;
    empty.SetRelativeEnergy(1.0);
    scan.AddPoint(empty);
    scan.Finalize();
    ASSERT_EQ(scan.GetNumPoints(), 3u);
    EXPECT_DOUBLE_EQ(scan.GetPoint(0).GetRelativeEnergy(), 3.0);
    EXPECT_DOUBLE_EQ(scan.GetPoint(1).GetRelativeEnergy(), 0.0);
    EXPECT_FALSE(scan.GetPoint(2).HasRelativeEnergy());
    EXPECT_EQ(scan.GetPoint(2).GetStatus(), ScanPointStatus::NOT_ATTEMPTED);
    EXPECT_THROW(scan.GetPoint(2).GetEnergy(), OEVibeError);
    EXPECT_THROW(scan.GetPoint(2).GetCoordinates(), OEVibeError);
    EXPECT_THROW(scan.GetPoint(3), OEVibeError);
    EXPECT_EQ(scan_point_status_name(ScanPointStatus::OK), "OK");
    EXPECT_EQ(scan_point_status_name(ScanPointStatus::NOT_CONVERGED), "NOT_CONVERGED");
    EXPECT_EQ(scan_point_status_name(ScanPointStatus::ENGINE_ERROR), "ENGINE_ERROR");
    EXPECT_EQ(scan_point_status_name(ScanPointStatus::UNSUPPORTED), "UNSUPPORTED");
    EXPECT_EQ(scan_point_status_name(ScanPointStatus::NOT_ATTEMPTED), "NOT_ATTEMPTED");
}
