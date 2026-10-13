#include <gtest/gtest.h>

#include <oechem.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Frequencies.h"
#include "core/Hessian.h"
#include "core/Optimizer.h"
#include "oevibe/Error.h"
#include "oevibe/engines/OEFFEngine.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

// CODATA 2018, restated here so the tests check the implementation against independent values.
constexpr double TEST_PI = 3.14159265358979323846;
constexpr double AVOGADRO = 6.02214076e23;         // 1/mol
constexpr double ATOMIC_MASS = 1.66053906660e-27;  // kg
constexpr double LIGHT_SPEED_CM = 2.99792458e10;   // cm/s
constexpr double JOULES_PER_KCAL = 4184.0;
constexpr double MASS_C = 12.0107;  // OEGetAverageWeight(6)
constexpr double MASS_O = 15.9994;  // OEGetAverageWeight(8)

/// Harmonic frequency (1 / (2 pi c)) sqrt(k / mu) in cm^-1, evaluated in SI units.
double oscillator_wavenumber(double force_constant, double reduced_mass_amu) {
    const double k_si = force_constant * JOULES_PER_KCAL / AVOGADRO / 1e-20;  // J/m^2
    return std::sqrt(k_si / (reduced_mass_amu * ATOMIC_MASS)) / (2.0 * TEST_PI * LIGHT_SPEED_CM);
}

/// Carbon at ``origin`` and oxygen 1.128 Angstrom away along ``direction``, held by a spring of
/// ``force_constant`` kcal/mol/Angstrom^2: H = k [[u u^T, -u u^T], [-u u^T, u u^T]].
struct Diatomic {
    std::vector<double> x;
    std::vector<double> hessian;
    Eigen::Vector3d axis;
};

Diatomic carbon_monoxide(double force_constant) {
    const Eigen::Vector3d origin(0.3, -1.2, 2.0);
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 2.0).normalized();
    const Eigen::Vector3d oxygen = origin + 1.128 * axis;
    Diatomic diatomic;
    diatomic.axis = axis;
    diatomic.x = {origin(0), origin(1), origin(2), oxygen(0), oxygen(1), oxygen(2)};
    diatomic.hessian.assign(36, 0.0);
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            const double sign = a == b ? 1.0 : -1.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    diatomic.hessian[(3 * a + i) * 6 + 3 * b + j] =
                        sign * force_constant * axis(i) * axis(j);
                }
            }
        }
    }
    return diatomic;
}

OEChem::OEMol molecule_from(const std::string& smiles, const std::vector<double>& xyz) {
    OEChem::OEMol mol;
    if (!OEChem::OESmilesToMol(mol, smiles) || 3 * mol.GetMaxAtomIdx() != xyz.size()) {
        throw std::runtime_error("cannot build " + smiles);
    }
    mol.SetCoords(xyz.data());
    mol.SetDimension(3);
    return mol;
}

std::shared_ptr<EngineSession> mmff94_session(const MolecularSystem& system) {
    OEFFEngine engine;
    EngineSettings settings;
    settings.method = Method(MMFF94);
    return engine.CreateSession(system, settings);
}

/// Minimizes far below the preset thresholds, so the analysis sees the minimum itself.
detail::OptimizationOutcome tight_minimum(EngineSession& session, const MolecularSystem& system) {
    detail::OptimizationProblem problem;
    problem.x0 = system.GetConformerCoordinates(0);
    problem.atomic_numbers = system.GetAtomicNumbers();
    problem.settings.algorithm = Algorithm::LBFGS;
    problem.settings.max_iterations = 2000;
    problem.settings.energy_change = 1e-8;
    problem.settings.gradient_norm = 1e-4;
    return detail::run_optimization(session, problem);
}

/// Water (H, O, H) started off its MMFF94 minimum.
const std::vector<double> WATER_START = {0.80, 0.62, 0.0, 0.0, 0.0, 0.0, -0.74, 0.57, 0.02};

std::vector<double> mass_weighted(const std::vector<double>& mode,
                                  const std::vector<double>& masses) {
    std::vector<double> weighted(mode.size());
    for (std::size_t k = 0; k < mode.size(); ++k) {
        weighted[k] = std::sqrt(masses[k / 3]) * mode[k];
    }
    return weighted;
}

std::vector<double> row(const std::vector<double>& matrix, std::size_t index, std::size_t width) {
    return {matrix.begin() + static_cast<std::ptrdiff_t>(index * width),
            matrix.begin() + static_cast<std::ptrdiff_t>((index + 1) * width)};
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double total = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        total += a[k] * b[k];
    }
    return total;
}

}  // namespace

TEST(FrequenciesTest, FrequencyFactorFollowsFromCodata2018) {
    const double expected = std::sqrt(JOULES_PER_KCAL / (AVOGADRO * 1e-20 * ATOMIC_MASS)) /
                            (2.0 * TEST_PI * LIGHT_SPEED_CM);
    EXPECT_NEAR(detail::FREQUENCY_FACTOR, expected, 1e-12 * expected);
}

TEST(FrequenciesTest, MassesAreAbundanceAveraged) {
    const std::vector<double> masses = detail::atomic_masses({1, 6, 8, 18});
    ASSERT_EQ(masses.size(), 4u);
    EXPECT_NEAR(masses[0], 1.00794, 1e-12);
    EXPECT_NEAR(masses[1], MASS_C, 1e-12);
    EXPECT_NEAR(masses[2], MASS_O, 1e-12);
    EXPECT_NEAR(masses[3], 39.948, 1e-12);
    EXPECT_THROW(detail::atomic_masses({0}), OEVibeError);
    EXPECT_THROW(detail::atomic_masses({119}), OEVibeError);
}

TEST(FrequenciesTest, DiatomicIsALinearHarmonicOscillator) {
    const double k = 1200.0;  // kcal/mol/Angstrom^2
    const Diatomic co = carbon_monoxide(k);
    const detail::FrequencyAnalysis analysis =
        detail::analyze_frequencies(co.hessian, co.x, {6, 8});
    const double mu = MASS_C * MASS_O / (MASS_C + MASS_O);
    ASSERT_EQ(analysis.frequencies.size(), 1u);
    EXPECT_TRUE(analysis.linear);
    EXPECT_EQ(analysis.imaginary_count, 0u);
    const double expected = oscillator_wavenumber(k, mu);
    EXPECT_NEAR(analysis.frequencies[0], expected, 1e-9 * expected);
    EXPECT_NEAR(analysis.total_mass, MASS_C + MASS_O, 1e-12);
    ASSERT_EQ(analysis.principal_moments.size(), 3u);
    EXPECT_NEAR(analysis.principal_moments[0], 0.0, 1e-10);
    EXPECT_NEAR(analysis.principal_moments[1], mu * 1.128 * 1.128, 1e-10);
    EXPECT_NEAR(analysis.principal_moments[2], mu * 1.128 * 1.128, 1e-10);

    // The stretch moves both atoms along the bond, opposite ways, with the centre of mass fixed.
    ASSERT_EQ(analysis.normal_modes.size(), 6u);
    const Eigen::Map<const Eigen::Vector3d> carbon(analysis.normal_modes.data());
    const Eigen::Map<const Eigen::Vector3d> oxygen(analysis.normal_modes.data() + 3);
    EXPECT_NEAR(carbon.squaredNorm() + oxygen.squaredNorm(), 1.0, 1e-12);
    EXPECT_LT(carbon.cross(co.axis).norm(), 1e-10);
    EXPECT_LT((MASS_C * carbon + MASS_O * oxygen).norm(), 1e-10);
    EXPECT_LT(carbon.dot(oxygen), 0.0);
}

TEST(FrequenciesTest, NegativeForceConstantGivesAnImaginaryMode) {
    const Diatomic co = carbon_monoxide(-1200.0);
    const detail::FrequencyAnalysis analysis =
        detail::analyze_frequencies(co.hessian, co.x, {6, 8});
    const double expected = oscillator_wavenumber(1200.0, MASS_C * MASS_O / (MASS_C + MASS_O));
    ASSERT_EQ(analysis.frequencies.size(), 1u);
    EXPECT_NEAR(analysis.frequencies[0], -expected, 1e-9 * expected);
    EXPECT_EQ(analysis.imaginary_count, 1u);
}

TEST(FrequenciesTest, SingleAtomHasNoModes) {
    const detail::FrequencyAnalysis analysis =
        detail::analyze_frequencies(std::vector<double>(9, 0.0), {1.0, 2.0, 3.0}, {18});
    EXPECT_TRUE(analysis.frequencies.empty());
    EXPECT_TRUE(analysis.normal_modes.empty());
    EXPECT_FALSE(analysis.linear);
    EXPECT_NEAR(analysis.total_mass, 39.948, 1e-12);
    EXPECT_EQ(analysis.principal_moments, (std::vector<double>{0.0, 0.0, 0.0}));
}

// Reference: OpenEye 2026.1.0 Python MMFF94 (oeff.OEMMFF), Newton-minimized from WATER_START to a
// gradient norm of 2e-13, central differences of the gradient with step 1e-4 Angstrom, analyzed
// by an independent numpy implementation (the reference probe in Step 1 of this task):
// 1581.562, 3722.520, 3782.451 cm^-1; moments 0.637488, 1.175023, 1.812511 amu Angstrom^2.
TEST(FrequenciesTest, WaterAtItsMmff94MinimumHasThreeRealModes) {
    const OEChem::OEMol mol = molecule_from("[H]O[H]", WATER_START);
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const detail::OptimizationOutcome minimum = tight_minimum(*session, system);
    ASSERT_TRUE(minimum.converged) << minimum.message;
    const std::vector<double> hessian = detail::compute_hessian(*session, minimum.x, true, 0.005);
    const detail::FrequencyAnalysis analysis =
        detail::analyze_frequencies(hessian, minimum.x, system.GetAtomicNumbers());
    EXPECT_FALSE(analysis.linear);
    EXPECT_EQ(analysis.imaginary_count, 0u);
    ASSERT_EQ(analysis.frequencies.size(), 3u);
    EXPECT_NEAR(analysis.frequencies[0], 1581.562, 0.1);
    EXPECT_NEAR(analysis.frequencies[1], 3722.520, 0.1);
    EXPECT_NEAR(analysis.frequencies[2], 3782.451, 0.1);
    EXPECT_NEAR(analysis.principal_moments[0], 0.637488, 1e-4);
    EXPECT_NEAR(analysis.principal_moments[1], 1.175023, 1e-4);
    EXPECT_NEAR(analysis.principal_moments[2], 1.812511, 1e-4);
}

// Reference: the same probe for O=C=O started linear with unequal bonds, with the step
// 0.005 Angstrom used here: 615.309 (twice), 1192.933, 2283.521 cm^-1. The finite-difference
// Hessian keeps the test independent of how OpenEye differentiates a bend at exactly 180 degrees.
TEST(FrequenciesTest, CarbonDioxideIsLinearWithFourModes) {
    const OEChem::OEMol mol =
        molecule_from("O=C=O", {-1.20, 0.0, 0.0, 0.0, 0.0, 0.0, 1.12, 0.0, 0.0});
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const detail::OptimizationOutcome minimum = tight_minimum(*session, system);
    ASSERT_TRUE(minimum.converged) << minimum.message;
    const std::vector<double> hessian =
        detail::finite_difference_hessian(*session, minimum.x, 0.005);
    const detail::FrequencyAnalysis analysis =
        detail::analyze_frequencies(hessian, minimum.x, system.GetAtomicNumbers());
    EXPECT_TRUE(analysis.linear);
    EXPECT_LT(analysis.principal_moments[0], detail::LINEAR_MOMENT_THRESHOLD);
    EXPECT_EQ(analysis.imaginary_count, 0u);
    ASSERT_EQ(analysis.frequencies.size(), 4u);
    EXPECT_NEAR(analysis.frequencies[0], 615.309, 0.1);
    EXPECT_NEAR(analysis.frequencies[1], 615.309, 0.1);
    EXPECT_NEAR(analysis.frequencies[0], analysis.frequencies[1], 1e-6);
    EXPECT_NEAR(analysis.frequencies[2], 1192.933, 0.1);
    EXPECT_NEAR(analysis.frequencies[3], 2283.521, 0.1);
    EXPECT_EQ(analysis.normal_modes.size(), 4u * 9u);
}

TEST(FrequenciesTest, RigidMotionLeavesFrequenciesUnchanged) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const std::vector<double>& x = system.GetConformerCoordinates(0);
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, -2.0, 0.5).normalized()).toRotationMatrix();
    const Eigen::Vector3d shift(3.0, -2.0, 5.0);
    std::vector<double> moved(x.size());
    for (std::size_t a = 0; a < x.size() / 3; ++a) {
        const Eigen::Vector3d p =
            rotation * Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]) + shift;
        moved[3 * a] = p(0);
        moved[3 * a + 1] = p(1);
        moved[3 * a + 2] = p(2);
    }
    // MMFF94 is invariant under rigid motion, so its Hessian at the moved geometry is the rotated
    // Hessian; the analysis must not depend on the frame.
    const detail::FrequencyAnalysis original = detail::analyze_frequencies(
        detail::compute_hessian(*session, x, true, 0.005), x, system.GetAtomicNumbers());
    const detail::FrequencyAnalysis transformed = detail::analyze_frequencies(
        detail::compute_hessian(*session, moved, true, 0.005), moved, system.GetAtomicNumbers());
    ASSERT_EQ(original.frequencies.size(), 21u);
    ASSERT_EQ(transformed.frequencies.size(), original.frequencies.size());
    for (std::size_t i = 0; i < original.frequencies.size(); ++i) {
        EXPECT_NEAR(transformed.frequencies[i], original.frequencies[i], 1e-4) << "mode " << i;
    }
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(transformed.principal_moments[k], original.principal_moments[k], 1e-9);
    }
}

TEST(FrequenciesTest, OneNegativeCurvatureDirectionGivesOneImaginaryMode) {
    const OEChem::OEMol mol = molecule_from("[H]O[H]", WATER_START);
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const detail::OptimizationOutcome minimum = tight_minimum(*session, system);
    ASSERT_TRUE(minimum.converged) << minimum.message;
    std::vector<double> hessian = detail::compute_hessian(*session, minimum.x, true, 0.005);
    const std::vector<double> masses = detail::atomic_masses(system.GetAtomicNumbers());
    const detail::FrequencyAnalysis real =
        detail::analyze_frequencies(hessian, minimum.x, system.GetAtomicNumbers());
    ASSERT_EQ(real.frequencies.size(), 3u);

    // Flip the bend's curvature: with q its unit mass-weighted vector and lambda its eigenvalue,
    // H - 2 lambda (M^1/2 q)(M^1/2 q)^T has eigenvalue -lambda along q and leaves the others.
    std::vector<double> q = mass_weighted(row(real.normal_modes, 0, 9), masses);
    const double norm = std::sqrt(dot(q, q));
    for (double& value : q) {
        value /= norm;
    }
    const double lambda = std::pow(real.frequencies[0] / detail::FREQUENCY_FACTOR, 2);
    const std::vector<double> direction = mass_weighted(q, masses);
    for (std::size_t i = 0; i < 9; ++i) {
        for (std::size_t j = 0; j < 9; ++j) {
            hessian[i * 9 + j] -= 2.0 * lambda * direction[i] * direction[j];
        }
    }
    const detail::FrequencyAnalysis flipped =
        detail::analyze_frequencies(hessian, minimum.x, system.GetAtomicNumbers());
    ASSERT_EQ(flipped.frequencies.size(), 3u);
    EXPECT_EQ(flipped.imaginary_count, 1u);
    EXPECT_NEAR(flipped.frequencies[0], -real.frequencies[0], 1e-6 * real.frequencies[0]);
    EXPECT_NEAR(flipped.frequencies[1], real.frequencies[1], 1e-6 * real.frequencies[1]);
    EXPECT_NEAR(flipped.frequencies[2], real.frequencies[2], 1e-6 * real.frequencies[2]);
}

TEST(FrequenciesTest, NormalModesAreUnitAndMassWeightedOrthogonal) {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    auto session = mmff94_session(system);
    const std::vector<double>& x = system.GetConformerCoordinates(0);
    const detail::FrequencyAnalysis analysis = detail::analyze_frequencies(
        detail::compute_hessian(*session, x, true, 0.005), x, system.GetAtomicNumbers());
    const std::vector<double> masses = detail::atomic_masses(system.GetAtomicNumbers());
    const std::size_t dim = x.size();
    const std::size_t modes = analysis.frequencies.size();
    ASSERT_EQ(modes, dim - 6);
    ASSERT_EQ(analysis.normal_modes.size(), modes * dim);
    EXPECT_FALSE(analysis.linear);
    EXPECT_TRUE(std::is_sorted(analysis.frequencies.begin(), analysis.frequencies.end()));

    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (std::size_t a = 0; a < masses.size(); ++a) {
        center += masses[a] * Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]);
    }
    center /= analysis.total_mass;
    for (std::size_t i = 0; i < modes; ++i) {
        const std::vector<double> mode = row(analysis.normal_modes, i, dim);
        EXPECT_NEAR(dot(mode, mode), 1.0, 1e-12) << "mode " << i;
        const std::vector<double> weighted_i = mass_weighted(mode, masses);
        for (std::size_t j = 0; j < i; ++j) {
            const std::vector<double> weighted_j =
                mass_weighted(row(analysis.normal_modes, j, dim), masses);
            const double overlap =
                dot(weighted_i, weighted_j) /
                std::sqrt(dot(weighted_i, weighted_i) * dot(weighted_j, weighted_j));
            EXPECT_NEAR(overlap, 0.0, 1e-10) << "modes " << i << ", " << j;
        }
        // Vibrations carry no linear or angular momentum about the centre of mass.
        Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
        Eigen::Vector3d angular = Eigen::Vector3d::Zero();
        for (std::size_t a = 0; a < masses.size(); ++a) {
            const Eigen::Vector3d step(mode[3 * a], mode[3 * a + 1], mode[3 * a + 2]);
            const Eigen::Vector3d p =
                Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]) - center;
            momentum += masses[a] * step;
            angular += masses[a] * p.cross(step);
        }
        EXPECT_LT(momentum.norm(), 1e-10) << "mode " << i;
        EXPECT_LT(angular.norm(), 1e-10) << "mode " << i;
    }
}

TEST(FrequenciesTest, InvalidInputsRaise) {
    const Diatomic co = carbon_monoxide(1200.0);
    EXPECT_THROW(detail::analyze_frequencies(co.hessian, co.x, {6}), OEVibeError);
    EXPECT_THROW(detail::analyze_frequencies({1.0, 2.0}, co.x, {6, 8}), OEVibeError);
    std::vector<double> broken = co.hessian;
    broken[7] = std::nan("");
    EXPECT_THROW(detail::analyze_frequencies(broken, co.x, {6, 8}), OEVibeError);
    EXPECT_THROW(detail::analyze_frequencies(co.hessian, co.x, {6, 0}), OEVibeError);
    EXPECT_THROW(detail::analyze_frequencies({}, {}, {}), OEVibeError);
    EXPECT_THROW(detail::analyze_frequencies(co.hessian, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, {6, 8}),
                 OEVibeError);
}

// With every atom at one point the inertia tensor vanishes, so no rotation vector can be
// normalized; the analysis must say so rather than hand NaNs to the eigensolver.
TEST(FrequenciesTest, CoincidentAtomsRaiseInsteadOfReturningNaN) {
    const Diatomic co = carbon_monoxide(1200.0);
    EXPECT_THROW(detail::analyze_frequencies(co.hessian, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, {6, 8}),
                 OEVibeError);
    EXPECT_THROW(detail::analyze_frequencies(std::vector<double>(81, 1.0),
                                             std::vector<double>(9, 0.5), {1, 6, 8}),
                 OEVibeError);
}

// The floor is on the rotational moment, not a test for exactly equal coordinates: atoms 0.01
// Angstrom apart are as unusable as coincident ones (moment 7e-5 amu Angstrom^2).
TEST(FrequenciesTest, NearlyCoincidentAtomsRaise) {
    Diatomic co = carbon_monoxide(1200.0);
    for (int i = 0; i < 3; ++i) {
        co.x[3 + i] = co.x[i] + 0.01 * co.axis(i);
    }
    EXPECT_THROW(detail::analyze_frequencies(co.hessian, co.x, {6, 8}), OEVibeError);
}

// Every entry is finite, but mass weighting and symmetrizing overflow; the result must not be NaN.
TEST(FrequenciesTest, OverflowingHessianRaisesInsteadOfReturningNaN) {
    const Diatomic co = carbon_monoxide(1200.0);
    EXPECT_THROW(detail::analyze_frequencies(std::vector<double>(36, 1e308), co.x, {6, 8}),
                 OEVibeError);
}
