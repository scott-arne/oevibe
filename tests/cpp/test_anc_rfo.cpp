#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <vector>

#include "core/AncRfo.h"
#include "core/Geometry.h"
#include "oevibe/Units.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

detail::MinimizerSettings tight(unsigned iterations = 500) {
    detail::MinimizerSettings s;
    s.max_iterations = iterations;
    s.energy_change = 1e-8;
    s.gradient_norm = 1e-3;
    return s;
}

detail::ObjectiveFn springs(const std::vector<double>& reference) {
    return [reference](const std::vector<double>& x, std::vector<double>& g) {
        return test::spring_energy(reference, x, 50.0, &g);
    };
}

detail::AncRfoProblem carbons(std::size_t atoms) {
    detail::AncRfoProblem problem;
    problem.atomic_numbers.assign(atoms, 6);
    return problem;
}

}  // namespace

TEST(RfoStepTest, ApproachesTheNewtonStepForSmallGradients) {
    Eigen::MatrixXd hessian(2, 2);
    hessian << 2.0, 0.0, 0.0, 4.0;
    const Eigen::VectorXd gradient = Eigen::Vector2d(2e-4, 4e-4);
    const detail::RfoStep step = detail::rfo_step(hessian, gradient, 0.3);
    EXPECT_NEAR(step.step(0), -1e-4, 1e-8);
    EXPECT_NEAR(step.step(1), -1e-4, 1e-8);
    EXPECT_LT(step.predicted_change, 0.0);
}

TEST(RfoStepTest, TrustRadiusCapsTheStep) {
    const Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(2, 2);
    const detail::RfoStep step = detail::rfo_step(hessian, Eigen::Vector2d(50.0, -30.0), 0.2);
    EXPECT_NEAR(step.step.norm(), 0.2, 1e-12);
    EXPECT_LT(step.predicted_change, 0.0);
}

TEST(BfgsUpdateTest, SatisfiesTheSecantCondition) {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::Vector2d s(1.0, 0.0);
    const Eigen::Vector2d y(2.0, 0.5);
    detail::bfgs_update(hessian, s, y);
    EXPECT_LT((hessian * s - y).norm(), 1e-12);
}

TEST(BfgsUpdateTest, SkipsNegativeCurvature) {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(2, 2);
    detail::bfgs_update(hessian, Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(-1.0, 0.0));
    EXPECT_LT((hessian - Eigen::MatrixXd::Identity(2, 2)).norm(), 1e-15);
}

TEST(AncRfoTest, MinimizesASpringNetwork) {
    const std::vector<double> reference = test::sample_geometry();
    const detail::MinimizerResult result = detail::anc_rfo_minimize(
        springs(reference), test::perturbed(reference, 0.15), carbons(6), tight());
    ASSERT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.energy, 1e-5);
    EXPECT_LT(result.iterations, 100u);
}

TEST(AncRfoTest, FrozenAtomsNeverMove) {
    const std::vector<double> reference = test::sample_geometry();
    const std::vector<double> start = test::perturbed(reference, 0.15);
    detail::AncRfoProblem problem = carbons(6);
    problem.frozen = {0, 3};
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(springs(reference), start, problem, tight());
    ASSERT_TRUE(result.converged) << result.message;
    for (unsigned atom : {0u, 3u}) {
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(result.x[3 * atom + axis], start[3 * atom + axis], 1e-12);
        }
    }
}

// An energy term tied to absolute position, like a position restraint, is only minimized when
// the coordinates include translation.
TEST(AncRfoTest, TranslatesTheMoleculeWhenTheObjectiveIsNotRigidInvariant) {
    const std::vector<double> reference = test::sample_geometry();
    std::vector<double> start = reference;
    for (std::size_t k = 0; k < start.size(); k += 3) {
        start[k] += 0.3;
        start[k + 1] -= 0.2;
    }
    const double anchor = 10.0;  // kcal/mol/Angstrom^2 on atom 0
    const detail::ObjectiveFn objective = [reference, anchor](const std::vector<double>& x,
                                                              std::vector<double>& g) {
        double energy = test::spring_energy(reference, x, 50.0, &g);
        for (int axis = 0; axis < 3; ++axis) {
            const double d = x[axis] - reference[axis];
            energy += 0.5 * anchor * d * d;
            g[axis] += anchor * d;
        }
        return energy;
    };
    detail::AncRfoProblem problem = carbons(6);
    problem.rigid_invariant = false;
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(objective, start, problem, tight());
    ASSERT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.energy, 1e-5);
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(result.x[axis], reference[axis], 1e-3);
    }
}

TEST(AncRfoTest, SatisfiesAnUnsatisfiedDistanceConstraint) {
    const std::vector<double> reference = test::sample_geometry();
    detail::AncRfoProblem problem = carbons(6);
    detail::ResolvedConstraint bond;
    bond.kind = ConstraintKind::DISTANCE;
    bond.atoms = {0, 1};
    bond.target = detail::distance(reference.data(), 0, 1, nullptr) + 0.3;
    problem.constraints = {bond};
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(springs(reference), reference, problem, tight());
    ASSERT_TRUE(result.converged) << result.message;
    EXPECT_NEAR(detail::distance(result.x.data(), 0, 1, nullptr), bond.target, 1e-4);
    EXPECT_LE(result.max_distance_residual, 1e-4);
}

// Compressing a bond makes the constraint curvature (-lambda * Hessian of c) significant, which
// a BFGS update fed with the plain energy gradient cannot learn.
TEST(AncRfoTest, SatisfiesACompressedDistanceConstraint) {
    const std::vector<double> reference = test::sample_geometry();
    detail::AncRfoProblem problem = carbons(6);
    detail::ResolvedConstraint bond;
    bond.kind = ConstraintKind::DISTANCE;
    bond.atoms = {0, 1};
    bond.target = detail::distance(reference.data(), 0, 1, nullptr) - 0.3;
    problem.constraints = {bond};
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(springs(reference), reference, problem, tight());
    ASSERT_TRUE(result.converged) << result.message;
    EXPECT_LE(result.max_distance_residual, detail::DISTANCE_RESIDUAL_TOLERANCE);
    EXPECT_NEAR(detail::distance(result.x.data(), 0, 1, nullptr), bond.target,
                detail::DISTANCE_RESIDUAL_TOLERANCE);
}

TEST(AncRfoTest, ConvergesImmediatelyWhenFrozenAtomsAlreadySatisfyTheConstraints) {
    const std::vector<double> reference = test::sample_geometry();
    const std::vector<double> start = test::perturbed(reference, 0.1);
    detail::AncRfoProblem problem = carbons(6);
    problem.frozen = {0, 1, 2, 3, 4, 5};
    detail::ResolvedConstraint bond;
    bond.kind = ConstraintKind::DISTANCE;
    bond.atoms = {0, 1};
    bond.target = detail::distance(start.data(), 0, 1, nullptr);
    problem.constraints = {bond};
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(springs(reference), start, problem, tight());
    EXPECT_TRUE(result.converged) << result.message;
    EXPECT_EQ(result.iterations, 0u);
}

TEST(AncRfoTest, ReportsFrozenAtomsThatCannotSatisfyTheConstraints) {
    const std::vector<double> reference = test::sample_geometry();
    detail::AncRfoProblem problem = carbons(6);
    problem.frozen = {0, 1, 2, 3, 4, 5};
    detail::ResolvedConstraint bond;
    bond.kind = ConstraintKind::DISTANCE;
    bond.atoms = {0, 1};
    bond.target = detail::distance(reference.data(), 0, 1, nullptr) + 0.3;
    problem.constraints = {bond};
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(springs(reference), reference, problem, tight());
    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.message.find("constraints"), std::string::npos) << result.message;
    EXPECT_EQ(result.x, reference);
}

TEST(AncRfoTest, HoldsADihedralConstraint) {
    const std::vector<double> reference = test::sample_geometry();
    detail::AncRfoProblem problem = carbons(6);
    detail::ResolvedConstraint torsion;
    torsion.kind = ConstraintKind::DIHEDRAL;
    torsion.atoms = {0, 1, 2, 3};
    torsion.target = detail::wrap_angle(detail::dihedral(reference.data(), 0, 1, 2, 3, nullptr) +
                                        30.0 * Units::DEGREES_TO_RADIANS);
    problem.constraints = {torsion};
    const detail::MinimizerResult result =
        detail::anc_rfo_minimize(springs(reference), reference, problem, tight());
    ASSERT_TRUE(result.converged) << result.message;
    const double achieved = detail::dihedral(result.x.data(), 0, 1, 2, 3, nullptr);
    EXPECT_LE(std::abs(detail::wrap_angle(achieved - torsion.target)) * Units::RADIANS_TO_DEGREES,
              1e-3);
}

TEST(AncRfoTest, StopsAtTheIterationLimit) {
    const std::vector<double> reference = test::sample_geometry();
    const detail::MinimizerResult result = detail::anc_rfo_minimize(
        springs(reference), test::perturbed(reference, 0.3), carbons(6), tight(2));
    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.message.find("iteration"), std::string::npos);
}
