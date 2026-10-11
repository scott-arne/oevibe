#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <vector>

#include "core/AncCoordinates.h"
#include "core/ModelHessian.h"
#include "oevibe/MolecularSystem.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

struct Molecule {
    std::vector<unsigned> numbers;
    std::vector<double> x;
};

Molecule ethanol() {
    const OEChem::OEMol mol = test::make_conformers("CCO");
    const MolecularSystem system(mol, 0, 1);
    return {system.GetAtomicNumbers(), system.GetConformerCoordinates(0)};
}

Eigen::VectorXd translation(std::size_t atoms, int axis) {
    Eigen::VectorXd t = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3 * atoms));
    for (std::size_t a = 0; a < atoms; ++a) {
        t(static_cast<Eigen::Index>(3 * a + axis)) = 1.0;
    }
    return t;
}

Eigen::VectorXd rotation(const std::vector<double>& x, int axis) {
    const std::size_t atoms = x.size() / 3;
    Eigen::VectorXd r = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(x.size()));
    for (std::size_t a = 0; a < atoms; ++a) {
        const Eigen::Vector3d p(x[3 * a], x[3 * a + 1], x[3 * a + 2]);
        r.segment<3>(static_cast<Eigen::Index>(3 * a)) = Eigen::Vector3d::Unit(axis).cross(p);
    }
    return r;
}

}  // namespace

TEST(ModelHessianTest, SymmetricSemidefiniteAndRigidInvariant) {
    const Molecule mol = ethanol();
    const Eigen::MatrixXd h = detail::lindh_model_hessian(mol.numbers, mol.x);
    ASSERT_EQ(h.rows(), static_cast<Eigen::Index>(mol.x.size()));
    EXPECT_LT((h - h.transpose()).cwiseAbs().maxCoeff(), 1e-9);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h);
    EXPECT_GT(solver.eigenvalues().minCoeff(), -1e-6);
    const double scale = h.cwiseAbs().maxCoeff();
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_LT((h * translation(mol.numbers.size(), axis)).norm(), 1e-8 * scale);
        EXPECT_LT((h * rotation(mol.x, axis)).norm(), 1e-6 * scale);
    }
}

TEST(AncBasisTest, NonlinearMoleculeKeeps3NMinus6Coordinates) {
    const Molecule mol = ethanol();
    const detail::AncBasis basis =
        detail::build_anc_basis(detail::lindh_model_hessian(mol.numbers, mol.x), mol.x, {});
    const Eigen::Index dim = static_cast<Eigen::Index>(mol.x.size());
    ASSERT_EQ(basis.vectors.cols(), dim - 6);
    const Eigen::MatrixXd gram = basis.vectors.transpose() * basis.vectors;
    EXPECT_LT((gram - Eigen::MatrixXd::Identity(dim - 6, dim - 6)).cwiseAbs().maxCoeff(), 1e-10);
    EXPECT_LT((basis.vectors.transpose() * translation(mol.numbers.size(), 0)).norm(), 1e-10);
    EXPECT_GE(basis.eigenvalues.minCoeff(), detail::MIN_ANC_EIGENVALUE);
}

TEST(AncBasisTest, LinearMoleculeKeeps3NMinus5Coordinates) {
    const std::vector<unsigned> numbers = {8, 6, 8};
    const std::vector<double> x = {-1.16, 0.0, 0.0, 0.0, 0.0, 0.0, 1.16, 0.0, 0.0};
    const detail::AncBasis basis =
        detail::build_anc_basis(detail::lindh_model_hessian(numbers, x), x, {});
    EXPECT_EQ(basis.vectors.cols(), 4);
}

TEST(AncBasisTest, FrozenAtomsAreExcluded) {
    const Molecule mol = ethanol();
    const detail::AncBasis basis =
        detail::build_anc_basis(detail::lindh_model_hessian(mol.numbers, mol.x), mol.x, {0});
    const Eigen::Index dim = static_cast<Eigen::Index>(mol.x.size());
    ASSERT_EQ(basis.vectors.cols(), dim - 3);
    EXPECT_LT(basis.vectors.topRows(3).cwiseAbs().maxCoeff(), 1e-12);
}
