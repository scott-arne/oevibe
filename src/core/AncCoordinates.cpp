#include "core/AncCoordinates.h"

namespace OEVIBE {
namespace detail {

AncBasis build_anc_basis(const Eigen::MatrixXd& hessian, const std::vector<double>& x,
                         const std::vector<unsigned>& frozen) {
    const auto dim = static_cast<Eigen::Index>(x.size());
    const Eigen::Index atoms = dim / 3;
    std::vector<Eigen::VectorXd> excluded;
    if (frozen.empty()) {
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        for (Eigen::Index a = 0; a < atoms; ++a) {
            center += Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]);
        }
        center /= static_cast<double>(atoms);
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::VectorXd shift = Eigen::VectorXd::Zero(dim);
            Eigen::VectorXd turn = Eigen::VectorXd::Zero(dim);
            for (Eigen::Index a = 0; a < atoms; ++a) {
                shift(3 * a + axis) = 1.0;
                const Eigen::Vector3d p =
                    Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]) - center;
                turn.segment<3>(3 * a) = Eigen::Vector3d::Unit(axis).cross(p);
            }
            excluded.push_back(shift);
            excluded.push_back(turn);
        }
    } else {
        for (unsigned atom : frozen) {
            for (int axis = 0; axis < 3; ++axis) {
                Eigen::VectorXd unit = Eigen::VectorXd::Zero(dim);
                unit(3 * static_cast<Eigen::Index>(atom) + axis) = 1.0;
                excluded.push_back(unit);
            }
        }
    }
    Eigen::MatrixXd span(dim, static_cast<Eigen::Index>(excluded.size()));
    for (std::size_t c = 0; c < excluded.size(); ++c) {
        span.col(static_cast<Eigen::Index>(c)) = excluded[c];
    }
    // Column-pivoted QR finds the excluded rank, which drops to 5 for a linear molecule.
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(span);
    qr.setThreshold(1e-8);
    const Eigen::Index rank = qr.rank();
    const Eigen::MatrixXd q = qr.householderQ();
    const Eigen::MatrixXd complement = q.rightCols(dim - rank);

    AncBasis basis;
    if (complement.cols() == 0) {
        basis.vectors = Eigen::MatrixXd(dim, 0);
        basis.eigenvalues = Eigen::VectorXd(0);
        return basis;
    }
    const Eigen::MatrixXd reduced = complement.transpose() * hessian * complement;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reduced);
    basis.vectors = complement * solver.eigenvectors();
    basis.eigenvalues = solver.eigenvalues().cwiseMax(MIN_ANC_EIGENVALUE);
    return basis;
}

}  // namespace detail
}  // namespace OEVIBE
