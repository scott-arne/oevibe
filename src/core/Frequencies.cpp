#include "core/Frequencies.h"

#include <oechem.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {
namespace detail {

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// The squared norm of the mass-weighted rotation about a principal axis equals that axis's
// moment, so a rotation can only be normalized when its moment is clearly above zero. The floor
// is the moment below which an axis counts as vanishing (the linear case), which makes the two
// decisions agree: an axis is either a real rotation or the one a linear molecule loses, and a
// geometry that would need a second vanishing axis, with every atom within a few hundredths of an
// Angstrom of one point, has no meaningful rotations and is rejected instead.
constexpr double MIN_ROTATION_MOMENT = LINEAR_MOMENT_THRESHOLD;  // amu Angstrom^2

bool all_finite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

}  // namespace

std::vector<double> atomic_masses(const std::vector<unsigned>& atomic_numbers) {
    std::vector<double> masses;
    masses.reserve(atomic_numbers.size());
    for (unsigned z : atomic_numbers) {
        // OpenEye returns 0 for an atomic number it has no weight for.
        const double mass = OEChem::OEGetAverageWeight(z);
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw OEVibeError("no standard atomic weight for atomic number " + std::to_string(z));
        }
        masses.push_back(mass);
    }
    return masses;
}

FrequencyAnalysis analyze_frequencies(const std::vector<double>& hessian,
                                      const std::vector<double>& x,
                                      const std::vector<unsigned>& atomic_numbers) {
    const std::size_t num_atoms = atomic_numbers.size();
    const std::size_t dim = 3 * num_atoms;
    if (num_atoms == 0 || x.size() != dim || hessian.size() != dim * dim) {
        throw OEVibeError("frequency analysis needs 3N coordinates and a 3N x 3N Hessian; got " +
                          std::to_string(num_atoms) + " atoms, " + std::to_string(x.size()) +
                          " coordinates, and " + std::to_string(hessian.size()) +
                          " Hessian values");
    }
    if (!all_finite(x) || !all_finite(hessian)) {
        throw OEVibeError("frequency analysis needs finite coordinates and Hessian values");
    }
    const std::vector<double> masses = atomic_masses(atomic_numbers);
    const auto n = static_cast<Eigen::Index>(num_atoms);
    const auto d = static_cast<Eigen::Index>(dim);

    FrequencyAnalysis analysis;
    double total_mass = 0.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (Eigen::Index a = 0; a < n; ++a) {
        const double m = masses[static_cast<std::size_t>(a)];
        center += m * Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]);
        total_mass += m;
    }
    center /= total_mass;
    std::vector<Eigen::Vector3d> relative(num_atoms);
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for (Eigen::Index a = 0; a < n; ++a) {
        const auto atom = static_cast<std::size_t>(a);
        relative[atom] = Eigen::Vector3d(x[3 * a], x[3 * a + 1], x[3 * a + 2]) - center;
        const Eigen::Vector3d& p = relative[atom];
        inertia +=
            masses[atom] * (p.squaredNorm() * Eigen::Matrix3d::Identity() - p * p.transpose());
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> principal(inertia);
    if (principal.info() != Eigen::Success) {
        throw OEVibeError("frequency analysis could not diagonalize the inertia tensor");
    }
    const Eigen::Vector3d moments = principal.eigenvalues();  // ascending
    const Eigen::Matrix3d axes = principal.eigenvectors();
    analysis.total_mass = total_mass;
    // Rounding can leave the vanishing moment of a linear molecule slightly negative.
    analysis.principal_moments = {std::max(0.0, moments(0)), moments(1), moments(2)};
    analysis.linear = num_atoms >= 2 && moments(0) < LINEAR_MOMENT_THRESHOLD;

    // Rigid-body directions in mass-weighted coordinates. Translations, and rotations about the
    // principal axes through the centre of mass, are mutually orthogonal there (the cross terms
    // are the total momentum about the centre of mass and the off-diagonal inertia products,
    // both zero), so normalizing each gives an orthonormal set. A linear molecule loses the
    // rotation about its axis and a single atom every rotation; deciding this from the moments
    // keeps the mode count consistent with ``linear``.
    std::vector<Eigen::VectorXd> rigid;
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd shift = Eigen::VectorXd::Zero(d);
        for (Eigen::Index a = 0; a < n; ++a) {
            shift(3 * a + axis) = std::sqrt(masses[static_cast<std::size_t>(a)]);
        }
        rigid.push_back(shift / std::sqrt(total_mass));
    }
    for (int k = 0; k < 3 && num_atoms >= 2; ++k) {
        if (analysis.linear && k == 0) {
            continue;
        }
        Eigen::VectorXd turn = Eigen::VectorXd::Zero(d);
        for (Eigen::Index a = 0; a < n; ++a) {
            const auto atom = static_cast<std::size_t>(a);
            turn.segment<3>(3 * a) = std::sqrt(masses[atom]) * axes.col(k).cross(relative[atom]);
        }
        const double moment = turn.squaredNorm();
        if (!std::isfinite(moment) || moment < MIN_ROTATION_MOMENT) {
            throw OEVibeError(
                "frequency analysis needs a geometry that can rotate: the rotation about "
                "principal axis " +
                std::to_string(k) + " has moment " + std::to_string(moment) +
                " amu Angstrom^2, below " + std::to_string(MIN_ROTATION_MOMENT) +
                " (coincident or nearly coincident atoms)");
        }
        rigid.push_back(turn / std::sqrt(moment));
    }
    const auto num_rigid = static_cast<Eigen::Index>(rigid.size());
    const Eigen::Index num_modes = d - num_rigid;
    if (num_modes == 0) {
        return analysis;
    }

    // Completing the orthonormal rigid-body columns with a Householder QR gives an orthonormal
    // basis of the vibrational space in the trailing columns. With P its projector, diagonalizing
    // the mass-weighted Hessian in that basis is diagonalizing P H' P on the range of P.
    Eigen::MatrixXd span(d, num_rigid);
    for (Eigen::Index c = 0; c < num_rigid; ++c) {
        span.col(c) = rigid[static_cast<std::size_t>(c)];
    }
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(span);
    const Eigen::MatrixXd q = qr.householderQ();
    const Eigen::MatrixXd vibrational = q.rightCols(num_modes);

    Eigen::VectorXd inverse_sqrt_mass(d);
    for (Eigen::Index a = 0; a < n; ++a) {
        inverse_sqrt_mass.segment<3>(3 * a).setConstant(
            1.0 / std::sqrt(masses[static_cast<std::size_t>(a)]));
    }
    const Eigen::Map<const RowMajorMatrix> cartesian(hessian.data(), d, d);
    const Eigen::MatrixXd weighted = inverse_sqrt_mass.asDiagonal() *
                                     (0.5 * (cartesian + cartesian.transpose())) *
                                     inverse_sqrt_mass.asDiagonal();
    const Eigen::MatrixXd reduced = vibrational.transpose() * weighted * vibrational;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reduced);
    if (solver.info() != Eigen::Success) {
        throw OEVibeError("frequency analysis could not diagonalize the projected Hessian");
    }
    const Eigen::MatrixXd modes = vibrational * solver.eigenvectors();

    analysis.frequencies.reserve(static_cast<std::size_t>(num_modes));
    analysis.normal_modes.reserve(static_cast<std::size_t>(num_modes * d));
    for (Eigen::Index i = 0; i < num_modes; ++i) {
        const double lambda = solver.eigenvalues()(i);
        const double frequency =
            std::copysign(std::sqrt(std::abs(lambda)), lambda) * FREQUENCY_FACTOR;
        analysis.frequencies.push_back(frequency);
        if (frequency < 0.0) {
            ++analysis.imaginary_count;
        }
        const Eigen::VectorXd displacement =
            inverse_sqrt_mass.cwiseProduct(modes.col(i)).normalized();
        analysis.normal_modes.insert(analysis.normal_modes.end(), displacement.data(),
                                     displacement.data() + d);
    }
    // Finite inputs can still overflow while mass weighting; downstream thermochemistry must never
    // receive NaN or infinite frequencies.
    if (!all_finite(analysis.frequencies) || !all_finite(analysis.normal_modes)) {
        throw OEVibeError("frequency analysis produced values that are not finite");
    }
    return analysis;
}

}  // namespace detail
}  // namespace OEVIBE
