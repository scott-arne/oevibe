#include "core/ModelHessian.h"

#include <cmath>

#include "core/Geometry.h"
#include "oevibe/Units.h"

namespace OEVIBE {
namespace detail {

namespace {

// Lindh et al. (1995), Table 1: exponents (Bohr^-2) and reference distances (Bohr) by
// periodic-table row; rows beyond the third use the third-row values.
constexpr double ALPHA[3][3] = {
    {1.0000, 0.3949, 0.3949}, {0.3949, 0.2800, 0.2800}, {0.3949, 0.2800, 0.2800}};
constexpr double R_REF[3][3] = {{1.35, 2.10, 2.53}, {2.10, 2.87, 3.40}, {2.53, 3.40, 3.40}};
constexpr double K_STRETCH = 0.45;   // Hartree/Bohr^2
constexpr double K_BEND = 0.15;      // Hartree/rad^2
constexpr double K_TORSION = 0.005;  // Hartree/rad^2
// Products of pair weights below this contribute negligibly and are skipped for speed.
constexpr double WEIGHT_CUTOFF = 1e-4;

int lindh_row(unsigned atomic_number) {
    if (atomic_number <= 2) {
        return 0;
    }
    return atomic_number <= 10 ? 1 : 2;
}

void add_term(Eigen::MatrixXd& hessian, double force_constant, const unsigned* atoms, int count,
              const double* gradient) {
    for (int s = 0; s < count; ++s) {
        for (int t = 0; t < count; ++t) {
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    hessian(3 * atoms[s] + p, 3 * atoms[t] + q) +=
                        force_constant * gradient[3 * s + p] * gradient[3 * t + q];
                }
            }
        }
    }
}

}  // namespace

Eigen::MatrixXd lindh_model_hessian(const std::vector<unsigned>& atomic_numbers,
                                    const std::vector<double>& x) {
    const auto n = static_cast<unsigned>(atomic_numbers.size());
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(3 * n, 3 * n);
    Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(n, n);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = i + 1; j < n; ++j) {
            const double r = distance(x.data(), i, j, nullptr) * Units::ANGSTROM_TO_BOHR;
            const int a = lindh_row(atomic_numbers[i]);
            const int b = lindh_row(atomic_numbers[j]);
            rho(i, j) = rho(j, i) = std::exp(ALPHA[a][b] * (R_REF[a][b] * R_REF[a][b] - r * r));
        }
    }
    const double stretch_scale =
        Units::HARTREE_TO_KCAL * Units::ANGSTROM_TO_BOHR * Units::ANGSTROM_TO_BOHR;
    const double angular_scale = Units::HARTREE_TO_KCAL;
    double gradient[12];

    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = i + 1; j < n; ++j) {
            if (rho(i, j) <= WEIGHT_CUTOFF) {
                continue;
            }
            distance(x.data(), i, j, gradient);
            const unsigned atoms[2] = {i, j};
            add_term(hessian, K_STRETCH * rho(i, j) * stretch_scale, atoms, 2, gradient);
        }
    }
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned k = i + 1; k < n; ++k) {
                if (i == j || k == j) {
                    continue;
                }
                const double weight = rho(i, j) * rho(j, k);
                if (weight <= WEIGHT_CUTOFF) {
                    continue;
                }
                const double theta = angle(x.data(), i, j, k, gradient);
                if (std::sin(theta) < 1e-3) {
                    continue;
                }
                const unsigned atoms[3] = {i, j, k};
                add_term(hessian, K_BEND * weight * angular_scale, atoms, 3, gradient);
            }
        }
    }
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned k = j + 1; k < n; ++k) {
            if (rho(j, k) <= WEIGHT_CUTOFF) {
                continue;
            }
            for (unsigned i = 0; i < n; ++i) {
                if (i == j || i == k || rho(i, j) <= WEIGHT_CUTOFF) {
                    continue;
                }
                for (unsigned l = 0; l < n; ++l) {
                    if (l == i || l == j || l == k) {
                        continue;
                    }
                    const double weight = rho(i, j) * rho(j, k) * rho(k, l);
                    if (weight <= WEIGHT_CUTOFF) {
                        continue;
                    }
                    if (std::sin(angle(x.data(), i, j, k, nullptr)) < 1e-3 ||
                        std::sin(angle(x.data(), j, k, l, nullptr)) < 1e-3) {
                        continue;
                    }
                    dihedral(x.data(), i, j, k, l, gradient);
                    const unsigned atoms[4] = {i, j, k, l};
                    add_term(hessian, K_TORSION * weight * angular_scale, atoms, 4, gradient);
                }
            }
        }
    }
    return hessian;
}

}  // namespace detail
}  // namespace OEVIBE
