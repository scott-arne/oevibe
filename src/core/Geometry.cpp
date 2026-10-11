#include "core/Geometry.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

#include "oevibe/Units.h"

namespace OEVIBE {
namespace detail {

namespace {

using Vec3 = Eigen::Vector3d;

Vec3 position(const double* x, unsigned atom) {
    return Vec3(x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]);
}

void put(double* gradient, unsigned slot, const Vec3& value) {
    gradient[3 * slot] = value.x();
    gradient[3 * slot + 1] = value.y();
    gradient[3 * slot + 2] = value.z();
}

}  // namespace

double wrap_angle(double radians) {
    double wrapped = std::fmod(radians, 2.0 * Units::PI);
    if (wrapped <= -Units::PI) {
        wrapped += 2.0 * Units::PI;
    }
    if (wrapped > Units::PI) {
        wrapped -= 2.0 * Units::PI;
    }
    return wrapped;
}

double distance(const double* x, unsigned i, unsigned j, double* gradient) {
    const Vec3 d = position(x, i) - position(x, j);
    const double r = d.norm();
    if (gradient != nullptr) {
        const Vec3 unit = r > 0.0 ? Vec3(d / r) : Vec3::Zero();
        put(gradient, 0, unit);
        put(gradient, 1, -unit);
    }
    return r;
}

double angle(const double* x, unsigned i, unsigned j, unsigned k, double* gradient) {
    const Vec3 u = position(x, i) - position(x, j);
    const Vec3 v = position(x, k) - position(x, j);
    const double nu = u.norm();
    const double nv = v.norm();
    const double theta = std::atan2(u.cross(v).norm(), u.dot(v));
    if (gradient != nullptr) {
        const double s = std::sin(theta);
        if (nu == 0.0 || nv == 0.0 || std::abs(s) < 1e-8) {
            std::fill(gradient, gradient + 9, 0.0);
        } else {
            const double c = std::cos(theta);
            const Vec3 eu = u / nu;
            const Vec3 ev = v / nv;
            const Vec3 gi = (c * eu - ev) / (nu * s);
            const Vec3 gk = (c * ev - eu) / (nv * s);
            put(gradient, 0, gi);
            put(gradient, 1, -(gi + gk));
            put(gradient, 2, gk);
        }
    }
    return theta;
}

double dihedral(const double* x, unsigned i, unsigned j, unsigned k, unsigned l, double* gradient) {
    const Vec3 f = position(x, i) - position(x, j);
    const Vec3 g = position(x, j) - position(x, k);
    const Vec3 h = position(x, l) - position(x, k);
    const Vec3 a = f.cross(g);
    const Vec3 b = h.cross(g);
    const double gn = g.norm();
    const double phi = std::atan2(-gn * f.dot(b), a.dot(b));
    if (gradient != nullptr) {
        const double a2 = a.squaredNorm();
        const double b2 = b.squaredNorm();
        if (gn < 1e-12 || a2 < 1e-16 || b2 < 1e-16) {
            std::fill(gradient, gradient + 12, 0.0);
        } else {
            const double fg = f.dot(g) / (a2 * gn);
            const double hg = h.dot(g) / (b2 * gn);
            const Vec3 gi = -gn / a2 * a;
            const Vec3 gl = gn / b2 * b;
            const Vec3 gj = gn / a2 * a + fg * a - hg * b;
            const Vec3 gk = -gn / b2 * b - fg * a + hg * b;
            put(gradient, 0, gi);
            put(gradient, 1, gj);
            put(gradient, 2, gk);
            put(gradient, 3, gl);
        }
    }
    return phi;
}

}  // namespace detail
}  // namespace OEVIBE
