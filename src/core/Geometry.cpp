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

/// :returns: A unit vector perpendicular to the unit vector ``axis``, built from the coordinate
///     axis least aligned with it so that the same bond always gives the same direction.
Vec3 perpendicular_to(const Vec3& axis) {
    Eigen::Index least = 0;
    axis.cwiseAbs().minCoeff(&least);
    const Vec3 unit = Vec3::Unit(least);
    return (unit - unit.dot(axis) * axis).normalized();
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
        if (nu == 0.0 || nv == 0.0) {
            std::fill(gradient, gradient + 9, 0.0);
            return theta;
        }
        const Vec3 eu = u / nu;
        const Vec3 ev = v / nv;
        Vec3 gi;
        Vec3 gk;
        if (is_linear_bend(theta)) {
            // A linear bend has no derivative, and a zero gradient would leave a restraint or
            // constraint toward any other angle with nothing to follow. Use the limit of the
            // regular formula as the bend opens toward a fixed perpendicular direction: the
            // one-sided derivative along it.
            const Vec3 bend = perpendicular_to(eu);
            gi = -bend / nu;
            gk = (u.dot(v) < 0.0 ? -bend : bend) / nv;
        } else {
            const double s = std::sin(theta);
            const double c = std::cos(theta);
            gi = (c * eu - ev) / (nu * s);
            gk = (c * ev - eu) / (nv * s);
        }
        put(gradient, 0, gi);
        put(gradient, 1, -(gi + gk));
        put(gradient, 2, gk);
    }
    return theta;
}

bool is_linear_bend(double theta) { return std::abs(std::sin(theta)) < 1e-8; }

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
