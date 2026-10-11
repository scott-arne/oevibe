#ifndef OEVIBE_CORE_GEOMETRY_H
#define OEVIBE_CORE_GEOMETRY_H

namespace OEVIBE {
namespace detail {

/// :returns: ``radians`` mapped to (-pi, pi].
double wrap_angle(double radians);

/// Distance between atoms ``i`` and ``j`` of dense coordinates ``x`` (Angstrom).
/// :param gradient: When not null, receives dr/dx_i then dr/dx_j (6 values).
double distance(const double* x, unsigned i, unsigned j, double* gradient);

/// Angle i-j-k in radians, in [0, pi].
/// :param gradient: When not null, receives 9 values; zero within 1e-8 of 0 or pi, where the
///     derivative is undefined.
double angle(const double* x, unsigned i, unsigned j, unsigned k, double* gradient);

/// Dihedral i-j-k-l in radians, in (-pi, pi], with the IUPAC sign.
/// :param gradient: When not null, receives 12 values (Blondel and Karplus, J. Comput. Chem.
///     17, 1132, 1996); zero when three atoms are collinear.
double dihedral(const double* x, unsigned i, unsigned j, unsigned k, unsigned l, double* gradient);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_GEOMETRY_H
