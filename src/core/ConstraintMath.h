#ifndef OEVIBE_CORE_CONSTRAINTMATH_H
#define OEVIBE_CORE_CONSTRAINTMATH_H

#include <array>
#include <vector>

#include "oevibe/Constraints.h"
#include "oevibe/MolecularSystem.h"

namespace OEVIBE {
namespace detail {

constexpr double DISTANCE_RESIDUAL_TOLERANCE = 1e-4;  ///< Angstrom (spec 7.1)
constexpr double ANGLE_RESIDUAL_TOLERANCE = 1e-3;     ///< degrees (spec 7.1)

/// An exact constraint on dense atoms; the target is in Angstrom or radians.
struct ResolvedConstraint {
    ConstraintKind kind = ConstraintKind::DISTANCE;
    std::vector<unsigned> atoms;
    double target = 0.0;
};

/// A restraint on dense atoms; ``reference`` holds a position restraint's starting coordinates.
struct ResolvedRestraint {
    RestraintKind kind = RestraintKind::DISTANCE;
    std::vector<unsigned> atoms;
    double force_constant = 0.0;  ///< kcal/mol/Angstrom^2 or kcal/mol/rad^2
    double target = 0.0;          ///< Angstrom or radians
    std::array<double, 3> reference{{0.0, 0.0, 0.0}};
};

struct ResolvedConstraintSet {
    std::vector<unsigned> frozen;  ///< Dense indices, sorted.
    std::vector<ResolvedConstraint> constraints;
    std::vector<ResolvedRestraint> restraints;
};

/// Largest constraint violations at one geometry.
struct ResidualSummary {
    double max_distance = 0.0;  ///< Angstrom
    double max_angle = 0.0;     ///< degrees, angles and dihedrals
};

/// :raises InputError: When a referenced OpenEye atom index is not in the system.
void check_constraint_atoms(const ConstraintSet& constraints, const MolecularSystem& system);
/// :raises InputError: When a referenced OpenEye atom index is not an atom of ``mol``.
void check_constraint_atoms(const ConstraintSet& constraints, const OEChem::OEMolBase& mol);

/// Maps OpenEye atom indices to dense ones and fills absent targets from ``x0``.
ResolvedConstraintSet resolve_constraint_set(const ConstraintSet& constraints,
                                             const MolecularSystem& system,
                                             const std::vector<double>& x0);

/// :returns: Value minus target, dihedrals wrapped to (-pi, pi].
/// :param jacobian_row: When not null, resized to 3N and filled with the derivative.
double constraint_residual(const ResolvedConstraint& constraint, const std::vector<double>& x,
                           std::vector<double>* jacobian_row);

ResidualSummary summarize_residuals(const std::vector<ResolvedConstraint>& constraints,
                                    const std::vector<double>& x);

bool residuals_satisfied(const ResidualSummary& summary);

/// :returns: The total restraint energy, E = 1/2 k d^2 per restraint (spec 3.1).
/// :param gradient: When not null (3N), the restraint gradient is added into it.
double restraint_energy(const std::vector<ResolvedRestraint>& restraints,
                        const std::vector<double>& x, std::vector<double>* gradient);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_CONSTRAINTMATH_H
