#include "core/ConstraintMath.h"

#include <algorithm>
#include <cmath>

#include "core/Geometry.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"

namespace OEVIBE {
namespace detail {

namespace {

double value_and_local_gradient(ConstraintKind kind, const std::vector<unsigned>& atoms,
                                const double* x, double* local) {
    switch (kind) {
        case ConstraintKind::DISTANCE:
            return distance(x, atoms[0], atoms[1], local);
        case ConstraintKind::ANGLE:
            return angle(x, atoms[0], atoms[1], atoms[2], local);
        case ConstraintKind::DIHEDRAL:
            return dihedral(x, atoms[0], atoms[1], atoms[2], atoms[3], local);
    }
    return 0.0;
}

ConstraintKind geometry_of(RestraintKind kind) {
    switch (kind) {
        case RestraintKind::ANGLE:
            return ConstraintKind::ANGLE;
        case RestraintKind::DIHEDRAL:
            return ConstraintKind::DIHEDRAL;
        case RestraintKind::DISTANCE:
        case RestraintKind::POSITION:
            return ConstraintKind::DISTANCE;
    }
    return ConstraintKind::DISTANCE;
}

void scatter(const std::vector<unsigned>& atoms, const double* local, double scale,
             std::vector<double>& full) {
    for (std::size_t slot = 0; slot < atoms.size(); ++slot) {
        for (int axis = 0; axis < 3; ++axis) {
            full[3 * atoms[slot] + axis] += scale * local[3 * slot + axis];
        }
    }
}

std::vector<unsigned> to_dense(const std::vector<unsigned>& atoms, const MolecularSystem& system) {
    std::vector<unsigned> dense;
    dense.reserve(atoms.size());
    for (unsigned atom : atoms) {
        dense.push_back(system.GetDenseIndex(atom));
    }
    return dense;
}

// Targets are stored in degrees on the public side and in radians here.
double target_in_internal_units(ConstraintKind kind, double target) {
    return kind == ConstraintKind::DISTANCE ? target : target * Units::DEGREES_TO_RADIANS;
}

}  // namespace

void check_constraint_atoms(const ConstraintSet& constraints, const MolecularSystem& system) {
    for (unsigned atom : constraints.GetReferencedAtoms()) {
        system.GetDenseIndex(atom);
    }
}

void check_constraint_atoms(const ConstraintSet& constraints, const OEChem::OEMolBase& mol) {
    std::vector<bool> present(mol.GetMaxAtomIdx(), false);
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        present[atom->GetIdx()] = true;
    }
    for (unsigned atom : constraints.GetReferencedAtoms()) {
        if (atom >= present.size() || !present[atom]) {
            throw InputError("atom index " + std::to_string(atom) + " is not in the molecule");
        }
    }
}

ResolvedConstraintSet resolve_constraint_set(const ConstraintSet& constraints,
                                             const MolecularSystem& system,
                                             const std::vector<double>& x0) {
    ResolvedConstraintSet resolved;
    resolved.frozen = to_dense(constraints.GetFrozenAtoms(), system);
    std::sort(resolved.frozen.begin(), resolved.frozen.end());
    double local[12];
    for (const Constraint& constraint : constraints.GetConstraints()) {
        ResolvedConstraint item;
        item.kind = constraint.kind;
        item.atoms = to_dense(constraint.atoms, system);
        item.target = constraint.has_target
                          ? target_in_internal_units(constraint.kind, constraint.target)
                          : value_and_local_gradient(item.kind, item.atoms, x0.data(), local);
        resolved.constraints.push_back(item);
    }
    for (const Restraint& restraint : constraints.GetRestraints()) {
        ResolvedRestraint item;
        item.kind = restraint.kind;
        item.atoms = to_dense(restraint.atoms, system);
        item.force_constant = restraint.force_constant;
        if (restraint.kind == RestraintKind::POSITION) {
            for (int axis = 0; axis < 3; ++axis) {
                item.reference[axis] = x0[3 * item.atoms[0] + axis];
            }
        } else {
            const ConstraintKind geometry = geometry_of(restraint.kind);
            item.target = restraint.has_target
                              ? target_in_internal_units(geometry, restraint.target)
                              : value_and_local_gradient(geometry, item.atoms, x0.data(), local);
        }
        resolved.restraints.push_back(item);
    }
    return resolved;
}

double constraint_residual(const ResolvedConstraint& constraint, const std::vector<double>& x,
                           std::vector<double>* jacobian_row) {
    double local[12];
    const double value = value_and_local_gradient(constraint.kind, constraint.atoms, x.data(),
                                                  jacobian_row != nullptr ? local : nullptr);
    if (jacobian_row != nullptr) {
        jacobian_row->assign(x.size(), 0.0);
        scatter(constraint.atoms, local, 1.0, *jacobian_row);
    }
    const double residual = value - constraint.target;
    return constraint.kind == ConstraintKind::DIHEDRAL ? wrap_angle(residual) : residual;
}

ResidualSummary summarize_residuals(const std::vector<ResolvedConstraint>& constraints,
                                    const std::vector<double>& x) {
    ResidualSummary summary;
    for (const ResolvedConstraint& constraint : constraints) {
        const double residual = std::abs(constraint_residual(constraint, x, nullptr));
        if (constraint.kind == ConstraintKind::DISTANCE) {
            summary.max_distance = std::max(summary.max_distance, residual);
        } else {
            summary.max_angle = std::max(summary.max_angle, residual * Units::RADIANS_TO_DEGREES);
        }
    }
    return summary;
}

bool residuals_satisfied(const ResidualSummary& summary) {
    return summary.max_distance <= DISTANCE_RESIDUAL_TOLERANCE &&
           summary.max_angle <= ANGLE_RESIDUAL_TOLERANCE;
}

double restraint_energy(const std::vector<ResolvedRestraint>& restraints,
                        const std::vector<double>& x, std::vector<double>* gradient) {
    double energy = 0.0;
    double local[12];
    for (const ResolvedRestraint& restraint : restraints) {
        if (restraint.kind == RestraintKind::POSITION) {
            const unsigned atom = restraint.atoms[0];
            for (int axis = 0; axis < 3; ++axis) {
                const double d = x[3 * atom + axis] - restraint.reference[axis];
                energy += 0.5 * restraint.force_constant * d * d;
                if (gradient != nullptr) {
                    (*gradient)[3 * atom + axis] += restraint.force_constant * d;
                }
            }
            continue;
        }
        const ConstraintKind geometry = geometry_of(restraint.kind);
        const double value = value_and_local_gradient(geometry, restraint.atoms, x.data(),
                                                      gradient != nullptr ? local : nullptr);
        double deviation = value - restraint.target;
        if (geometry == ConstraintKind::DIHEDRAL) {
            deviation = wrap_angle(deviation);
        }
        energy += 0.5 * restraint.force_constant * deviation * deviation;
        if (gradient != nullptr) {
            scatter(restraint.atoms, local, restraint.force_constant * deviation, *gradient);
        }
    }
    return energy;
}

}  // namespace detail
}  // namespace OEVIBE
