#include "oevibe/Constraints.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw OptionsError(message);
    }
}

void require_distinct(const std::vector<unsigned>& atoms) {
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            require(atoms[i] != atoms[j], "constraint and restraint atoms must be distinct");
        }
    }
}

void require_force_constant(double force_constant) {
    require(std::isfinite(force_constant) && force_constant > 0.0,
            "restraint force constants must be finite and greater than 0");
}

double checked_target(ConstraintKind kind, double target) {
    require(std::isfinite(target), "constraint and restraint targets must be finite");
    switch (kind) {
        case ConstraintKind::DISTANCE:
            require(target > 0.0, "distance targets must be greater than 0 Angstrom");
            return target;
        case ConstraintKind::ANGLE:
            require(target >= 0.0 && target <= 180.0, "angle targets must lie in [0, 180] degrees");
            return target;
        case ConstraintKind::DIHEDRAL:
            return normalize_dihedral_degrees(target);
    }
    return target;
}

Constraint make_constraint(ConstraintKind kind, std::vector<unsigned> atoms, bool has_target,
                           double target) {
    require_distinct(atoms);
    Constraint constraint;
    constraint.kind = kind;
    constraint.atoms = std::move(atoms);
    constraint.has_target = has_target;
    constraint.target = has_target ? checked_target(kind, target) : 0.0;
    return constraint;
}

Restraint make_restraint(RestraintKind kind, ConstraintKind geometry, std::vector<unsigned> atoms,
                         double force_constant, bool has_target, double target) {
    require_distinct(atoms);
    require_force_constant(force_constant);
    Restraint restraint;
    restraint.kind = kind;
    restraint.atoms = std::move(atoms);
    restraint.force_constant = force_constant;
    restraint.has_target = has_target;
    restraint.target = has_target ? checked_target(geometry, target) : 0.0;
    return restraint;
}

}  // namespace

double normalize_dihedral_degrees(double degrees) {
    double wrapped = std::fmod(degrees, 360.0);
    if (wrapped <= -180.0) {
        wrapped += 360.0;
    }
    if (wrapped > 180.0) {
        wrapped -= 360.0;
    }
    return wrapped;
}

void ConstraintSet::FreezeAtom(const OEChem::OEAtomBase& atom) {
    FreezeAtomIndices({atom.GetIdx()});
}

void ConstraintSet::FreezeAtomIndices(const std::vector<unsigned>& indices) {
    frozen_.insert(frozen_.end(), indices.begin(), indices.end());
    std::sort(frozen_.begin(), frozen_.end());
    frozen_.erase(std::unique(frozen_.begin(), frozen_.end()), frozen_.end());
}

void ConstraintSet::FreezeAtoms(const OEChem::OEMolBase& mol,
                                const OESystem::OEUnaryPredicate<OEChem::OEAtomBase>& predicate) {
    std::vector<unsigned> indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(predicate); atom; ++atom) {
        indices.push_back(atom->GetIdx());
    }
    FreezeAtomIndices(indices);
}

void ConstraintSet::AddDistance(unsigned a, unsigned b) {
    constraints_.push_back(make_constraint(ConstraintKind::DISTANCE, {a, b}, false, 0.0));
}
void ConstraintSet::AddDistance(unsigned a, unsigned b, double target) {
    constraints_.push_back(make_constraint(ConstraintKind::DISTANCE, {a, b}, true, target));
}
void ConstraintSet::AddAngle(unsigned a, unsigned b, unsigned c) {
    constraints_.push_back(make_constraint(ConstraintKind::ANGLE, {a, b, c}, false, 0.0));
}
void ConstraintSet::AddAngle(unsigned a, unsigned b, unsigned c, double target) {
    constraints_.push_back(make_constraint(ConstraintKind::ANGLE, {a, b, c}, true, target));
}
void ConstraintSet::AddDihedral(unsigned a, unsigned b, unsigned c, unsigned d) {
    constraints_.push_back(make_constraint(ConstraintKind::DIHEDRAL, {a, b, c, d}, false, 0.0));
}
void ConstraintSet::AddDihedral(unsigned a, unsigned b, unsigned c, unsigned d, double target) {
    constraints_.push_back(make_constraint(ConstraintKind::DIHEDRAL, {a, b, c, d}, true, target));
}

void ConstraintSet::AddDistanceRestraint(unsigned a, unsigned b, double force_constant) {
    restraints_.push_back(make_restraint(RestraintKind::DISTANCE, ConstraintKind::DISTANCE, {a, b},
                                         force_constant, false, 0.0));
}
void ConstraintSet::AddDistanceRestraint(unsigned a, unsigned b, double force_constant,
                                         double target) {
    restraints_.push_back(make_restraint(RestraintKind::DISTANCE, ConstraintKind::DISTANCE, {a, b},
                                         force_constant, true, target));
}
void ConstraintSet::AddAngleRestraint(unsigned a, unsigned b, unsigned c, double force_constant) {
    restraints_.push_back(make_restraint(RestraintKind::ANGLE, ConstraintKind::ANGLE, {a, b, c},
                                         force_constant, false, 0.0));
}
void ConstraintSet::AddAngleRestraint(unsigned a, unsigned b, unsigned c, double force_constant,
                                      double target) {
    restraints_.push_back(make_restraint(RestraintKind::ANGLE, ConstraintKind::ANGLE, {a, b, c},
                                         force_constant, true, target));
}
void ConstraintSet::AddDihedralRestraint(unsigned a, unsigned b, unsigned c, unsigned d,
                                         double force_constant) {
    restraints_.push_back(make_restraint(RestraintKind::DIHEDRAL, ConstraintKind::DIHEDRAL,
                                         {a, b, c, d}, force_constant, false, 0.0));
}
void ConstraintSet::AddDihedralRestraint(unsigned a, unsigned b, unsigned c, unsigned d,
                                         double force_constant, double target) {
    restraints_.push_back(make_restraint(RestraintKind::DIHEDRAL, ConstraintKind::DIHEDRAL,
                                         {a, b, c, d}, force_constant, true, target));
}
void ConstraintSet::AddPositionRestraint(unsigned atom, double force_constant) {
    require_force_constant(force_constant);
    Restraint restraint;
    restraint.kind = RestraintKind::POSITION;
    restraint.atoms = {atom};
    restraint.force_constant = force_constant;
    restraints_.push_back(restraint);
}

const std::vector<unsigned>& ConstraintSet::GetFrozenAtoms() const { return frozen_; }
const std::vector<Constraint>& ConstraintSet::GetConstraints() const { return constraints_; }
const std::vector<Restraint>& ConstraintSet::GetRestraints() const { return restraints_; }
bool ConstraintSet::HasFrozenAtoms() const { return !frozen_.empty(); }
bool ConstraintSet::HasExactConstraints() const { return !constraints_.empty(); }
bool ConstraintSet::HasRestraints() const { return !restraints_.empty(); }
bool ConstraintSet::IsEmpty() const {
    return frozen_.empty() && constraints_.empty() && restraints_.empty();
}

std::vector<unsigned> ConstraintSet::GetReferencedAtoms() const {
    std::vector<unsigned> atoms(frozen_);
    for (const Constraint& constraint : constraints_) {
        atoms.insert(atoms.end(), constraint.atoms.begin(), constraint.atoms.end());
    }
    for (const Restraint& restraint : restraints_) {
        atoms.insert(atoms.end(), restraint.atoms.begin(), restraint.atoms.end());
    }
    std::sort(atoms.begin(), atoms.end());
    atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
    return atoms;
}

}  // namespace OEVIBE
