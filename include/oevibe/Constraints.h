#ifndef OEVIBE_CONSTRAINTS_H
#define OEVIBE_CONSTRAINTS_H

#include <oechem.h>

#include <vector>

namespace OEVIBE {

// NOLINTBEGIN(performance-enum-size)
enum class ConstraintKind { DISTANCE, ANGLE, DIHEDRAL };
enum class RestraintKind { DISTANCE, ANGLE, DIHEDRAL, POSITION };
// NOLINTEND(performance-enum-size)

/// An exact constraint on OpenEye atom indices (spec 6.6). Targets use spec 3.1 units:
/// Angstrom for distances, degrees for angles and dihedrals. Without a target the conformer's
/// starting value is held.
struct Constraint {
    ConstraintKind kind = ConstraintKind::DISTANCE;
    std::vector<unsigned> atoms;  ///< 2, 3, or 4 OpenEye atom indices.
    bool has_target = false;
    double target = 0.0;
};

/// A harmonic restraint, E = 1/2 k d^2 (spec 3.1). Force constants are kcal/mol/Angstrom^2 for
/// distance and position restraints and kcal/mol/rad^2 for angle and dihedral restraints. A
/// position restraint's reference is the atom's coordinates at the start of the optimization.
struct Restraint {
    RestraintKind kind = RestraintKind::DISTANCE;
    std::vector<unsigned> atoms;  ///< 1 (position), 2, 3, or 4 OpenEye atom indices.
    double force_constant = 0.0;
    bool has_target = false;
    double target = 0.0;
};

/// :returns: The angle in degrees mapped to (-180, 180].
double normalize_dihedral_degrees(double degrees);

/// Frozen atoms, exact constraints, and restraints on OpenEye atom indices (spec 6.6). It holds
/// no predicates and no molecule reference; atoms are resolved against a molecule when a
/// calculation runs.
class ConstraintSet {
public:
    void FreezeAtom(const OEChem::OEAtomBase& atom);
    void FreezeAtomIndices(const std::vector<unsigned>& indices);
    /// Evaluates the predicate over ``mol`` now and stores the matching indices.
    void FreezeAtoms(const OEChem::OEMolBase& mol,
                     const OESystem::OEUnaryPredicate<OEChem::OEAtomBase>& predicate);

    /// :raises OptionsError: When atoms repeat, or a target is not finite and greater than 0.
    void AddDistance(unsigned a, unsigned b);
    void AddDistance(unsigned a, unsigned b, double target);
    /// :raises OptionsError: When atoms repeat, or a target lies outside [0, 180].
    void AddAngle(unsigned a, unsigned b, unsigned c);
    void AddAngle(unsigned a, unsigned b, unsigned c, double target);
    /// :raises OptionsError: When atoms repeat, or a target is not finite.
    void AddDihedral(unsigned a, unsigned b, unsigned c, unsigned d);
    void AddDihedral(unsigned a, unsigned b, unsigned c, unsigned d, double target);

    /// :raises OptionsError: When atoms repeat, a force constant is not finite and greater than
    ///     0, or a target is out of range as for the matching constraint.
    void AddDistanceRestraint(unsigned a, unsigned b, double force_constant);
    void AddDistanceRestraint(unsigned a, unsigned b, double force_constant, double target);
    void AddAngleRestraint(unsigned a, unsigned b, unsigned c, double force_constant);
    void AddAngleRestraint(unsigned a, unsigned b, unsigned c, double force_constant,
                           double target);
    void AddDihedralRestraint(unsigned a, unsigned b, unsigned c, unsigned d,
                              double force_constant);
    void AddDihedralRestraint(unsigned a, unsigned b, unsigned c, unsigned d, double force_constant,
                              double target);
    void AddPositionRestraint(unsigned atom, double force_constant);

    /// :returns: Frozen OpenEye atom indices, sorted and unique.
    const std::vector<unsigned>& GetFrozenAtoms() const;
    const std::vector<Constraint>& GetConstraints() const;
    const std::vector<Restraint>& GetRestraints() const;
    bool HasFrozenAtoms() const;
    bool HasExactConstraints() const;
    bool HasRestraints() const;
    bool IsEmpty() const;
    /// :returns: Every OpenEye atom index any term references, sorted and unique.
    std::vector<unsigned> GetReferencedAtoms() const;

private:
    std::vector<unsigned> frozen_;
    std::vector<Constraint> constraints_;
    std::vector<Restraint> restraints_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_CONSTRAINTS_H
