#ifndef OEVIBE_MOLECULARSYSTEM_H
#define OEVIBE_MOLECULARSYSTEM_H

#include <oechem.h>

#include <vector>

namespace OEVIBE {

// NOLINTBEGIN(performance-enum-size)
/// How partial charges on the snapshot are prepared (spec 6.2).
enum class PartialChargePolicy { NONE, ASSIGN_AM1BCC_ELF10, USE_INPUT };
// NOLINTEND(performance-enum-size)

/// Immutable snapshot of one input molecule, built once per call and shared by every session
/// of that call (spec 6.2). Engines see only this, never the caller's molecule.
class MolecularSystem {
public:
    /// :param mol: An input already passed through input preparation (explicit hydrogens, 3D).
    /// :param charge: Total charge after overrides.
    /// :param multiplicity: Spin multiplicity after overrides.
    /// :param policy: ASSIGN_AM1BCC_ELF10 assigns Quacpac charges to the snapshot using every
    ///     conformer; USE_INPUT keeps the input's charges after checking them.
    /// :raises InputError: When charge assignment fails, or USE_INPUT finds a charge that is
    ///     not finite or a sum that differs from ``charge`` by more than 1e-3 e.
    MolecularSystem(const OEChem::OEMolBase& mol, int charge, unsigned multiplicity,
                    PartialChargePolicy policy = PartialChargePolicy::NONE);

    unsigned NumAtoms() const;
    unsigned NumConformers() const;
    /// :returns: Atomic numbers in dense order.
    const std::vector<unsigned>& GetAtomicNumbers() const;
    int GetCharge() const;
    unsigned GetMultiplicity() const;
    int GetFormalChargeSum() const;
    bool HasDenseIndex(unsigned oe_index) const;
    /// :raises InputError: When the OpenEye atom index is not in the molecule.
    unsigned GetDenseIndex(unsigned oe_index) const;
    unsigned GetOEAtomIndex(unsigned dense_index) const;
    /// :returns: The OpenEye atom index of each dense position.
    const std::vector<unsigned>& GetOEAtomIndices() const;
    /// :returns: The input molecule's ``GetMaxAtomIdx()``, the size of OpenEye-layout arrays.
    unsigned GetMaxOEAtomIndex() const;
    /// :returns: Each conformer's ``OEConfBase::GetIdx()`` in input order (0 for an
    ///     ``OEGraphMol``).
    const std::vector<unsigned>& GetConformerIndices() const;
    /// :returns: Dense 3N coordinates in Angstrom of the conformer at ``position``.
    const std::vector<double>& GetConformerCoordinates(unsigned position) const;
    /// :returns: The snapshot: explicit hydrogens, every conformer, atoms in dense order.
    const OEChem::OEMolBase& GetMolecule() const;
    /// Copies the snapshot into ``out``, replacing its contents.
    void CopyMolecule(OEChem::OEMolBase& out) const;
    /// :returns: A 3 * GetMaxOEAtomIndex() array with zeros at unused indices.
    std::vector<double> ToOELayout(const std::vector<double>& dense_xyz) const;
    /// :returns: A GetMaxOEAtomIndex() array with zeros at unused indices.
    std::vector<double> ToOEScalarLayout(const std::vector<double>& dense_values) const;

private:
    OEChem::OEMol snapshot_;
    std::vector<unsigned> atomic_numbers_;
    std::vector<unsigned> oe_of_dense_;
    std::vector<unsigned> dense_of_oe_;
    std::vector<unsigned> conformer_indices_;
    std::vector<std::vector<double>> conformer_coordinates_;
    unsigned max_oe_index_ = 0;
    int charge_ = 0;
    unsigned multiplicity_ = 1;
    int formal_charge_sum_ = 0;
};

}  // namespace OEVIBE

#endif  // OEVIBE_MOLECULARSYSTEM_H
