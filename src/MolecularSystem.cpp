#include "oevibe/MolecularSystem.h"

#include <oequacpac.h>

#include <cmath>
#include <limits>
#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

constexpr unsigned NO_INDEX = std::numeric_limits<unsigned>::max();

OEChem::OEAtomBase* other_neighbor(const OEChem::OEAtomBase* atom,
                                   const OEChem::OEAtomBase* excluded) {
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom->GetAtoms(); nbr; ++nbr) {
        if (nbr->GetIdx() != excluded->GetIdx()) {
            return nbr;
        }
    }
    return nullptr;
}

// Copies tetrahedral and cis/trans stereo onto the snapshot. OpenEye stores stereo relative to
// neighbor atoms, so this runs after every bond exists and maps each neighbor to its copy. The
// input's own stereo is copied, rather than perceived from the 3D coordinates, so the snapshot
// never specifies a center that the input left unspecified.
void copy_stereo(const OEChem::OEMolBase& mol, const std::vector<unsigned>& dense_of_oe,
                 const std::vector<OEChem::OEAtomBase*>& atom_copies,
                 const std::vector<OEChem::OEBondBase*>& bond_copies) {
    const auto copy_of = [&](const OEChem::OEAtomBase* atom) {
        return atom_copies[dense_of_oe[atom->GetIdx()]];
    };
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!atom->HasStereoSpecified(OEChem::OEAtomStereo::Tetrahedral)) {
            continue;
        }
        std::vector<OEChem::OEAtomBase*> nbrs;
        std::vector<OEChem::OEAtomBase*> nbr_copies;
        for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom->GetAtoms(); nbr; ++nbr) {
            nbrs.push_back(nbr);
            nbr_copies.push_back(copy_of(nbr));
        }
        copy_of(atom)->SetStereo(nbr_copies, OEChem::OEAtomStereo::Tetrahedral,
                                 atom->GetStereo(nbrs, OEChem::OEAtomStereo::Tetrahedral));
    }
    std::size_t position = 0;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond, ++position) {
        if (!bond->HasStereoSpecified(OEChem::OEBondStereo::CisTrans)) {
            continue;
        }
        OEChem::OEAtomBase* bgn_nbr = other_neighbor(bond->GetBgn(), bond->GetEnd());
        OEChem::OEAtomBase* end_nbr = other_neighbor(bond->GetEnd(), bond->GetBgn());
        if (bgn_nbr == nullptr || end_nbr == nullptr) {
            continue;
        }
        const std::vector<OEChem::OEAtomBase*> reference = {bgn_nbr, end_nbr};
        bond_copies[position]->SetStereo(
            {copy_of(bgn_nbr), copy_of(end_nbr)}, OEChem::OEBondStereo::CisTrans,
            bond->GetStereo(reference, OEChem::OEBondStereo::CisTrans));
    }
}

}  // namespace

MolecularSystem::MolecularSystem(const OEChem::OEMolBase& mol, int charge, unsigned multiplicity,
                                 PartialChargePolicy policy)
    : max_oe_index_(mol.GetMaxAtomIdx()), charge_(charge), multiplicity_(multiplicity) {
    dense_of_oe_.assign(max_oe_index_, NO_INDEX);
    std::vector<OEChem::OEAtomBase*> copies;
    // Atoms are created in iteration order, so each snapshot atom's GetIdx() is its dense index.
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto dense = static_cast<unsigned>(oe_of_dense_.size());
        oe_of_dense_.push_back(atom->GetIdx());
        dense_of_oe_[atom->GetIdx()] = dense;
        atomic_numbers_.push_back(atom->GetAtomicNum());
        formal_charge_sum_ += atom->GetFormalCharge();
        OEChem::OEAtomBase* copy = snapshot_.NewAtom(atom->GetAtomicNum());
        copy->SetFormalCharge(atom->GetFormalCharge());
        copy->SetPartialCharge(atom->GetPartialCharge());
        copy->SetIsotope(atom->GetIsotope());
        copy->SetImplicitHCount(atom->GetImplicitHCount());
        copy->SetAromatic(atom->IsAromatic());
        copies.push_back(copy);
    }
    std::vector<OEChem::OEBondBase*> bond_copies;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        OEChem::OEBondBase* copy =
            snapshot_.NewBond(copies[dense_of_oe_[bond->GetBgnIdx()]],
                              copies[dense_of_oe_[bond->GetEndIdx()]], bond->GetOrder());
        copy->SetAromatic(bond->IsAromatic());
        bond_copies.push_back(copy);
    }
    copy_stereo(mol, dense_of_oe_, copies, bond_copies);
    OEChem::OEFindRingAtomsAndBonds(snapshot_);
    snapshot_.SetTitle(mol.GetTitle());

    snapshot_.DeleteConfs();
    std::vector<double> oe_coords(3 * max_oe_index_);
    const auto add_conformer = [&](const OEChem::OEMolBase& conf, unsigned conf_index) {
        conf.GetCoords(oe_coords.data());
        std::vector<double> dense(3 * oe_of_dense_.size());
        for (std::size_t d = 0; d < oe_of_dense_.size(); ++d) {
            for (int axis = 0; axis < 3; ++axis) {
                dense[3 * d + axis] = oe_coords[3 * oe_of_dense_[d] + axis];
            }
        }
        snapshot_.NewConf(dense.data());
        conformer_indices_.push_back(conf_index);
        conformer_coordinates_.push_back(std::move(dense));
    };
    if (const auto* multi = dynamic_cast<const OEChem::OEMCMolBase*>(&mol)) {
        for (OESystem::OEIter<OEChem::OEConfBase> conf = multi->GetConfs(); conf; ++conf) {
            add_conformer(*conf, conf->GetIdx());
        }
    } else {
        add_conformer(mol, 0);
    }

    switch (policy) {
        case PartialChargePolicy::NONE:
            break;
        case PartialChargePolicy::ASSIGN_AM1BCC_ELF10:
            if (!OEProton::OEAssignCharges(snapshot_, OEProton::OEAM1BCCELF10Charges())) {
                throw InputError("Quacpac AM1BCC-ELF10 charge assignment failed for '" +
                                 std::string(mol.GetTitle()) + "'");
            }
            break;
        case PartialChargePolicy::USE_INPUT: {
            double sum = 0.0;
            for (OESystem::OEIter<OEChem::OEAtomBase> atom = snapshot_.GetAtoms(); atom; ++atom) {
                if (!std::isfinite(atom->GetPartialCharge())) {
                    throw InputError("input partial charges must be finite");
                }
                sum += atom->GetPartialCharge();
            }
            if (std::abs(sum - charge_) > 1e-3) {
                throw InputError("input partial charges sum to " + std::to_string(sum) +
                                 " but the total charge is " + std::to_string(charge_));
            }
            break;
        }
    }
}

unsigned MolecularSystem::NumAtoms() const { return static_cast<unsigned>(atomic_numbers_.size()); }
unsigned MolecularSystem::NumConformers() const {
    return static_cast<unsigned>(conformer_coordinates_.size());
}
const std::vector<unsigned>& MolecularSystem::GetAtomicNumbers() const { return atomic_numbers_; }
int MolecularSystem::GetCharge() const { return charge_; }
unsigned MolecularSystem::GetMultiplicity() const { return multiplicity_; }
int MolecularSystem::GetFormalChargeSum() const { return formal_charge_sum_; }
bool MolecularSystem::HasDenseIndex(unsigned oe_index) const {
    return oe_index < dense_of_oe_.size() && dense_of_oe_[oe_index] != NO_INDEX;
}
unsigned MolecularSystem::GetDenseIndex(unsigned oe_index) const {
    if (!HasDenseIndex(oe_index)) {
        throw InputError("atom index " + std::to_string(oe_index) + " is not in the molecule");
    }
    return dense_of_oe_[oe_index];
}
unsigned MolecularSystem::GetOEAtomIndex(unsigned dense_index) const {
    return oe_of_dense_.at(dense_index);
}
const std::vector<unsigned>& MolecularSystem::GetOEAtomIndices() const { return oe_of_dense_; }
unsigned MolecularSystem::GetMaxOEAtomIndex() const { return max_oe_index_; }
const std::vector<unsigned>& MolecularSystem::GetConformerIndices() const {
    return conformer_indices_;
}
const std::vector<double>& MolecularSystem::GetConformerCoordinates(unsigned position) const {
    return conformer_coordinates_.at(position);
}
const OEChem::OEMolBase& MolecularSystem::GetMolecule() const { return snapshot_; }

void MolecularSystem::CopyMolecule(OEChem::OEMolBase& out) const {
    out.Clear();
    if (auto* multi = dynamic_cast<OEChem::OEMCMolBase*>(&out)) {
        OEChem::OECopyMol(*multi, snapshot_);
    } else {
        OEChem::OECopyMol(out, snapshot_);
    }
}

std::vector<double> MolecularSystem::ToOELayout(const std::vector<double>& dense_xyz) const {
    std::vector<double> out(3 * max_oe_index_, 0.0);
    for (std::size_t d = 0; d < oe_of_dense_.size(); ++d) {
        for (int axis = 0; axis < 3; ++axis) {
            out[3 * oe_of_dense_[d] + axis] = dense_xyz.at(3 * d + axis);
        }
    }
    return out;
}

std::vector<double> MolecularSystem::ToOEScalarLayout(
    const std::vector<double>& dense_values) const {
    std::vector<double> out(max_oe_index_, 0.0);
    for (std::size_t d = 0; d < oe_of_dense_.size(); ++d) {
        out[oe_of_dense_[d]] = dense_values.at(d);
    }
    return out;
}

}  // namespace OEVIBE
