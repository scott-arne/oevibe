#include "core/InputPrep.h"

#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {
namespace detail {

namespace {

std::string describe(const OEChem::OEMolBase& mol) {
    const std::string title = mol.GetTitle();
    return title.empty() ? std::string("molecule") : "molecule '" + title + "'";
}

}  // namespace

PreparedInput prepare_input(OEChem::OEMolBase& mol, const CalculationOptions& options) {
    PreparedInput prepared;
    if (mol.NumAtoms() == 0) {
        throw InputError(describe(mol) + " has no atoms");
    }
    if (mol.GetDimension() != 3) {
        throw InputError(describe(mol) + " needs 3D coordinates (dimension is " +
                         std::to_string(mol.GetDimension()) +
                         "); OEVibe does not generate conformers");
    }
    // The molecule-level dimension only follows the active conformer, yet MolecularSystem keeps
    // every conformer, so each one has to be 3D.
    if (auto* multi = dynamic_cast<OEChem::OEMCMolBase*>(&mol)) {
        for (OESystem::OEIter<OEChem::OEConfBase> conf = multi->GetConfs(); conf; ++conf) {
            if (conf->GetDimension() != 3) {
                throw InputError("conformer " + std::to_string(conf->GetIdx()) + " of " +
                                 describe(mol) + " needs 3D coordinates (dimension is " +
                                 std::to_string(conf->GetDimension()) + ")");
            }
        }
    }
    // Every check runs before the molecule is modified, so a rejected input is left untouched
    // and a retry with other options starts from the caller's original molecule.
    const bool has_implicit = OEChem::OEHasImplicitHydrogens(mol);
    if (has_implicit) {
        if (!options.GetAddHydrogens()) {
            throw InputError(describe(mol) +
                             " has implicit hydrogens; add explicit hydrogens or enable "
                             "SetAddHydrogens");
        }
        if (options.GetUseInputPartialCharges()) {
            throw InputError(describe(mol) +
                             " has implicit hydrogens, which cannot be added while "
                             "SetUseInputPartialCharges is on: they would have no charges");
        }
    }

    int formal = 0;
    long long protons = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        formal += atom->GetFormalCharge();
        // The implicit hydrogens are not atoms yet, but they will be, so they count as protons.
        protons += atom->GetAtomicNum() + atom->GetImplicitHCount();
    }
    prepared.charge = options.HasCharge() ? options.GetCharge() : formal;
    const long long electrons = protons - prepared.charge;
    if (electrons < 0) {
        throw InputError("a charge of " + std::to_string(prepared.charge) + " leaves " +
                         describe(mol) + " with a negative electron count");
    }
    prepared.multiplicity =
        options.HasMultiplicity() ? options.GetMultiplicity() : (electrons % 2 == 0 ? 1u : 2u);
    const long long unpaired = static_cast<long long>(prepared.multiplicity) - 1;
    if (unpaired > electrons || (electrons - unpaired) % 2 != 0) {
        throw InputError("multiplicity " + std::to_string(prepared.multiplicity) +
                         " is impossible with " + std::to_string(electrons) + " electrons");
    }

    if (has_implicit) {
        const unsigned before = mol.NumAtoms();
        bool added = false;
        // The multi-conformer overload places the hydrogens in 3D in every conformer, which
        // spike S3 verified; an OEGraphMol has one geometry and takes the plain overload.
        if (auto* multi = dynamic_cast<OEChem::OEMCMolBase*>(&mol)) {
            added = OEChem::OEAddExplicitHydrogens(*multi, false, true);
        } else {
            added = OEChem::OEAddExplicitHydrogens(mol, false, true);
        }
        if (!added) {
            throw InputError("adding explicit hydrogens failed for " + describe(mol));
        }
        prepared.warnings.push_back("added " + std::to_string(mol.NumAtoms() - before) +
                                    " explicit hydrogens with 3D placement");
    }
    return prepared;
}

}  // namespace detail
}  // namespace OEVIBE
