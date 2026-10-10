#ifndef OEVIBE_TEST_SUPPORT_H
#define OEVIBE_TEST_SUPPORT_H

#include <oechem.h>
#include <oeomega2.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace OEVIBE {
namespace test {

/// SMILES to a 3D OEMol with explicit hydrogens and up to ``max_confs`` OMEGA conformers.
inline OEChem::OEMol make_conformers(const std::string& smiles, unsigned max_confs = 1) {
    OEChem::OEMol mol;
    if (!OEChem::OESmilesToMol(mol, smiles)) {
        throw std::runtime_error("cannot parse SMILES " + smiles);
    }
    OEConfGen::OEOmegaOptions options;
    options.SetMaxConfs(max_confs);
    OEConfGen::OEOmega omega(options);
    if (omega.Build(mol) != OEConfGen::OEOmegaReturnCode::Success) {
        throw std::runtime_error("OMEGA failed for " + smiles);
    }
    return mol;
}

/// Methane whose atom indices have a gap: C is 0, a deleted helium was 1, hydrogens are 2-5.
inline OEChem::OEMol make_gapped_methane() {
    OEChem::OEMol mol;
    OEChem::OEAtomBase* carbon = mol.NewAtom(6);
    OEChem::OEAtomBase* dummy = mol.NewAtom(2);
    std::vector<OEChem::OEAtomBase*> hydrogens;
    for (int k = 0; k < 4; ++k) {
        OEChem::OEAtomBase* hydrogen = mol.NewAtom(1);
        mol.NewBond(carbon, hydrogen, 1);
        hydrogens.push_back(hydrogen);
    }
    mol.DeleteAtom(dummy);
    const double positions[4][3] = {{0.629, 0.629, 0.629},
                                    {-0.629, -0.629, 0.629},
                                    {-0.629, 0.629, -0.629},
                                    {0.629, -0.629, -0.629}};
    std::vector<double> coords(3 * mol.GetMaxAtomIdx(), 0.0);
    for (int k = 0; k < 4; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
            coords[3 * hydrogens[k]->GetIdx() + axis] = positions[k][axis];
        }
    }
    mol.SetCoords(coords.data());
    mol.SetDimension(3);
    return mol;
}

}  // namespace test
}  // namespace OEVIBE

#endif  // OEVIBE_TEST_SUPPORT_H
