#include "oevibe/oevibe.h"

#include <oechem.h>

namespace OEVIBE {

double calculate_molecular_weight(const OEChem::OEMolBase& mol) {
    return OEChem::OECalculateMolecularWeight(mol);
}

} // namespace OEVIBE
