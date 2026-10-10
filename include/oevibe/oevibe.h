#ifndef OEVIBE_H
#define OEVIBE_H

// Version information
#define OEVIBE_VERSION_MAJOR 0
#define OEVIBE_VERSION_MINOR 1
#define OEVIBE_VERSION_PATCH 0

#include <oechem.h>

namespace OEVIBE {

/// \brief Calculate the molecular weight of a molecule.
///
/// Passes the molecule natively from Python to C++ via SWIG typemaps,
/// then delegates to OEChem's OECalculateMolecularWeight.
///
/// \param mol Reference to an OEMolBase object.
/// \returns Molecular weight in Daltons.
double calculate_molecular_weight(const OEChem::OEMolBase& mol);

} // namespace OEVIBE

#endif // OEVIBE_H
