#ifndef OEVIBE_CORE_INPUTPREP_H
#define OEVIBE_CORE_INPUTPREP_H

#include <oechem.h>

#include <string>
#include <vector>

#include "oevibe/Options.h"

namespace OEVIBE {
namespace detail {

/// What input preparation decided about one molecule (spec 6.1).
struct PreparedInput {
    std::vector<std::string> warnings;
    int charge = 0;
    unsigned multiplicity = 1;
};

/// Applies spec 6.1 to the caller's molecule, adding explicit hydrogens in place when allowed.
/// :raises InputError: For an empty or non-3D molecule, implicit hydrogens that may not be
///     added, a failed hydrogen addition, or a charge and multiplicity that the electron count
///     cannot satisfy.
PreparedInput prepare_input(OEChem::OEMolBase& mol, const CalculationOptions& options);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_INPUTPREP_H
