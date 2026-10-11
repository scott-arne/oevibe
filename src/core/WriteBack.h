#ifndef OEVIBE_CORE_WRITEBACK_H
#define OEVIBE_CORE_WRITEBACK_H

#include <oechem.h>

#include <string>

#include "oevibe/MolecularSystem.h"
#include "oevibe/Options.h"
#include "oevibe/Result.h"

namespace OEVIBE {
namespace detail {

/// :returns: ``value`` with six decimals, the SD tag number format.
std::string format_value(double value);

/// Applies spec 6.5 to the caller's molecule: coordinates (when ``write_coordinates``), SD
/// tags, and partial charges from the lowest OK conformer. Adds a warning to every result when
/// charges were requested but no conformer can supply them.
void write_back(OEChem::OEMolBase& mol, const MolecularSystem& system, ResultSet& results,
                const CalculationOptions& options, const std::string& solvent_tag,
                bool write_coordinates);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_WRITEBACK_H
