#ifndef OEVIBE_CORE_MOLFUNCSESSION_H
#define OEVIBE_CORE_MOLFUNCSESSION_H

#include <oechem.h>
#include <oemolpotential.h>

#include <memory>
#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {
namespace detail {

/// Session over an OpenEye potential set up on a private copy of the system's snapshot
/// (spec 8.6). The copy stays alive because potentials may keep pointers into it.
class MolFuncSession : public EngineSession {
public:
    /// :raises UnsupportedError: When ``PrepMol`` or ``Setup`` fails, ``PrepMol`` changes the
    ///     atom set, or the potential's variable count is not 3N.
    MolFuncSession(std::unique_ptr<OEMolPotential::OEMolFunc1> function,
                   const MolecularSystem& system);

    Evaluation Evaluate(const std::vector<double>& coords_angstrom,
                        const EvaluationRequest& request) override;

private:
    // Declared before function_ so the potential is destroyed first: it may touch the molecule
    // it was set up on while it is destroyed.
    OEChem::OEMol molecule_;
    std::unique_ptr<OEMolPotential::OEMolFunc1> function_;
    unsigned num_atoms_;
};

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_MOLFUNCSESSION_H
