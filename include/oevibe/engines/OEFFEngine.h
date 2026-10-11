#ifndef OEVIBE_ENGINES_OEFFENGINE_H
#define OEVIBE_ENGINES_OEFFENGINE_H

#include <memory>
#include <string>
#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {

/// OpenEye's built-in force fields through ``oeff`` (spec 8.6): MMFF94 and MMFF94s with
/// optional Sheffield water, and Sage and Parsley in the gas phase.
class OEFFEngine : public Engine {
public:
    std::string GetName() const override;
    /// :returns: The OEFF toolkit release, which also identifies the bundled Sage and Parsley.
    std::string GetVersion() const override;
    bool IsAvailable() const override;
    std::string GetUnavailableReason() const override;
    std::vector<MethodCapabilities> GetMethods() const override;
    /// :raises UnsupportedError: When the force field cannot be set up on the system.
    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override;
};

}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_OEFFENGINE_H
