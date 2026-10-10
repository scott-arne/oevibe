#ifndef OEVIBE_OEADAPTERS_H
#define OEVIBE_OEADAPTERS_H

#include <oemolpotential.h>

#include <memory>
#include <string>
#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {

/// One OpenEye potential as a single-method OEVibe engine (spec 8.8), so user-composed OpenEye
/// potentials get OEVibe's optimizer and scans. It is not registered automatically.
class MolFuncEngine : public Engine {
public:
    /// :param name: Registry name.
    /// :param version: Reported engine version.
    /// :param method: The one method this engine declares.
    /// :param prototype: Copied with ``CreateCopy()`` now; each session gets its own copy.
    /// :param requires_partial_charges: Declared in the method's capabilities.
    MolFuncEngine(std::string name, std::string version, Method method,
                  const OEMolPotential::OEMolFunc1& prototype,
                  bool requires_partial_charges = false);

    std::string GetName() const override;
    std::string GetVersion() const override;
    bool IsAvailable() const override;
    std::string GetUnavailableReason() const override;
    std::vector<MethodCapabilities> GetMethods() const override;
    /// :raises UnsupportedError: When the potential cannot be set up on the system.
    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override;

private:
    std::string name_;
    std::string version_;
    Method method_;
    std::unique_ptr<OEMolPotential::OEMolFunc1> prototype_;
    bool requires_partial_charges_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_OEADAPTERS_H
