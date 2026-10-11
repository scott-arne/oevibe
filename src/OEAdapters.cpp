#include "oevibe/OEAdapters.h"

#include "core/MolFuncSession.h"
#include "oevibe/Error.h"

namespace OEVIBE {

MolFuncEngine::MolFuncEngine(std::string name, std::string version, Method method,
                             const OEMolPotential::OEMolFunc1& prototype,
                             bool requires_partial_charges)
    : name_(std::move(name)),
      version_(std::move(version)),
      method_(std::move(method)),
      prototype_(prototype.CreateCopy()),
      requires_partial_charges_(requires_partial_charges) {
    // Some OpenEye potentials (scaled and adaptor wrappers) return null from CreateCopy().
    if (!prototype_) {
        throw OptionsError("OpenEye potential for engine '" + name_ +
                           "' cannot be copied; MolFuncEngine needs a copyable potential");
    }
}

std::string MolFuncEngine::GetName() const { return name_; }
std::string MolFuncEngine::GetVersion() const { return version_; }
bool MolFuncEngine::IsAvailable() const { return true; }
std::string MolFuncEngine::GetUnavailableReason() const { return ""; }

std::vector<MethodCapabilities> MolFuncEngine::GetMethods() const {
    MethodCapabilities caps;
    caps.method = method_;
    caps.analytic_gradient = true;
    caps.analytic_hessian =
        dynamic_cast<const OEMolPotential::OEMolFunc2*>(prototype_.get()) != nullptr;
    caps.requires_partial_charges = requires_partial_charges_;
    caps.charge_from_topology = true;
    caps.force_field = true;
    return {caps};
}

std::shared_ptr<EngineSession> MolFuncEngine::CreateSession(const MolecularSystem& system,
                                                            const EngineSettings& settings) {
    static_cast<void>(settings);
    std::unique_ptr<OEMolPotential::OEMolFunc1> copy(prototype_->CreateCopy());
    if (!copy) {
        throw UnsupportedError("OpenEye potential for engine '" + name_ +
                               "' could not be copied for a session");
    }
    return std::make_shared<detail::MolFuncSession>(std::move(copy), system);
}

}  // namespace OEVIBE
