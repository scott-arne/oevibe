#include "oevibe/engines/OEFFEngine.h"

#include <oeff.h>
#include <oemmff.h>
#include <oesmirnoff.h>

#include "core/MolFuncSession.h"
#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

MethodCapabilities base_capabilities(const char* method) {
    MethodCapabilities caps;
    caps.method = Method(method);
    caps.analytic_gradient = true;
    caps.analytic_hessian = true;
    caps.charge_from_topology = true;
    caps.force_field = true;
    return caps;
}

std::unique_ptr<OEMolPotential::OEMolFunc1> make_potential(const EngineSettings& settings) {
    const std::string& method = settings.method.GetName();
    const bool water = !settings.solvent.IsGasPhase();
    if (method == MMFF94 || method == MMFF94S) {
        const bool mmff94s = method == MMFF94S;
        if (water) {
            OEFF::OEMMFFSheffieldOptions options;
            options.SetForceFieldType(mmff94s ? OEFF::OEMMFFSheffieldFFType::MMFF94S_SHEFF
                                              : OEFF::OEMMFFSheffieldFFType::MMFF_SHEFF);
            return std::make_unique<OEFF::OEMMFFSheffield>(options);
        }
        // This uses the deprecated flag constructor because one built from OEMMFF94Params or
        // OEMMFF94sParams fails Setup ("Ligand has no partial charges") when every MMFF charge
        // is zero, as for any alkane. Both give the same energy wherever both set up.
        return std::make_unique<OEMolMMFF::OEMMFF>(mmff94s);
    }
    if (method == SAGE) {
        return std::make_unique<OEMolSmirnoff::OESage>();
    }
    if (method == PARSLEY) {
        return std::make_unique<OEMolSmirnoff::OEParsley>();
    }
    throw OptionsError("engine 'oeff' does not provide method '" + method + "'");
}

}  // namespace

std::string OEFFEngine::GetName() const { return "oeff"; }
std::string OEFFEngine::GetVersion() const { return OEFF::OEFFGetRelease(); }
bool OEFFEngine::IsAvailable() const { return true; }
std::string OEFFEngine::GetUnavailableReason() const { return ""; }

std::vector<MethodCapabilities> OEFFEngine::GetMethods() const {
    std::vector<MethodCapabilities> methods;
    for (const char* name : {MMFF94, MMFF94S}) {
        MethodCapabilities caps = base_capabilities(name);
        caps.solvent_models = {SolventModel::SHEFFIELD};
        caps.solvents = {"water"};
        methods.push_back(caps);
    }
    for (const char* name : {SAGE, PARSLEY}) {
        MethodCapabilities caps = base_capabilities(name);
        caps.requires_partial_charges = true;
        methods.push_back(caps);
    }
    return methods;
}

std::shared_ptr<EngineSession> OEFFEngine::CreateSession(const MolecularSystem& system,
                                                         const EngineSettings& settings) {
    return std::make_shared<detail::MolFuncSession>(make_potential(settings), system);
}

}  // namespace OEVIBE
