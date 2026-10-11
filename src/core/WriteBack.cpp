#include "core/WriteBack.h"

#include <cstdio>

namespace OEVIBE {
namespace detail {

namespace {

OEChem::OEMolBase* find_conformer(OEChem::OEMolBase& mol, unsigned conf_index) {
    auto* multi = dynamic_cast<OEChem::OEMCMolBase*>(&mol);
    if (multi == nullptr) {
        return &mol;
    }
    for (OESystem::OEIter<OEChem::OEConfBase> conf = multi->GetConfs(); conf; ++conf) {
        if (conf->GetIdx() == conf_index) {
            return conf;
        }
    }
    return nullptr;
}

}  // namespace

std::string format_value(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6f", value);
    return buffer;
}

void write_back(OEChem::OEMolBase& mol, const MolecularSystem& system, ResultSet& results,
                const CalculationOptions& options, const std::string& solvent_tag,
                bool write_coordinates) {
    const std::string prefix = "OEVibe." + options.GetMethod().GetName() + "." + solvent_tag + ".";
    for (std::size_t i = 0; i < results.Size(); ++i) {
        const ConformerResult& result = results.Get(i);
        OEChem::OEMolBase* target = find_conformer(mol, result.GetConformerIndex());
        if (target == nullptr) {
            continue;
        }
        const bool has_geometry = result.GetStatus() == ConformerStatus::OK ||
                                  result.GetStatus() == ConformerStatus::NOT_CONVERGED;
        if (write_coordinates && options.GetUpdateCoordinates() && has_geometry &&
            result.HasCoordinates()) {
            const std::vector<double> coordinates = result.GetCoordinates();
            target->SetCoords(coordinates.data());
        }
        if (options.GetWriteTags()) {
            OEChem::OESetSDData(*target, prefix + "Status",
                                conformer_status_name(result.GetStatus()));
            if (result.HasEnergy()) {
                OEChem::OESetSDData(*target, prefix + "Energy", format_value(result.GetEnergy()));
            }
            if (result.HasRelativeEnergy()) {
                OEChem::OESetSDData(*target, prefix + "RelativeEnergy",
                                    format_value(result.GetRelativeEnergy()));
            }
        }
    }
    if (options.GetAssignPartialCharges()) {
        // Partial charges are atom properties shared by every conformer, so exactly one
        // conformer supplies them: the lowest-energy OK one (spec 6.5).
        if (results.HasLowest() && results.GetLowest().HasCharges()) {
            const std::vector<double> charges = results.GetLowest().GetCharges();
            for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
                if (system.HasDenseIndex(atom->GetIdx())) {
                    atom->SetPartialCharge(charges[atom->GetIdx()]);
                }
            }
        } else {
            results.AddWarningToAll(
                "partial charges were not written: no conformer finished OK with charges");
        }
    }
}

}  // namespace detail
}  // namespace OEVIBE
