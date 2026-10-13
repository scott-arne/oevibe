#include "oevibe/Result.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/Thermochemistry.h"
#include "core/WriteBack.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"

namespace OEVIBE {

namespace {

template <typename T>
const T& require_value(const std::optional<T>& value, const char* name) {
    if (!value.has_value()) {
        throw OEVibeError(std::string(name) + " is not available for this conformer");
    }
    return *value;
}

std::pair<unsigned, unsigned> ordered(unsigned a, unsigned b) {
    return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

}  // namespace

std::string conformer_status_name(ConformerStatus status) {
    switch (status) {
        case ConformerStatus::OK:
            return "OK";
        case ConformerStatus::NOT_CONVERGED:
            return "NOT_CONVERGED";
        case ConformerStatus::ENGINE_ERROR:
            return "ENGINE_ERROR";
        case ConformerStatus::UNSUPPORTED:
            return "UNSUPPORTED";
    }
    return "ENGINE_ERROR";
}

unsigned ConformerResult::GetConformerIndex() const { return conformer_index_; }
void ConformerResult::SetConformerIndex(unsigned index) { conformer_index_ = index; }
ConformerStatus ConformerResult::GetStatus() const { return status_; }
const std::string& ConformerResult::GetMessage() const { return message_; }
void ConformerResult::SetStatus(ConformerStatus status, const std::string& message) {
    status_ = status;
    message_ = message;
}
std::vector<std::string> ConformerResult::GetWarnings() const { return warnings_; }
void ConformerResult::AddWarning(const std::string& warning) { warnings_.push_back(warning); }
std::vector<unsigned> ConformerResult::GetAtomIndices() const { return atom_indices_; }
void ConformerResult::SetAtomIndices(const std::vector<unsigned>& indices) {
    atom_indices_ = indices;
}

bool ConformerResult::HasEnergy() const { return energy_.has_value(); }
double ConformerResult::GetEnergy() const { return require_value(energy_, "energy"); }
void ConformerResult::SetEnergy(double energy) { energy_ = energy; }
bool ConformerResult::HasRelativeEnergy() const { return relative_energy_.has_value(); }
double ConformerResult::GetRelativeEnergy() const {
    return require_value(relative_energy_, "relative energy");
}
void ConformerResult::SetRelativeEnergy(double energy) { relative_energy_ = energy; }
void ConformerResult::ClearRelativeEnergy() { relative_energy_.reset(); }
bool ConformerResult::HasCoordinates() const { return coordinates_.has_value(); }
std::vector<double> ConformerResult::GetCoordinates() const {
    return require_value(coordinates_, "coordinates");
}
void ConformerResult::SetCoordinates(const std::vector<double>& coordinates) {
    coordinates_ = coordinates;
}
bool ConformerResult::HasGradient() const { return gradient_.has_value(); }
std::vector<double> ConformerResult::GetGradient() const {
    return require_value(gradient_, "gradient");
}
void ConformerResult::SetGradient(const std::vector<double>& gradient) { gradient_ = gradient; }
bool ConformerResult::HasCharges() const { return charges_.has_value(); }
std::vector<double> ConformerResult::GetCharges() const {
    return require_value(charges_, "partial charges");
}
double ConformerResult::GetCharge(const OEChem::OEAtomBase& atom) const {
    const std::vector<double>& charges = require_value(charges_, "partial charges");
    if (atom.GetIdx() >= charges.size()) {
        throw OEVibeError("atom index " + std::to_string(atom.GetIdx()) +
                          " is outside the result's atoms");
    }
    return charges[atom.GetIdx()];
}
void ConformerResult::SetCharges(const std::vector<double>& charges) { charges_ = charges; }
bool ConformerResult::HasDipole() const { return dipole_.has_value(); }
std::vector<double> ConformerResult::GetDipole() const { return require_value(dipole_, "dipole"); }
void ConformerResult::SetDipole(const std::vector<double>& dipole) { dipole_ = dipole; }
bool ConformerResult::HasOrbitalEnergies() const { return orbital_energies_.has_value(); }
std::vector<double> ConformerResult::GetOrbitalEnergies() const {
    return require_value(orbital_energies_, "orbital energies");
}
bool ConformerResult::HasHOMO() const {
    return orbital_energies_.has_value() && homo_index_ >= 0 &&
           static_cast<std::size_t>(homo_index_) < orbital_energies_->size();
}
double ConformerResult::GetHOMO() const {
    if (!HasHOMO()) {
        throw OEVibeError("HOMO is not available for this conformer");
    }
    return (*orbital_energies_)[static_cast<std::size_t>(homo_index_)];
}
bool ConformerResult::HasLUMO() const {
    return orbital_energies_.has_value() && homo_index_ >= -1 &&
           static_cast<long long>(homo_index_) + 1 <
               static_cast<long long>(orbital_energies_->size());
}
double ConformerResult::GetLUMO() const {
    if (!HasLUMO()) {
        throw OEVibeError("LUMO is not available for this conformer");
    }
    return (*orbital_energies_)[static_cast<std::size_t>(homo_index_ + 1)];
}
void ConformerResult::SetOrbitalEnergies(const std::vector<double>& energies, int homo_index) {
    if (homo_index < -1 || homo_index >= static_cast<long long>(energies.size())) {
        throw OEVibeError("HOMO index " + std::to_string(homo_index) + " is out of range for " +
                          std::to_string(energies.size()) + " orbital energies");
    }
    orbital_energies_ = energies;
    homo_index_ = homo_index;
}
bool ConformerResult::HasBondOrders() const { return bond_orders_.has_value(); }
std::vector<BondOrder> ConformerResult::GetBondOrders() const {
    std::vector<BondOrder> orders;
    for (const auto& [pair, order] : require_value(bond_orders_, "bond orders")) {
        orders.push_back({pair.first, pair.second, order});
    }
    return orders;
}
double ConformerResult::GetBondOrder(unsigned oe_a, unsigned oe_b) const {
    const auto& orders = require_value(bond_orders_, "bond orders");
    const auto it = orders.find(ordered(oe_a, oe_b));
    return it == orders.end() ? 0.0 : it->second;
}
double ConformerResult::GetBondOrder(const OEChem::OEBondBase& bond) const {
    return GetBondOrder(bond.GetBgnIdx(), bond.GetEndIdx());
}
void ConformerResult::SetBondOrders(const std::vector<BondOrder>& orders) {
    std::map<std::pair<unsigned, unsigned>, double> stored;
    for (const BondOrder& order : orders) {
        stored[ordered(order.i, order.j)] = order.order;
    }
    bond_orders_ = stored;
}
bool ConformerResult::HasHessian() const { return hessian_.has_value(); }
std::vector<double> ConformerResult::GetHessian() const {
    return require_value(hessian_, "Hessian");
}
void ConformerResult::SetHessian(const std::vector<double>& hessian) {
    const auto dim =
        static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(hessian.size()))));
    if (hessian.empty() || dim * dim != hessian.size() || dim % 3 != 0) {
        throw OEVibeError("a Hessian needs (3N)^2 values, got " + std::to_string(hessian.size()));
    }
    hessian_ = hessian;
}
bool ConformerResult::HasFrequencies() const { return frequencies_.has_value(); }
std::vector<double> ConformerResult::GetFrequencies() const {
    return require_value(frequencies_, "frequencies");
}
std::vector<double> ConformerResult::GetNormalModes() const {
    require_value(frequencies_, "normal modes");
    return normal_modes_;
}
unsigned ConformerResult::GetImaginaryModeCount() const {
    require_value(frequencies_, "imaginary-mode count");
    return imaginary_count_;
}
void ConformerResult::SetFrequencies(const std::vector<double>& frequencies,
                                     const std::vector<double>& normal_modes,
                                     unsigned imaginary_count) {
    const bool shaped = frequencies.empty()
                            ? normal_modes.empty()
                            : normal_modes.size() % frequencies.size() == 0 &&
                                  (normal_modes.size() / frequencies.size()) % 3 == 0 &&
                                  !normal_modes.empty();
    if (!shaped) {
        throw OEVibeError("normal modes must hold one row of 3N values per frequency; got " +
                          std::to_string(normal_modes.size()) + " values for " +
                          std::to_string(frequencies.size()) + " frequencies");
    }
    const auto negative = static_cast<unsigned>(
        std::count_if(frequencies.begin(), frequencies.end(), [](double f) { return f < 0.0; }));
    if (imaginary_count != negative) {
        throw OEVibeError("the imaginary-mode count " + std::to_string(imaginary_count) +
                          " differs from the " + std::to_string(negative) +
                          " negative frequencies");
    }
    frequencies_ = frequencies;
    normal_modes_ = normal_modes;
    imaginary_count_ = imaginary_count;
}
bool ConformerResult::HasThermochemistry() const { return thermochemistry_.has_value(); }
const Thermochemistry& ConformerResult::GetThermochemistry() const {
    return require_value(thermochemistry_, "thermochemistry");
}
void ConformerResult::SetThermochemistry(const Thermochemistry& thermochemistry) {
    thermochemistry_ = thermochemistry;
}
bool ConformerResult::HasOptimizationStats() const { return optimization_stats_.has_value(); }
const OptimizationStats& ConformerResult::GetOptimizationStats() const {
    return require_value(optimization_stats_, "optimization statistics");
}
void ConformerResult::SetOptimizationStats(const OptimizationStats& stats) {
    optimization_stats_ = stats;
}
const Provenance& ConformerResult::GetProvenance() const { return provenance_; }
void ConformerResult::SetProvenance(const Provenance& provenance) { provenance_ = provenance; }
const Timings& ConformerResult::GetTimings() const { return timings_; }
void ConformerResult::SetTimings(const Timings& timings) { timings_ = timings; }

std::size_t ResultSet::Size() const { return results_.size(); }

const ConformerResult& ResultSet::Get(std::size_t index) const {
    if (index >= results_.size()) {
        throw OEVibeError("result index " + std::to_string(index) + " is out of range");
    }
    return results_[index];
}

void ResultSet::RequireComparable() const {
    const ConformerResult* reference = nullptr;
    for (const ConformerResult& result : results_) {
        if (result.GetStatus() != ConformerStatus::OK || !result.HasEnergy()) {
            continue;
        }
        if (reference == nullptr) {
            reference = &result;
        } else {
            const Provenance& ref_prov = reference->GetProvenance();
            const Provenance& result_prov = result.GetProvenance();
            if (ref_prov.method != result_prov.method) {
                throw OEVibeError("cannot compare energies: methods differ (\"" + ref_prov.method +
                                  "\" vs \"" + result_prov.method + "\")");
            }
            if (ref_prov.solvent_model != result_prov.solvent_model) {
                throw OEVibeError("cannot compare energies: solvent models differ (\"" +
                                  ref_prov.solvent_model + "\" vs \"" + result_prov.solvent_model +
                                  "\")");
            }
            if (ref_prov.solvent != result_prov.solvent) {
                throw OEVibeError("cannot compare energies: solvents differ (\"" +
                                  ref_prov.solvent + "\" vs \"" + result_prov.solvent + "\")");
            }
        }
    }
}

bool ResultSet::HasLowest() const {
    return std::any_of(results_.begin(), results_.end(), [](const ConformerResult& r) {
        return r.GetStatus() == ConformerStatus::OK && r.HasEnergy();
    });
}

const ConformerResult& ResultSet::GetLowest() const {
    RequireComparable();
    const ConformerResult* lowest = nullptr;
    for (const ConformerResult& result : results_) {
        if (result.GetStatus() != ConformerStatus::OK || !result.HasEnergy()) {
            continue;
        }
        if (lowest == nullptr || result.GetEnergy() < lowest->GetEnergy() ||
            (result.GetEnergy() == lowest->GetEnergy() &&
             result.GetConformerIndex() < lowest->GetConformerIndex())) {
            lowest = &result;
        }
    }
    if (lowest == nullptr) {
        throw OEVibeError("no conformer finished OK");
    }
    return *lowest;
}

std::map<unsigned, double> ResultSet::GetRelativeEnergies() const {
    std::map<unsigned, double> relative;
    for (const ConformerResult& result : results_) {
        if (result.HasRelativeEnergy()) {
            relative[result.GetConformerIndex()] = result.GetRelativeEnergy();
        }
    }
    return relative;
}

std::map<unsigned, double> ResultSet::GetBoltzmannWeights(double temperature) const {
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw OEVibeError("temperature must be finite and greater than 0 K");
    }
    RequireComparable();
    std::vector<const ConformerResult*> ok;
    for (const ConformerResult& result : results_) {
        if (result.GetStatus() == ConformerStatus::OK && result.HasEnergy()) {
            ok.push_back(&result);
        }
    }
    const auto with_thermochemistry = static_cast<std::size_t>(
        std::count_if(ok.begin(), ok.end(),
                      [](const ConformerResult* result) { return result->HasThermochemistry(); }));
    if (with_thermochemistry != 0 && with_thermochemistry != ok.size()) {
        throw OEVibeError("cannot weight a mix of conformers with and without thermochemistry (" +
                          std::to_string(with_thermochemistry) + " of " +
                          std::to_string(ok.size()) + " OK conformers have it)");
    }
    std::map<unsigned, double> weights;
    if (ok.empty()) {
        return weights;
    }
    // X_i of spec 5.7 in kcal/mol: free energies at this temperature, or electronic energies.
    std::vector<double> values;
    for (const ConformerResult* result : ok) {
        values.push_back(with_thermochemistry == 0 ? result->GetEnergy()
                                                   : detail::recompute_thermochemistry(
                                                         result->GetThermochemistry(), temperature)
                                                         .free_energy);
    }
    const double minimum = *std::min_element(values.begin(), values.end());
    const double rt = Units::GAS_CONSTANT_KCAL * temperature;
    double total = 0.0;
    for (std::size_t k = 0; k < ok.size(); ++k) {
        const double weight = std::exp(-(values[k] - minimum) / rt);
        weights[ok[k]->GetConformerIndex()] = weight;
        total += weight;
    }
    for (auto& [index, weight] : weights) {
        weight /= total;
    }
    return weights;
}

void ResultSet::Add(ConformerResult result) { results_.push_back(std::move(result)); }

void ResultSet::Finalize() {
    RequireComparable();
    const bool any = HasLowest();
    const double minimum = any ? GetLowest().GetEnergy() : 0.0;
    for (ConformerResult& result : results_) {
        if (any && result.GetStatus() == ConformerStatus::OK && result.HasEnergy()) {
            result.SetRelativeEnergy(result.GetEnergy() - minimum);
        } else {
            result.ClearRelativeEnergy();
        }
    }
}

void ResultSet::AddWarningToAll(const std::string& warning) {
    for (ConformerResult& result : results_) {
        result.AddWarning(warning);
    }
}

std::string scan_point_status_name(ScanPointStatus status) {
    switch (status) {
        case ScanPointStatus::OK:
            return "OK";
        case ScanPointStatus::NOT_CONVERGED:
            return "NOT_CONVERGED";
        case ScanPointStatus::ENGINE_ERROR:
            return "ENGINE_ERROR";
        case ScanPointStatus::UNSUPPORTED:
            return "UNSUPPORTED";
        case ScanPointStatus::NOT_ATTEMPTED:
            return "NOT_ATTEMPTED";
    }
    return "ENGINE_ERROR";
}

std::vector<double> ScanPoint::GetAngles() const { return angles_; }
void ScanPoint::SetAngles(const std::vector<double>& angles) { angles_ = angles; }
ScanPointStatus ScanPoint::GetStatus() const { return status_; }
const std::string& ScanPoint::GetMessage() const { return message_; }
void ScanPoint::SetStatus(ScanPointStatus status, const std::string& message) {
    status_ = status;
    message_ = message;
}
bool ScanPoint::HasEnergy() const { return energy_.has_value(); }
double ScanPoint::GetEnergy() const { return require_value(energy_, "energy"); }
void ScanPoint::SetEnergy(double energy) { energy_ = energy; }
bool ScanPoint::HasRelativeEnergy() const { return relative_energy_.has_value(); }
double ScanPoint::GetRelativeEnergy() const {
    return require_value(relative_energy_, "relative energy");
}
void ScanPoint::SetRelativeEnergy(double energy) { relative_energy_ = energy; }
void ScanPoint::ClearRelativeEnergy() { relative_energy_.reset(); }
bool ScanPoint::HasCoordinates() const { return coordinates_.has_value(); }
std::vector<double> ScanPoint::GetCoordinates() const {
    return require_value(coordinates_, "coordinates");
}
void ScanPoint::SetCoordinates(const std::vector<double>& coordinates) {
    coordinates_ = coordinates;
}

unsigned ScanResult::GetNumDihedrals() const { return static_cast<unsigned>(dihedrals_.size()); }
std::vector<unsigned> ScanResult::GetDihedral(unsigned index) const {
    if (index >= dihedrals_.size()) {
        throw OEVibeError("dihedral index " + std::to_string(index) + " is out of range");
    }
    return dihedrals_[index];
}
std::size_t ScanResult::GetNumPoints() const { return points_.size(); }
const ScanPoint& ScanResult::GetPoint(std::size_t index) const {
    if (index >= points_.size()) {
        throw OEVibeError("scan point index " + std::to_string(index) + " is out of range");
    }
    return points_[index];
}
bool ScanResult::IsTruncated() const { return truncated_; }
std::vector<std::string> ScanResult::GetWarnings() const { return warnings_; }
const Provenance& ScanResult::GetProvenance() const { return provenance_; }

OEChem::OEMol ScanResult::GetMolecule() const {
    OEChem::OEMol mol;
    CopyMolecule(mol);
    return mol;
}

void ScanResult::CopyMolecule(OEChem::OEMCMolBase& out) const {
    out.Clear();
    // The atom map ties each graph atom to its copy, whatever indices ``out`` hands out.
    std::vector<OEChem::OEAtomBase*> copies(graph_.GetMaxAtomIdx(), nullptr);
    OEChem::OECopyMol(out, graph_, copies.data());
    out.DeleteConfs();
    std::vector<double> coords(3 * out.GetMaxAtomIdx(), 0.0);
    for (const ScanPoint& point : points_) {
        if (!point.HasEnergy() || !point.HasCoordinates()) {
            continue;
        }
        const std::vector<double> scanned = point.GetCoordinates();
        for (OESystem::OEIter<OEChem::OEAtomBase> atom = graph_.GetAtoms(); atom; ++atom) {
            const std::size_t source = 3 * static_cast<std::size_t>(oe_index_of_[atom->GetIdx()]);
            const std::size_t target =
                3 * static_cast<std::size_t>(copies[atom->GetIdx()]->GetIdx());
            for (std::size_t axis = 0; axis < 3; ++axis) {
                coords[target + axis] = scanned.at(source + axis);
            }
        }
        OEChem::OEConfBase* conf = out.NewConf(coords.data());
        std::string angles;
        for (const double angle : point.GetAngles()) {
            angles += (angles.empty() ? "" : ",") + detail::format_value(angle);
        }
        OEChem::OESetSDData(*conf, tag_prefix_ + "ScanAngles", angles);
        OEChem::OESetSDData(*conf, tag_prefix_ + "Energy", detail::format_value(point.GetEnergy()));
        if (point.HasRelativeEnergy()) {
            OEChem::OESetSDData(*conf, tag_prefix_ + "RelativeEnergy",
                                detail::format_value(point.GetRelativeEnergy()));
        }
        OEChem::OESetSDData(*conf, tag_prefix_ + "Status",
                            scan_point_status_name(point.GetStatus()));
    }
}

void ScanResult::AddDihedral(const std::vector<unsigned>& atoms) { dihedrals_.push_back(atoms); }
void ScanResult::AddPoint(ScanPoint point) { points_.push_back(std::move(point)); }
void ScanResult::SetTruncated(bool truncated) { truncated_ = truncated; }
void ScanResult::AddWarning(const std::string& warning) { warnings_.push_back(warning); }
void ScanResult::SetProvenance(const Provenance& provenance) { provenance_ = provenance; }

void ScanResult::SetMolecule(const OEChem::OEMolBase& mol, const std::string& tag_prefix) {
    graph_.Clear();
    std::vector<OEChem::OEAtomBase*> copies(mol.GetMaxAtomIdx(), nullptr);
    OEChem::OECopyMol(graph_, mol, copies.data());
    oe_index_of_.assign(graph_.GetMaxAtomIdx(), 0);
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        oe_index_of_[copies[atom->GetIdx()]->GetIdx()] = atom->GetIdx();
    }
    tag_prefix_ = tag_prefix;
}

void ScanResult::Finalize() {
    bool any = false;
    double minimum = 0.0;
    for (const ScanPoint& point : points_) {
        if (point.HasEnergy() && (!any || point.GetEnergy() < minimum)) {
            minimum = point.GetEnergy();
            any = true;
        }
    }
    for (ScanPoint& point : points_) {
        if (point.HasEnergy()) {
            point.SetRelativeEnergy(point.GetEnergy() - minimum);
        } else {
            point.ClearRelativeEnergy();
        }
    }
}

}  // namespace OEVIBE
