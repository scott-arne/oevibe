#ifndef OEVIBE_RESULT_H
#define OEVIBE_RESULT_H

#include <oechem.h>

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "oevibe/Property.h"

namespace OEVIBE {

// NOLINTBEGIN(performance-enum-size)
/// Per-conformer outcome; a failure never aborts the other conformers (spec 7.5).
enum class ConformerStatus { OK, NOT_CONVERGED, ENGINE_ERROR, UNSUPPORTED };
// NOLINTEND(performance-enum-size)

/// :returns: The enumerator name, used in SD tags and messages.
std::string conformer_status_name(ConformerStatus status);

/// Optimizer outcome (spec 5.7). Gradient values are of the projected gradient.
struct OptimizationStats {
    unsigned iterations = 0;
    unsigned gradient_evaluations = 0;
    bool converged = false;
    double energy_change = 0.0;           ///< kcal/mol, last accepted step
    double gradient_norm = 0.0;           ///< kcal/mol/Angstrom
    double max_gradient_component = 0.0;  ///< kcal/mol/Angstrom
    double max_distance_residual = 0.0;   ///< Angstrom
    double max_angle_residual = 0.0;      ///< degrees, angles and dihedrals
    double restraint_energy = 0.0;        ///< kcal/mol, excluded from reported energies
};

/// Where a result came from (spec 5.7).
struct Provenance {
    std::string method;
    std::string engine;
    std::string engine_version;
    std::string solvent_model;  ///< ``gas`` or the resolved model name.
    std::string solvent;        ///< Empty for gas phase.
    std::map<std::string, std::string> resolved_options;
    std::string options_digest;
};

/// Wall time per stage, in seconds.
struct Timings {
    double setup = 0.0;    ///< The call's shared setup, the same for every conformer.
    double compute = 0.0;  ///< This conformer's own computation.
    /// Cumulative wall time from the start of the call to the end of this conformer, including
    /// setup and every earlier conformer, so summing it over conformers overcounts.
    double total = 0.0;
};

/// Ideal-gas, rigid-rotor, quasi-RRHO thermochemistry of one conformer (spec 5.7, 7.3), with
/// the inputs needed to recompute it at another temperature. H = E + ZPE + E_trans + E_rot +
/// E_vib + kT per molecule (RT per mole, the ideal-gas PV term); G = H - T S.
struct Thermochemistry {
    double temperature = 298.15;            ///< K
    double pressure = 1.0;                  ///< atm
    double zpe = 0.0;                       ///< kcal/mol, real modes only
    double enthalpy = 0.0;                  ///< kcal/mol, E + ZPE + thermal
    double entropy = 0.0;                   ///< cal/(mol K)
    double free_energy = 0.0;               ///< kcal/mol, H - T S
    double free_energy_correction = 0.0;    ///< kcal/mol, G - E
    double electronic_energy = 0.0;         ///< kcal/mol, the E used above
    std::vector<double> frequencies;        ///< cm^-1, every vibrational mode (imaginary negative)
    std::vector<double> principal_moments;  ///< 3 values, amu Angstrom^2
    double total_mass = 0.0;                ///< amu
    unsigned symmetry_number = 1;
    bool linear = false;
    unsigned multiplicity = 1;
    double quasi_rrho_cutoff = 100.0;  ///< cm^-1
};

/// One conformer's result. Atom-indexed arrays use the OpenEye layout (spec 3.1), except the
/// Hessian and normal modes, which are dense arrays: entry 3 i + k belongs to the atom whose
/// OpenEye index is ``GetAtomIndices()[i]``. Absent values throw ``OEVibeError`` from their
/// getters.
class ConformerResult {
public:
    unsigned GetConformerIndex() const;
    void SetConformerIndex(unsigned index);
    ConformerStatus GetStatus() const;
    const std::string& GetMessage() const;
    void SetStatus(ConformerStatus status, const std::string& message);
    std::vector<std::string> GetWarnings() const;
    void AddWarning(const std::string& warning);
    /// :returns: The OpenEye atom index of each dense position.
    std::vector<unsigned> GetAtomIndices() const;
    void SetAtomIndices(const std::vector<unsigned>& indices);

    bool HasEnergy() const;
    double GetEnergy() const;  ///< kcal/mol
    void SetEnergy(double energy);
    bool HasRelativeEnergy() const;
    double GetRelativeEnergy() const;  ///< kcal/mol, set by the owning ResultSet
    void SetRelativeEnergy(double energy);
    void ClearRelativeEnergy();
    bool HasCoordinates() const;
    std::vector<double> GetCoordinates() const;  ///< Angstrom, OpenEye layout
    void SetCoordinates(const std::vector<double>& coordinates);
    bool HasGradient() const;
    std::vector<double> GetGradient() const;  ///< kcal/mol/Angstrom, OpenEye layout
    void SetGradient(const std::vector<double>& gradient);
    bool HasCharges() const;
    std::vector<double> GetCharges() const;  ///< e, OpenEye layout
    double GetCharge(const OEChem::OEAtomBase& atom) const;
    void SetCharges(const std::vector<double>& charges);
    bool HasDipole() const;
    std::vector<double> GetDipole() const;  ///< Debye, (x, y, z)
    void SetDipole(const std::vector<double>& dipole);
    bool HasOrbitalEnergies() const;
    std::vector<double> GetOrbitalEnergies() const;  ///< eV, ascending
    bool HasHOMO() const;
    double GetHOMO() const;
    bool HasLUMO() const;
    double GetLUMO() const;
    /// :raises OEVibeError: Unless ``homo_index >= -1`` and ``homo_index < energies.size()``.
    void SetOrbitalEnergies(const std::vector<double>& energies, int homo_index);
    bool HasBondOrders() const;
    /// :returns: Bond orders keyed by OpenEye atom indices with ``i < j``.
    std::vector<BondOrder> GetBondOrders() const;
    /// :returns: The bond order, or 0 for a pair without one.
    double GetBondOrder(unsigned oe_a, unsigned oe_b) const;
    double GetBondOrder(const OEChem::OEBondBase& bond) const;
    void SetBondOrders(const std::vector<BondOrder>& orders);
    bool HasHessian() const;
    /// :returns: 9N^2 values, row-major, kcal/mol/Angstrom^2, dense order.
    std::vector<double> GetHessian() const;
    /// :raises OEVibeError: Unless the size is (3N)^2 for some N >= 1.
    void SetHessian(const std::vector<double>& hessian);
    bool HasFrequencies() const;
    /// :returns: cm^-1, ascending, imaginary modes negative.
    std::vector<double> GetFrequencies() const;
    /// :returns: One row of 3N Cartesian displacements per frequency, unit norm, dense order.
    std::vector<double> GetNormalModes() const;
    unsigned GetImaginaryModeCount() const;
    /// :raises OEVibeError: Unless every mode has the same length, a multiple of 3, and
    ///     ``imaginary_count`` equals the number of negative frequencies.
    void SetFrequencies(const std::vector<double>& frequencies,
                        const std::vector<double>& normal_modes, unsigned imaginary_count);
    bool HasThermochemistry() const;
    const Thermochemistry& GetThermochemistry() const;
    void SetThermochemistry(const Thermochemistry& thermochemistry);
    bool HasOptimizationStats() const;
    const OptimizationStats& GetOptimizationStats() const;
    void SetOptimizationStats(const OptimizationStats& stats);
    const Provenance& GetProvenance() const;
    void SetProvenance(const Provenance& provenance);
    const Timings& GetTimings() const;
    void SetTimings(const Timings& timings);

private:
    unsigned conformer_index_ = 0;
    ConformerStatus status_ = ConformerStatus::ENGINE_ERROR;
    std::string message_;
    std::vector<std::string> warnings_;
    std::vector<unsigned> atom_indices_;
    std::optional<double> energy_;
    std::optional<double> relative_energy_;
    std::optional<std::vector<double>> coordinates_;
    std::optional<std::vector<double>> gradient_;
    std::optional<std::vector<double>> charges_;
    std::optional<std::vector<double>> dipole_;
    std::optional<std::vector<double>> orbital_energies_;
    int homo_index_ = -1;
    std::optional<std::map<std::pair<unsigned, unsigned>, double>> bond_orders_;
    std::optional<std::vector<double>> hessian_;
    std::optional<std::vector<double>> frequencies_;
    std::vector<double> normal_modes_;  // Meaningful only while frequencies_ is set.
    unsigned imaginary_count_ = 0;
    std::optional<Thermochemistry> thermochemistry_;
    std::optional<OptimizationStats> optimization_stats_;
    Provenance provenance_;
    Timings timings_;
};

/// Results of one calculation call, one per input conformer in input order (spec 5.7).
class ResultSet {
public:
    std::size_t Size() const;
    /// :raises OEVibeError: For an index past the end.
    const ConformerResult& Get(std::size_t index) const;
    bool HasLowest() const;
    /// :returns: The lowest-energy OK conformer; ties go to the lowest conformer index.
    /// :raises OEVibeError: When no conformer is OK, or when OK conformers have different
    ///     methods, solvent models, or solvents (spec 9, 10).
    const ConformerResult& GetLowest() const;
    /// :returns: Conformer index to relative energy (kcal/mol) over OK conformers.
    std::map<unsigned, double> GetRelativeEnergies() const;
    /// Weights over OK conformers, w_i = exp(-(X_i - X_min) / RT) / sum_j exp(-(X_j - X_min) / RT)
    /// (spec 5.7). When every OK conformer has thermochemistry, X is its free energy recomputed
    /// at ``temperature`` (keeping its stored pressure); when none has it, X is the electronic
    /// energy. No OK conformer gives an empty map.
    /// :raises OEVibeError: Unless the temperature is finite and greater than 0; when only some
    ///     OK conformers have thermochemistry; or when OK conformers have different methods,
    ///     solvent models, or solvents (spec 9, 10).
    std::map<unsigned, double> GetBoltzmannWeights(double temperature) const;

    void Add(ConformerResult result);
    /// Sets each OK conformer's relative energy; call once after the last ``Add``.
    /// :raises OEVibeError: When OK conformers have different methods, solvent models, or
    ///     solvents (spec 9, 10).
    void Finalize();
    void AddWarningToAll(const std::string& warning);

private:
    /// Throws OEVibeError if any two OK results that have energies differ in method, solvent
    /// model, or solvent.
    void RequireComparable() const;

    std::vector<ConformerResult> results_;
};

// NOLINTBEGIN(performance-enum-size)
/// Per-grid-point outcome of a scan (spec 7.5).
enum class ScanPointStatus { OK, NOT_CONVERGED, ENGINE_ERROR, UNSUPPORTED, NOT_ATTEMPTED };
// NOLINTEND(performance-enum-size)

/// :returns: The enumerator name, used in SD tags and messages.
std::string scan_point_status_name(ScanPointStatus status);

/// One grid point of a scan. Coordinates use the scanned molecule's OpenEye layout (spec 3.1).
/// Absent values throw ``OEVibeError`` from their getters.
class ScanPoint {
public:
    /// :returns: Degrees in (-180, 180], one per scanned dihedral.
    std::vector<double> GetAngles() const;
    void SetAngles(const std::vector<double>& angles);
    ScanPointStatus GetStatus() const;
    const std::string& GetMessage() const;
    void SetStatus(ScanPointStatus status, const std::string& message);
    bool HasEnergy() const;
    double GetEnergy() const;  ///< kcal/mol, engine energy of the point's best result
    void SetEnergy(double energy);
    bool HasRelativeEnergy() const;
    double GetRelativeEnergy() const;  ///< kcal/mol, set by the owning ScanResult
    void SetRelativeEnergy(double energy);
    void ClearRelativeEnergy();
    bool HasCoordinates() const;
    std::vector<double> GetCoordinates() const;  ///< Angstrom, OpenEye layout
    void SetCoordinates(const std::vector<double>& coordinates);

private:
    std::vector<double> angles_;
    ScanPointStatus status_ = ScanPointStatus::NOT_ATTEMPTED;
    std::string message_;
    std::optional<double> energy_;
    std::optional<double> relative_energy_;
    std::optional<std::vector<double>> coordinates_;
};

/// A relaxed dihedral scan (spec 5.7): one point per grid point, in grid order (row-major, the
/// first dihedral varying slowest).
class ScanResult {
public:
    unsigned GetNumDihedrals() const;
    /// :returns: The dihedral's four OpenEye atom indices.
    /// :raises OEVibeError: For an index past the end.
    std::vector<unsigned> GetDihedral(unsigned index) const;
    std::size_t GetNumPoints() const;
    /// :raises OEVibeError: For an index past the end.
    const ScanPoint& GetPoint(std::size_t index) const;
    /// :returns: Whether the optimization cap stopped the scan before propagation finished.
    bool IsTruncated() const;
    std::vector<std::string> GetWarnings() const;
    const Provenance& GetProvenance() const;
    /// :returns: A copy of the scanned molecule's graph with one conformer per point that has an
    ///     energy, in grid order, each carrying the SD data ``<prefix>ScanAngles`` (degrees, comma
    ///     separated in dihedral order), ``<prefix>Energy``, ``<prefix>RelativeEnergy`` (kcal/mol),
    ///     and ``<prefix>Status``, where ``<prefix>`` is ``OEVibe.<method>.<solvent-tag>.``
    ///     (spec 6.5).
    OEChem::OEMol GetMolecule() const;
    /// Replaces the contents of ``out`` with ``GetMolecule()``'s. Python receives the molecule
    /// this way, through the cross-runtime reference typemap, as for
    /// ``MolecularSystem::CopyMolecule`` (spec 6.2).
    void CopyMolecule(OEChem::OEMCMolBase& out) const;

    void AddDihedral(const std::vector<unsigned>& atoms);
    void AddPoint(ScanPoint point);
    void SetTruncated(bool truncated);
    void AddWarning(const std::string& warning);
    void SetProvenance(const Provenance& provenance);
    /// Copies the graph of the scanned molecule, whose OpenEye layout the points' coordinates
    /// use, and the SD tag prefix ``OEVibe.<method>.<solvent-tag>.`` of ``GetMolecule()``.
    void SetMolecule(const OEChem::OEMolBase& mol, const std::string& tag_prefix);
    /// Sets the relative energy of every point that has an energy, from the lowest (spec 5.7);
    /// call once after the last ``AddPoint``.
    void Finalize();

private:
    std::vector<std::vector<unsigned>> dihedrals_;
    std::vector<ScanPoint> points_;
    bool truncated_ = false;
    std::vector<std::string> warnings_;
    Provenance provenance_;
    OEChem::OEGraphMol graph_;
    std::vector<unsigned> oe_index_of_;  ///< Scanned molecule's atom index of each graph atom.
    std::string tag_prefix_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_RESULT_H
