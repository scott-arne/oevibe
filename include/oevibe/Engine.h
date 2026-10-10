#ifndef OEVIBE_ENGINE_H
#define OEVIBE_ENGINE_H

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "oevibe/Method.h"
#include "oevibe/MolecularSystem.h"
#include "oevibe/Preset.h"
#include "oevibe/Property.h"
#include "oevibe/Solvent.h"

namespace OEVIBE {

// NOLINTBEGIN(performance-enum-size)
enum class KeywordType { BOOL, INT, DOUBLE, STRING };
// NOLINTEND(performance-enum-size)

/// One keyword an engine accepts through ``CalculationOptions::SetKeyword`` (spec 8.1). Bounds
/// are inclusive and apply to INT and DOUBLE keywords.
struct KeywordSpec {
    std::string name;
    KeywordType type = KeywordType::STRING;
    std::string description;
    std::vector<Method> methods;  ///< Methods it applies to; empty means all of the engine's.
    bool has_min = false;
    double min = 0.0;
    bool has_max = false;
    double max = 0.0;
};

/// What one method of an engine supports (spec 8.1).
struct MethodCapabilities {
    Method method;
    bool available = true;                     ///< Per-method availability within the engine.
    std::string unavailable_reason;            ///< Set when ``available`` is false.
    std::vector<SolventModel> solvent_models;  ///< Empty means gas phase only; first is AUTO's.
    std::vector<std::string> solvents;         ///< Accepted names; empty means any the model takes.
    unsigned properties = 0;                   ///< Bit set of ``Property`` values.
    bool analytic_gradient = true;
    bool analytic_hessian = false;
    bool requires_partial_charges = false;
    bool charge_from_topology = false;  ///< Total charge is fixed by atom formal charges.
    bool open_shell = false;
    bool force_field = false;        ///< Drives AUTO optimizer selection (spec 7.1).
    std::vector<unsigned> elements;  ///< Supported atomic numbers; empty skips the check.
};

/// Resolved settings handed to an engine (spec 8.1).
struct EngineSettings {
    Method method;
    Solvent solvent;  ///< Gas phase or a concrete model.
    Preset preset = Preset::STANDARD;
    double accuracy = 1.0;    ///< Preset engine accuracy; an ``accuracy`` keyword overrides it.
    unsigned properties = 0;  ///< Requested ``Property`` bits.
    std::map<std::string, std::string> keywords;  ///< Validated against the keyword schema.
};

/// What to compute at one geometry.
struct EvaluationRequest {
    bool gradient = false;
    bool hessian = false;     ///< Only when the method declares an analytic Hessian.
    unsigned properties = 0;  ///< ``Property`` bits.
};

/// Engine output in spec 3.1 units and dense atom order. An empty vector means absent; each
/// requested quantity is present and nothing else is. ``energy`` has no meaningful default, so
/// it starts as NaN and ``validate_evaluation`` rejects an engine that never set it.
struct Evaluation {
    /// kcal/mol; must be set.
    double energy = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> gradient;          ///< 3N, kcal/mol/Angstrom
    std::vector<double> hessian;           ///< 9N^2 row-major, kcal/mol/Angstrom^2
    std::vector<double> charges;           ///< N, e
    std::vector<double> dipole;            ///< 3, Debye
    std::vector<double> orbital_energies;  ///< ascending, eV
    int homo_index = -1;                   ///< -1 when there are no occupied orbitals
    std::vector<BondOrder> bond_orders;
    std::vector<std::string> messages;
};

/// Evaluates one independent trajectory; never shared across conformers (spec 6.4).
class EngineSession {
public:
    virtual ~EngineSession() = default;
    /// :param coords_angstrom: Dense 3N coordinates in Angstrom.
    virtual Evaluation Evaluate(const std::vector<double>& coords_angstrom,
                                const EvaluationRequest& request) = 0;
};

/// A program that evaluates energies (spec 8.1). Built once, reused across molecules.
class Engine {
public:
    virtual ~Engine() = default;
    virtual std::string GetName() const = 0;
    virtual std::string GetVersion() const = 0;
    virtual bool IsAvailable() const = 0;
    virtual std::string GetUnavailableReason() const = 0;
    virtual std::vector<MethodCapabilities> GetMethods() const = 0;
    /// :returns: The keyword schema; the default accepts no keywords.
    virtual std::vector<KeywordSpec> GetKeywordSpecs() const;
    /// Default checks the method, and each keyword's existence, type, bounds, and method
    /// applicability against ``GetKeywordSpecs()``.
    /// :raises OptionsError: Listing every problem found.
    virtual void ValidateSettings(const EngineSettings& settings) const;
    /// Default checks elements, open-shell support, and, for ``charge_from_topology`` methods,
    /// that the total charge equals the snapshot's formal-charge sum.
    /// :raises InputError: Listing every problem found.
    virtual void ValidateSystem(const MolecularSystem& system,
                                const EngineSettings& settings) const;
    /// :raises UnsupportedError: For a system the engine cannot handle.
    virtual std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                         const EngineSettings& settings) = 0;
};

/// :returns: The capabilities of ``method`` in ``engine``, or nothing when it is not provided.
std::optional<MethodCapabilities> find_method(const Engine& engine, const Method& method);

/// :returns: An empty string for a valid evaluation, otherwise the first problem found.
std::string validate_evaluation(const Evaluation& evaluation, const EvaluationRequest& request,
                                unsigned num_atoms);

}  // namespace OEVIBE

#endif  // OEVIBE_ENGINE_H
