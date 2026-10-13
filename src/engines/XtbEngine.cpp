#include "oevibe/engines/XtbEngine.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "engines/ElectronicProperties.h"
#include "engines/RuntimeLibrary.h"
#include "engines/XtbLibrary.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"

namespace OEVIBE {

namespace detail {

/// What one attempt to load libxtb found; never changed after it is built.
struct XtbEngineState {
    std::shared_ptr<const XtbLibrary> library;  ///< Null when the engine is unavailable.
    std::string unavailable_reason;
    std::string gfn0_parameter_file;  ///< Empty when GFN0-xTB is unavailable.
    std::string gfn0_unavailable_reason;
};

}  // namespace detail

namespace {

constexpr unsigned XTB_LAST_ELEMENT = 86;  // radon, the last element GFN0-xTB parametrizes
constexpr int XTB_VERBOSITY_MUTED = 0;
constexpr std::size_t XTB_ERROR_BUFFER_SIZE = 4096;

struct DefaultLibraryPath {
    std::mutex mutex;
    std::string path;
};

DefaultLibraryPath& default_library_path() {
    static DefaultLibraryPath value;
    return value;
}

std::vector<unsigned> supported_elements() {
    std::vector<unsigned> elements;
    for (unsigned z = 1; z <= XTB_LAST_ELEMENT; ++z) {
        elements.push_back(z);
    }
    return elements;
}

std::shared_ptr<const detail::XtbEngineState> load_xtb(const std::string& explicit_path) {
    auto state = std::make_shared<detail::XtbEngineState>();
    detail::LibrarySearch search;
    search.engine = "xtb";
    search.explicit_path = explicit_path;
    search.environment_variable = "OEVIBE_XTB_LIBRARY";
    search.file_names = detail::library_file_names("xtb", "6");
    std::string reason;
    std::shared_ptr<detail::RuntimeLibrary> runtime = detail::RuntimeLibrary::Load(search, reason);
    if (!runtime) {
        state->unavailable_reason = reason;
        return state;
    }
    state->library = detail::XtbLibrary::Open(std::move(runtime), reason);
    if (!state->library) {
        state->unavailable_reason = reason;
        return state;
    }
    const std::string& path = state->library->GetPath();
    state->gfn0_parameter_file = detail::xtb_gfn0_parameter_file(path);
    if (state->gfn0_parameter_file.empty()) {
        state->gfn0_unavailable_reason = detail::xtb_gfn0_missing_reason(path);
    }
    return state;
}

std::vector<double> to_bohr(const std::vector<double>& angstrom) {
    std::vector<double> bohr(angstrom.size());
    for (std::size_t k = 0; k < angstrom.size(); ++k) {
        bohr[k] = angstrom[k] * Units::ANGSTROM_TO_BOHR;
    }
    return bohr;
}

/// One GFN0-xTB trajectory. The environment, molecule, calculator, and results are built by the
/// first evaluation, from the coordinates that evaluation is given, and live as long as the
/// session; each later evaluation only moves the atoms (spec 8.3). The session never reads a
/// conformer from the system, because a session is never shared across conformers and a result
/// must not depend on which conformer the system lists first (spec 6.4).
class XtbSession : public EngineSession {
public:
    XtbSession(std::shared_ptr<const detail::XtbLibrary> library, const MolecularSystem& system,
               double accuracy, const std::string& parameter_file)
        : library_(std::move(library)),
          num_atoms_(system.NumAtoms()),
          numbers_(system.GetAtomicNumbers().begin(), system.GetAtomicNumbers().end()),
          charge_(system.GetCharge()),
          unpaired_(static_cast<int>(system.GetMultiplicity()) - 1),
          accuracy_(accuracy),
          // xtb_loadGFN0xTB takes a non-const char*, so it receives a private copy of the path.
          parameter_path_(parameter_file.begin(), parameter_file.end()) {
        parameter_path_.push_back('\0');
    }

    ~XtbSession() override {
        std::lock_guard<std::mutex> lock(detail::xtb_mutex());
        ReleaseLocked();
    }

    XtbSession(const XtbSession&) = delete;
    XtbSession& operator=(const XtbSession&) = delete;

    Evaluation Evaluate(const std::vector<double>& coords_angstrom,
                        const EvaluationRequest& request) override {
        const std::size_t n = num_atoms_;
        if (coords_angstrom.size() != 3 * n) {
            throw std::runtime_error("the xtb session expects " + std::to_string(3 * n) +
                                     " coordinates, got " + std::to_string(coords_angstrom.size()));
        }
        if (request.hessian) {
            throw std::runtime_error("xtb provides no analytic Hessian");
        }
        const auto wants = [&request](Property property) {
            return (request.properties & static_cast<unsigned>(property)) != 0;
        };
        const std::vector<double> positions = to_bohr(coords_angstrom);
        const detail::XtbFunctions& xtb = library_->Functions();
        double energy = 0.0;
        std::vector<double> gradient;
        std::vector<double> charges;
        std::vector<double> dipole;
        std::vector<double> bond_matrix;
        std::vector<double> eigenvalues;
        std::vector<double> occupations;
        {
            std::lock_guard<std::mutex> lock(detail::xtb_mutex());
            if (!initialized_) {
                InitializeLocked(positions);
            } else {
                xtb.update_molecule(environment_, molecule_, positions.data(), nullptr);
                CheckLocked("xtb_updateMolecule");
            }
            xtb.singlepoint(environment_, molecule_, calculator_, results_);
            CheckLocked("xtb_singlepoint");
            xtb.get_energy(environment_, results_, &energy);
            CheckLocked("xtb_getEnergy");
            if (request.gradient) {
                gradient.resize(3 * n);
                xtb.get_gradient(environment_, results_, gradient.data());
                CheckLocked("xtb_getGradient");
            }
            if (wants(Property::CHARGES)) {
                charges.resize(n);
                xtb.get_charges(environment_, results_, charges.data());
                CheckLocked("xtb_getCharges");
            }
            if (wants(Property::DIPOLE)) {
                dipole.resize(3);
                xtb.get_dipole(environment_, results_, dipole.data());
                CheckLocked("xtb_getDipole");
            }
            if (wants(Property::BOND_ORDERS)) {
                bond_matrix.resize(n * n);
                xtb.get_bond_orders(environment_, results_, bond_matrix.data());
                CheckLocked("xtb_getBondOrders");
            }
            if (wants(Property::ORBITAL_ENERGIES)) {
                int nao = 0;
                xtb.get_nao(environment_, results_, &nao);
                CheckLocked("xtb_getNao");
                if (nao <= 0) {
                    throw std::runtime_error("xtb reported no orbitals");
                }
                eigenvalues.resize(static_cast<std::size_t>(nao));
                xtb.get_orbital_eigenvalues(environment_, results_, eigenvalues.data());
                CheckLocked("xtb_getOrbitalEigenvalues");
                // C API 1.0 writes nao spin-summed occupations (probe 2026-10-11).
                occupations.resize(static_cast<std::size_t>(nao));
                xtb.get_orbital_occupations(environment_, results_, occupations.data());
                CheckLocked("xtb_getOrbitalOccupations");
            }
        }

        Evaluation evaluation;
        evaluation.energy = energy * Units::HARTREE_TO_KCAL;
        for (double component : gradient) {
            evaluation.gradient.push_back(component * Units::HARTREE_PER_BOHR_TO_KCAL_PER_ANGSTROM);
        }
        evaluation.charges = std::move(charges);
        for (double component : dipole) {
            evaluation.dipole.push_back(component * Units::EBOHR_TO_DEBYE);
        }
        if (wants(Property::BOND_ORDERS)) {
            // libxtb fills a column-major matrix, but Wiberg bond orders are symmetric, so the
            // row-major reading wiberg_pairs expects sees the same values.
            evaluation.bond_orders = detail::wiberg_pairs(bond_matrix, num_atoms_);
        }
        if (wants(Property::ORBITAL_ENERGIES)) {
            for (double value : eigenvalues) {
                evaluation.orbital_energies.push_back(value * Units::HARTREE_TO_EV);
            }
            evaluation.homo_index = detail::homo_index(occupations);
        }
        return evaluation;
    }

private:
    /// Builds the libxtb objects around ``positions_bohr``; the caller holds
    /// ``detail::xtb_mutex()``. A failure releases everything, so the next evaluation starts again
    /// from its own coordinates.
    ///
    /// :raises std::runtime_error: With libxtb's message when it rejects the structure.
    void InitializeLocked(const std::vector<double>& positions_bohr) {
        const detail::XtbFunctions& xtb = library_->Functions();
        const int natoms = static_cast<int>(num_atoms_);
        try {
            environment_ = xtb.new_environment();
            if (environment_ == nullptr) {
                throw std::runtime_error("xtb_newEnvironment returned no environment");
            }
            xtb.set_verbosity(environment_, XTB_VERBOSITY_MUTED);
            molecule_ =
                xtb.new_molecule(environment_, &natoms, numbers_.data(), positions_bohr.data(),
                                 &charge_, &unpaired_, nullptr, nullptr);
            CheckLocked("xtb_newMolecule");
            calculator_ = xtb.new_calculator();
            results_ = xtb.new_results();
            if (calculator_ == nullptr || results_ == nullptr) {
                throw std::runtime_error("xtb could not allocate a calculator or results object");
            }
            xtb.load_gfn0(environment_, molecule_, calculator_, parameter_path_.data());
            CheckLocked("xtb_loadGFN0xTB");
            // libxtb rejects an accuracy until a calculator is loaded (probe 2026-10-11).
            xtb.set_accuracy(environment_, calculator_, accuracy_);
            CheckLocked("xtb_setAccuracy");
        } catch (...) {
            ReleaseLocked();
            throw;
        }
        initialized_ = true;
    }

    /// Releases every handle; the caller holds ``detail::xtb_mutex()``.
    void ReleaseLocked() noexcept {
        const detail::XtbFunctions& xtb = library_->Functions();
        if (results_ != nullptr) {
            xtb.del_results(&results_);
            results_ = nullptr;
        }
        if (calculator_ != nullptr) {
            xtb.del_calculator(&calculator_);
            calculator_ = nullptr;
        }
        if (molecule_ != nullptr) {
            xtb.del_molecule(&molecule_);
            molecule_ = nullptr;
        }
        if (environment_ != nullptr) {
            xtb.del_environment(&environment_);
            environment_ = nullptr;
        }
    }

    /// Turns libxtb's error stack into an exception; the caller holds ``detail::xtb_mutex()``.
    /// :raises std::runtime_error: With libxtb's message when the environment holds an error.
    void CheckLocked(const char* call) const {
        const detail::XtbFunctions& xtb = library_->Functions();
        if (xtb.check_environment(environment_) == 0) {
            return;
        }
        std::vector<char> buffer(XTB_ERROR_BUFFER_SIZE, '\0');
        const int size = static_cast<int>(buffer.size());
        xtb.get_error(environment_, buffer.data(), &size);
        buffer.back() = '\0';
        std::string message(buffer.data());
        // libxtb stacks one line per frame; a result message reads better as one line.
        std::replace(message.begin(), message.end(), '\n', ' ');
        throw std::runtime_error(std::string("xtb ") + call +
                                 " failed: " + (message.empty() ? "no message" : message));
    }

    std::shared_ptr<const detail::XtbLibrary> library_;
    unsigned num_atoms_;
    std::vector<int> numbers_;
    double charge_;
    int unpaired_;
    double accuracy_;
    std::vector<char> parameter_path_;
    bool initialized_ = false;  ///< Guarded by ``detail::xtb_mutex()``.
    detail::XtbHandle environment_ = nullptr;
    detail::XtbHandle molecule_ = nullptr;
    detail::XtbHandle calculator_ = nullptr;
    detail::XtbHandle results_ = nullptr;
};

}  // namespace

XtbEngine::XtbEngine(std::string library_path) : library_path_(std::move(library_path)) {}

void XtbEngine::SetLibraryPath(const std::string& path) {
    DefaultLibraryPath& value = default_library_path();
    std::lock_guard<std::mutex> lock(value.mutex);
    value.path = path;
}

std::shared_ptr<const detail::XtbEngineState> XtbEngine::Load() const {
    // ``load_mutex_`` is taken before the default-path mutex, never the other way round.
    std::lock_guard<std::mutex> lock(load_mutex_);
    // Only a state that can run GFN0 is final. A library whose installation has no parameter
    // file leaves the engine half loaded, and that half is what a later path has to repair.
    if (state_ && state_->library && !state_->gfn0_parameter_file.empty()) {
        return state_;
    }
    std::string explicit_path = library_path_;
    if (explicit_path.empty()) {
        DefaultLibraryPath& value = default_library_path();
        std::lock_guard<std::mutex> default_lock(value.mutex);
        explicit_path = value.path;
    }
    // The registry's built-in reads the default path, and a method lookup reaches it before a
    // caller can set the path, so an incomplete load is retried when the path it used has
    // changed. An unchanged path keeps its outcome, and a usable state is never replaced.
    if (!state_ || explicit_path != attempted_path_) {
        std::shared_ptr<const detail::XtbEngineState> reloaded = load_xtb(explicit_path);
        attempted_path_ = explicit_path;
        // A retry that loads nothing keeps the library the engine already holds, which is
        // still open and still the one its sessions run on.
        if (!state_ || reloaded->library) {
            state_ = std::move(reloaded);
        }
    }
    // A caller may hold the returned state while a later call replaces an incomplete one.
    return state_;
}

std::string XtbEngine::GetName() const { return "xtb"; }

std::string XtbEngine::GetVersion() const {
    const std::shared_ptr<const detail::XtbEngineState> state = Load();
    if (!state->library) {
        return "";
    }
    // C API 1.0 has no call that reports the xtb release, so the C API version stands in.
    return "C API " + detail::xtb_api_version_text(state->library->GetApiVersion());
}

bool XtbEngine::IsAvailable() const { return Load()->library != nullptr; }

std::string XtbEngine::GetUnavailableReason() const { return Load()->unavailable_reason; }

std::vector<MethodCapabilities> XtbEngine::GetMethods() const {
    const std::shared_ptr<const detail::XtbEngineState> state = Load();
    MethodCapabilities gfn0;
    gfn0.method = Method(GFN0_XTB);
    // Spike S4: the C API returns charges, dipole, bond orders, and orbitals for GFN0-xTB, and
    // GFN0-xTB has no implicit solvent, so solvent_models stays empty (gas phase only).
    gfn0.properties = ALL_PROPERTIES;
    gfn0.open_shell = true;
    gfn0.elements = supported_elements();
    if (!state->library) {
        gfn0.available = false;
        gfn0.unavailable_reason = state->unavailable_reason;
    } else if (state->gfn0_parameter_file.empty()) {
        gfn0.available = false;
        gfn0.unavailable_reason = state->gfn0_unavailable_reason;
    }

    MethodCapabilities gfnff;
    gfnff.method = Method(GFN_FF);
    gfnff.available = false;
    gfnff.unavailable_reason = detail::XTB_GFNFF_UNAVAILABLE_REASON;
    // What spike S4 measured through the C API (GBSA with the gsolv reference state, and the
    // dipole as the only property), declared so that restoring the method later changes only
    // its availability.
    gfnff.solvent_models = {SolventModel::GBSA};
    gfnff.properties = static_cast<unsigned>(Property::DIPOLE);
    gfnff.force_field = true;
    gfnff.elements = supported_elements();
    return {gfn0, gfnff};
}

std::vector<KeywordSpec> XtbEngine::GetKeywordSpecs() const {
    // max_iterations and electronic_temperature are not offered: C API 1.0 accepts them only
    // for self-consistent calculators, and GFN0-xTB is not self-consistent (spec 8.5).
    KeywordSpec accuracy;
    accuracy.name = "accuracy";
    accuracy.type = KeywordType::DOUBLE;
    accuracy.description =
        "Numerical accuracy passed to xtb_setAccuracy (smaller is tighter); "
        "overrides the preset's engine accuracy";
    accuracy.has_min = true;
    accuracy.min = 1e-4;
    accuracy.has_max = true;
    accuracy.max = 1e3;
    return {accuracy};
}

std::shared_ptr<EngineSession> XtbEngine::CreateSession(const MolecularSystem& system,
                                                        const EngineSettings& settings) {
    const std::shared_ptr<const detail::XtbEngineState> state = Load();
    if (!state->library) {
        throw EngineUnavailableError("engine 'xtb' is unavailable: " + state->unavailable_reason);
    }
    const std::string& method = settings.method.GetName();
    if (method == GFN_FF) {
        throw EngineUnavailableError("method 'gfn-ff' of engine 'xtb' is unavailable: " +
                                     std::string(detail::XTB_GFNFF_UNAVAILABLE_REASON));
    }
    if (method != GFN0_XTB) {
        throw OptionsError("engine 'xtb' does not provide method '" + method + "'");
    }
    if (state->gfn0_parameter_file.empty()) {
        throw EngineUnavailableError("method 'gfn0-xtb' of engine 'xtb' is unavailable: " +
                                     state->gfn0_unavailable_reason);
    }
    if (!settings.solvent.IsGasPhase()) {
        throw OptionsError("method 'gfn0-xtb' supports gas phase only; solvent '" +
                           settings.solvent.GetName() + "' cannot be modeled");
    }
    return std::make_shared<XtbSession>(state->library, system, settings.accuracy,
                                        state->gfn0_parameter_file);
}

}  // namespace OEVIBE
