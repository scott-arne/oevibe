#include "oevibe/engines/TbliteEngine.h"

#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

#include "engines/ElectronicProperties.h"
#include "engines/RuntimeLibrary.h"
#include "engines/TbliteLibrary.h"
#include "oevibe/Error.h"
#include "oevibe/Units.h"

namespace OEVIBE {

namespace {

constexpr const char* TBLITE_LIBRARY_VARIABLE = "OEVIBE_TBLITE_LIBRARY";
/// CODATA 2018 Boltzmann constant in Hartree per kelvin; tblite takes the electronic
/// temperature as an energy.
constexpr double BOLTZMANN_HARTREE_PER_KELVIN = 3.1668115634556e-6;
// tblite's solvation parameter sets: ALPB and GBSA, each fitted for GFN1 and for GFN2.
constexpr int ALPB_GFN1 = 11;
constexpr int ALPB_GFN2 = 12;
constexpr int GBSA_GFN1 = 21;
constexpr int GBSA_GFN2 = 22;
/// tblite's "gsolv" reference state, the default of the xtb binary's --alpb (spike S5).
constexpr int GSOLV_REFERENCE_STATE = 1;
constexpr int ERROR_BUFFER_SIZE = 1024;

std::mutex& default_path_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string& default_path() {
    static std::string path;
    return path;
}

bool wants(unsigned properties, Property property) {
    return (properties & static_cast<unsigned>(property)) != 0;
}

std::vector<double> to_bohr(const std::vector<double>& angstrom) {
    std::vector<double> bohr(angstrom.size());
    for (std::size_t k = 0; k < angstrom.size(); ++k) {
        bohr[k] = angstrom[k] * Units::ANGSTROM_TO_BOHR;
    }
    return bohr;
}

/// One structure, calculator, and result over a lifetime of evaluations, so each SCF starts from
/// the previous wavefunction (spec 8.3). Every library call holds ``tblite_mutex()``.
class TbliteSession : public EngineSession {
public:
    /// :raises std::runtime_error: With tblite's message when the library rejects the system,
    ///     the settings, or the solvent.
    TbliteSession(std::shared_ptr<const detail::TbliteLibrary> library,
                  const MolecularSystem& system, const EngineSettings& settings)
        : library_(std::move(library)), num_atoms_(system.NumAtoms()) {
        const std::lock_guard<std::mutex> lock(detail::tblite_mutex());
        try {
            Initialize(system, settings);
        } catch (...) {
            Release();
            throw;
        }
    }

    ~TbliteSession() override {
        const std::lock_guard<std::mutex> lock(detail::tblite_mutex());
        Release();
    }

    TbliteSession(const TbliteSession&) = delete;
    TbliteSession& operator=(const TbliteSession&) = delete;

    Evaluation Evaluate(const std::vector<double>& coords_angstrom,
                        const EvaluationRequest& request) override {
        const std::size_t dim = 3 * static_cast<std::size_t>(num_atoms_);
        if (coords_angstrom.size() != dim) {
            throw std::invalid_argument("expected " + std::to_string(dim) + " coordinates, got " +
                                        std::to_string(coords_angstrom.size()));
        }
        if (request.hessian) {
            throw std::runtime_error("tblite provides no analytic Hessian");
        }
        const detail::TbliteApi& api = library_->api;
        const std::vector<double> positions = to_bohr(coords_angstrom);
        Evaluation evaluation;
        const std::lock_guard<std::mutex> lock(detail::tblite_mutex());
        api.update_structure_geometry(error_, structure_, positions.data(), nullptr);
        CheckError("tblite_update_structure_geometry");
        api.get_singlepoint(context_, structure_, calculator_, result_);
        CheckContext("tblite_get_singlepoint");
        double energy = 0.0;
        api.get_result_energy(error_, result_, &energy);
        CheckError("tblite_get_result_energy");
        evaluation.energy = energy * Units::HARTREE_TO_KCAL;
        if (request.gradient) {
            evaluation.gradient.assign(dim, 0.0);
            api.get_result_gradient(error_, result_, evaluation.gradient.data());
            CheckError("tblite_get_result_gradient");
            for (double& component : evaluation.gradient) {
                component *= Units::HARTREE_PER_BOHR_TO_KCAL_PER_ANGSTROM;
            }
        }
        ReadProperties(request.properties, evaluation);
        return evaluation;
    }

private:
    void Initialize(const MolecularSystem& system, const EngineSettings& settings) {
        const detail::TbliteApi& api = library_->api;
        context_ = api.new_context();
        error_ = api.new_error();
        if (context_ == nullptr || error_ == nullptr) {
            throw std::runtime_error("tblite could not allocate a context");
        }
        // Verbosity 0 keeps tblite silent on stdout and stderr (spec 8.3).
        api.set_context_verbosity(context_, 0);
        CheckContext("tblite_set_context_verbosity");

        const std::vector<unsigned>& atomic_numbers = system.GetAtomicNumbers();
        const std::vector<int> numbers(atomic_numbers.begin(), atomic_numbers.end());
        // The first evaluation moves the structure to its own coordinates; any conformer of the
        // system is a valid starting geometry.
        const std::vector<double> positions = to_bohr(system.GetConformerCoordinates(0));
        const double charge = system.GetCharge();
        const int unpaired = static_cast<int>(system.GetMultiplicity()) - 1;
        structure_ = api.new_structure(error_, static_cast<int>(num_atoms_), numbers.data(),
                                       positions.data(), &charge, &unpaired, nullptr, nullptr);
        CheckError("tblite_new_structure");

        const bool gfn1 = settings.method == Method(GFN1_XTB);
        if (!gfn1 && settings.method != Method(GFN2_XTB)) {
            throw OptionsError("engine 'tblite' does not provide method '" +
                               settings.method.GetName() + "'");
        }
        calculator_ = (gfn1 ? api.new_gfn1_calculator : api.new_gfn2_calculator)(
            context_, structure_, nullptr);
        CheckContext(gfn1 ? "tblite_new_gfn1_calculator" : "tblite_new_gfn2_calculator");
        api.set_calculator_accuracy(context_, calculator_, settings.accuracy);
        CheckContext("tblite_set_calculator_accuracy");
        // ValidateSettings has checked both keywords' types and bounds.
        const auto max_iterations = settings.keywords.find("max_iterations");
        if (max_iterations != settings.keywords.end()) {
            api.set_calculator_max_iter(context_, calculator_, std::stoi(max_iterations->second));
            CheckContext("tblite_set_calculator_max_iter");
        }
        const auto temperature = settings.keywords.find("electronic_temperature");
        if (temperature != settings.keywords.end()) {
            api.set_calculator_temperature(
                context_, calculator_,
                std::stod(temperature->second) * BOLTZMANN_HARTREE_PER_KELVIN);
            CheckContext("tblite_set_calculator_temperature");
        }
        if (!settings.solvent.IsGasPhase()) {
            AddSolvation(settings.solvent, gfn1);
        }
        result_ = api.new_result();
        if (result_ == nullptr) {
            throw std::runtime_error("tblite could not allocate a result");
        }
    }

    void AddSolvation(const Solvent& solvent, bool gfn1) {
        const detail::TbliteApi& api = library_->api;
        // Calculation resolves AUTO; a direct caller gets the method's AUTO default, ALPB.
        const SolventModel model =
            solvent.GetModel() == SolventModel::AUTO ? SolventModel::ALPB : solvent.GetModel();
        int parameters = 0;
        if (model == SolventModel::ALPB) {
            parameters = gfn1 ? ALPB_GFN1 : ALPB_GFN2;
        } else if (model == SolventModel::GBSA) {
            parameters = gfn1 ? GBSA_GFN1 : GBSA_GFN2;
        } else {
            throw OptionsError("engine 'tblite' does not provide solvent model '" +
                               solvent_model_name(model) + "'");
        }
        std::vector<char> name(solvent.GetName().begin(), solvent.GetName().end());
        name.push_back('\0');
        void* container = api.new_alpb_solvation_solvent(error_, structure_, name.data(),
                                                         parameters, GSOLV_REFERENCE_STATE);
        const std::string rejected = ErrorText();
        if (!rejected.empty()) {
            // tblite's message does not name the solvent, so the result would not say which one.
            throw std::runtime_error("tblite cannot model solvent '" + solvent.GetName() +
                                     "' with " + solvent_model_name(model) + ": " + rejected);
        }
        // The calculator takes ownership and clears the handle; on failure it stays ours.
        api.calculator_push_back(context_, calculator_, &container);
        const std::string failure = ContextText();
        if (container != nullptr) {
            api.delete_container(&container);
        }
        if (!failure.empty()) {
            throw std::runtime_error("tblite_calculator_push_back failed: " + failure);
        }
    }

    /// Fills each requested property of the last single point; the caller holds the mutex.
    void ReadProperties(unsigned properties, Evaluation& evaluation) {
        const detail::TbliteApi& api = library_->api;
        const std::size_t n = num_atoms_;
        if (wants(properties, Property::CHARGES)) {
            evaluation.charges.assign(n, 0.0);
            api.get_result_charges(error_, result_, evaluation.charges.data());
            CheckError("tblite_get_result_charges");
        }
        if (wants(properties, Property::DIPOLE)) {
            evaluation.dipole.assign(3, 0.0);
            api.get_result_dipole(error_, result_, evaluation.dipole.data());
            CheckError("tblite_get_result_dipole");
            for (double& component : evaluation.dipole) {
                component *= Units::EBOHR_TO_DEBYE;
            }
        }
        if (wants(properties, Property::BOND_ORDERS)) {
            // An open-shell result writes a second num_atoms^2 block (measured with tblite
            // 0.6.0: zeros, while the result reports one spin channel), so the buffer holds
            // two and only the first is read.
            std::vector<double> matrix(2 * n * n, 0.0);
            api.get_result_bond_orders(error_, result_, matrix.data());
            CheckError("tblite_get_result_bond_orders");
            matrix.resize(n * n);
            evaluation.bond_orders = detail::wiberg_pairs(matrix, num_atoms_);
        }
        if (wants(properties, Property::ORBITAL_ENERGIES)) {
            int orbitals = 0;
            int spins = 0;
            api.get_result_number_of_orbitals(error_, result_, &orbitals);
            CheckError("tblite_get_result_number_of_orbitals");
            api.get_result_number_of_spins(error_, result_, &spins);
            CheckError("tblite_get_result_number_of_spins");
            if (spins != 1) {
                throw std::runtime_error("tblite reported " + std::to_string(spins) +
                                         " spin channels; only restricted results are read");
            }
            if (orbitals <= 0) {
                throw std::runtime_error("tblite reported no orbitals");
            }
            const auto nao = static_cast<std::size_t>(orbitals);
            // Energies come as one block of nao values; occupations as an alpha block and then a
            // beta block of nao values each, every value in [0, 1] (measured with tblite 0.6.0).
            std::vector<double> energies(nao, 0.0);
            std::vector<double> occupations(2 * nao, 0.0);
            api.get_result_orbital_energies(error_, result_, energies.data());
            CheckError("tblite_get_result_orbital_energies");
            api.get_result_orbital_occupations(error_, result_, occupations.data());
            CheckError("tblite_get_result_orbital_occupations");
            std::vector<double> spin_summed(nao, 0.0);
            evaluation.orbital_energies.assign(nao, 0.0);
            for (std::size_t k = 0; k < nao; ++k) {
                evaluation.orbital_energies[k] = energies[k] * Units::HARTREE_TO_EV;
                spin_summed[k] = occupations[k] + occupations[nao + k];
            }
            evaluation.homo_index = detail::homo_index(spin_summed);
        }
    }

    /// :returns: The context's pending error message, which reading clears, or "" without one.
    std::string ContextText() {
        const detail::TbliteApi& api = library_->api;
        if (api.check_context(context_) == 0) {
            return "";
        }
        std::vector<char> buffer(ERROR_BUFFER_SIZE, '\0');
        const int size = ERROR_BUFFER_SIZE;
        api.get_context_error(context_, buffer.data(), &size);
        buffer.back() = '\0';
        return buffer[0] != '\0' ? std::string(buffer.data()) : std::string("unknown tblite error");
    }

    /// :returns: The error handle's message, clearing it, or "" without one.
    std::string ErrorText() {
        const detail::TbliteApi& api = library_->api;
        if (api.check_error(error_) == 0) {
            return "";
        }
        std::vector<char> buffer(ERROR_BUFFER_SIZE, '\0');
        const int size = ERROR_BUFFER_SIZE;
        api.get_error(error_, buffer.data(), &size);
        // Reading leaves the handle flagged, which would fail every later call.
        api.clear_error(error_);
        buffer.back() = '\0';
        return buffer[0] != '\0' ? std::string(buffer.data()) : std::string("unknown tblite error");
    }

    void CheckContext(const char* call) {
        const std::string text = ContextText();
        if (!text.empty()) {
            throw std::runtime_error(std::string(call) + " failed: " + text);
        }
    }

    void CheckError(const char* call) {
        const std::string text = ErrorText();
        if (!text.empty()) {
            throw std::runtime_error(std::string(call) + " failed: " + text);
        }
    }

    /// Deletes every handle; the caller holds ``tblite_mutex()``.
    void Release() {
        const detail::TbliteApi& api = library_->api;
        if (result_ != nullptr) {
            api.delete_result(&result_);
        }
        if (calculator_ != nullptr) {
            api.delete_calculator(&calculator_);
        }
        if (structure_ != nullptr) {
            api.delete_structure(&structure_);
        }
        if (error_ != nullptr) {
            api.delete_error(&error_);
        }
        if (context_ != nullptr) {
            api.delete_context(&context_);
        }
    }

    std::shared_ptr<const detail::TbliteLibrary> library_;
    unsigned num_atoms_;
    void* context_ = nullptr;
    void* error_ = nullptr;
    void* structure_ = nullptr;
    void* calculator_ = nullptr;
    void* result_ = nullptr;
};

KeywordSpec bounded_keyword(const char* name, KeywordType type, const char* description, double min,
                            double max) {
    KeywordSpec spec;
    spec.name = name;
    spec.type = type;
    spec.description = description;
    spec.has_min = true;
    spec.min = min;
    spec.has_max = true;
    spec.max = max;
    return spec;
}

}  // namespace

/// Outcome of the first load: a usable library, or why there is none.
struct TbliteEngine::Loaded {
    std::shared_ptr<const detail::TbliteLibrary> library;  ///< Null when unavailable.
    std::string unavailable_reason;
};

TbliteEngine::Loaded TbliteEngine::LoadFrom(const std::string& explicit_path) {
    Loaded loaded;
    detail::LibrarySearch search;
    search.engine = "tblite";
    search.explicit_path = explicit_path;
    search.environment_variable = TBLITE_LIBRARY_VARIABLE;
    search.file_names = detail::library_file_names("tblite", "0");
    std::string error;
    std::shared_ptr<detail::RuntimeLibrary> runtime = detail::RuntimeLibrary::Load(search, error);
    if (!runtime) {
        loaded.unavailable_reason = error;
        return loaded;
    }
    auto library = std::make_shared<detail::TbliteLibrary>();
    library->library = runtime;
    const std::string missing = detail::bind_tblite_api(*runtime, library->api);
    const std::string& path = runtime->GetPath();
    if (library->api.get_version == nullptr) {
        loaded.unavailable_reason = path + " is not libtblite: it lacks tblite_get_version";
        return loaded;
    }
    {
        const std::lock_guard<std::mutex> lock(detail::tblite_mutex());
        library->version = library->api.get_version();
    }
    if (!detail::tblite_version_supported(library->version)) {
        loaded.unavailable_reason = "tblite " + detail::tblite_version_text(library->version) +
                                    " at " + path +
                                    " is not supported; OEVibe needs tblite 0.6.x or 0.7.x";
        return loaded;
    }
    if (!missing.empty()) {
        loaded.unavailable_reason = "tblite at " + path + " lacks " + missing;
        return loaded;
    }
    // The loader resolves symbolic links in the image path, so both the path that was opened
    // and the file it resolved to are checked.
    for (const std::string& candidate :
         {path, detail::image_path_of(runtime->Symbol("tblite_get_version"))}) {
        if (const std::optional<std::string> reason = detail::known_bad_tblite_pairing(candidate)) {
            loaded.unavailable_reason = *reason;
            return loaded;
        }
    }
    loaded.library = library;
    return loaded;
}

TbliteEngine::TbliteEngine(std::string library_path) : library_path_(std::move(library_path)) {}

void TbliteEngine::SetLibraryPath(const std::string& path) {
    const std::lock_guard<std::mutex> lock(default_path_mutex());
    default_path() = path;
}

std::string TbliteEngine::GetName() const { return "tblite"; }

std::string TbliteEngine::GetVersion() const {
    const Loaded& loaded = Load();
    return loaded.library ? detail::tblite_version_text(loaded.library->version) : "";
}

bool TbliteEngine::IsAvailable() const { return Load().library != nullptr; }

std::string TbliteEngine::GetUnavailableReason() const { return Load().unavailable_reason; }

std::vector<MethodCapabilities> TbliteEngine::GetMethods() const {
    std::vector<unsigned> elements(86);
    std::iota(elements.begin(), elements.end(), 1u);
    std::vector<MethodCapabilities> methods;
    for (const char* name : {GFN2_XTB, GFN1_XTB}) {
        MethodCapabilities caps;
        caps.method = Method(name);
        caps.solvent_models = {SolventModel::ALPB, SolventModel::GBSA};
        caps.properties = ALL_PROPERTIES;
        caps.analytic_gradient = true;
        caps.open_shell = true;
        caps.elements = elements;
        methods.push_back(caps);
    }
    return methods;
}

std::vector<KeywordSpec> TbliteEngine::GetKeywordSpecs() const {
    return {
        bounded_keyword("accuracy", KeywordType::DOUBLE,
                        "overrides the preset's tblite accuracy; smaller is tighter", 1e-4, 1e3),
        bounded_keyword("max_iterations", KeywordType::INT, "SCF iteration limit", 1, 10000),
        bounded_keyword("electronic_temperature", KeywordType::DOUBLE,
                        "Fermi smearing temperature in K (tblite's default is 300)", 1, 10000)};
}

std::shared_ptr<EngineSession> TbliteEngine::CreateSession(const MolecularSystem& system,
                                                           const EngineSettings& settings) {
    const Loaded& loaded = Load();
    if (!loaded.library) {
        throw EngineUnavailableError("engine 'tblite' is unavailable: " +
                                     loaded.unavailable_reason);
    }
    // Calculation validates at construction; direct callers get the same checks here, so the
    // session can parse the keywords without guarding.
    ValidateSettings(settings);
    return std::make_shared<TbliteSession>(loaded.library, system, settings);
}

const TbliteEngine::Loaded& TbliteEngine::Load() const {
    std::call_once(load_once_, [this]() {
        std::string explicit_path = library_path_;
        if (explicit_path.empty()) {
            const std::lock_guard<std::mutex> lock(default_path_mutex());
            explicit_path = default_path();
        }
        loaded_ = std::make_shared<const Loaded>(LoadFrom(explicit_path));
    });
    return *loaded_;
}

}  // namespace OEVIBE
