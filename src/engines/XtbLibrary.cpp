#include "engines/XtbLibrary.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <utility>

namespace OEVIBE {
namespace detail {

const char* const XTB_GFNFF_UNAVAILABLE_REASON =
    "xtb C API 1.0 writes a gfnff_topo topology file into the process working directory and "
    "reads such a file back from there, and neither the xtb_loadGFNFF filename argument nor "
    "XTBPATH or XTBHOME redirects it (probe 2026-10-11); OEVibe never changes the working "
    "directory, so gfn-ff stays unavailable until isolated worker processes in a later "
    "sub-project restore it";

namespace {

std::string environment_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string join(const std::vector<std::string>& parts) {
    std::string joined;
    for (const std::string& part : parts) {
        joined += joined.empty() ? "" : ", ";
        joined += part;
    }
    return joined;
}

void add_candidate(std::vector<std::string>& candidates, const std::filesystem::path& directory) {
    candidates.push_back((directory / XTB_GFN0_PARAMETER_FILE_NAME).string());
}

template <typename Function>
void resolve(const RuntimeLibrary& runtime, const char* name, Function& target,
             std::vector<std::string>& missing) {
    void* symbol = runtime.Symbol(name);
    if (symbol == nullptr) {
        missing.emplace_back(name);
        return;
    }
    // POSIX requires a dlsym result to convert to the function type the symbol has.
    target = reinterpret_cast<Function>(symbol);
}

}  // namespace

bool xtb_api_version_supported(int version) { return version == XTB_SUPPORTED_API_VERSION; }

std::string xtb_api_version_text(int version) {
    return std::to_string(version / 10000) + "." + std::to_string(version / 100 % 100) + "." +
           std::to_string(version % 100);
}

std::vector<std::string> xtb_gfn0_parameter_candidates(const std::string& library_path,
                                                       const std::string& xtbpath,
                                                       const std::string& xtbhome) {
    std::vector<std::string> candidates;
    std::size_t start = 0;
    while (start <= xtbpath.size()) {
        const std::size_t end = std::min(xtbpath.find(':', start), xtbpath.size());
        if (end > start) {
            add_candidate(candidates, xtbpath.substr(start, end - start));
        }
        start = end + 1;
    }
    if (!xtbhome.empty()) {
        add_candidate(candidates, xtbhome);
    }
    const std::filesystem::path directory = std::filesystem::path(library_path).parent_path();
    if (!directory.empty()) {
        add_candidate(candidates, directory.parent_path() / "share" / "xtb");
    }
    return candidates;
}

std::string xtb_gfn0_parameter_file(const std::string& library_path, const std::string& xtbpath,
                                    const std::string& xtbhome) {
    for (const std::string& candidate :
         xtb_gfn0_parameter_candidates(library_path, xtbpath, xtbhome)) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) {
            return candidate;
        }
    }
    return "";
}

std::string xtb_gfn0_parameter_file(const std::string& library_path) {
    return xtb_gfn0_parameter_file(library_path, environment_value("XTBPATH"),
                                   environment_value("XTBHOME"));
}

std::string xtb_gfn0_missing_reason(const std::string& library_path, const std::string& xtbpath,
                                    const std::string& xtbhome) {
    const std::vector<std::string> candidates =
        xtb_gfn0_parameter_candidates(library_path, xtbpath, xtbhome);
    const std::string advice =
        "; set XTBPATH to the directory that contains " + std::string(XTB_GFN0_PARAMETER_FILE_NAME);
    if (candidates.empty()) {
        return "GFN0-xTB needs " + std::string(XTB_GFN0_PARAMETER_FILE_NAME) +
               ", but XTBPATH and XTBHOME are unset and the libxtb path '" + library_path +
               "' names no directory to search" + advice;
    }
    return "GFN0-xTB needs " + std::string(XTB_GFN0_PARAMETER_FILE_NAME) +
           ", which is not at any of: " + join(candidates) + advice;
}

std::string xtb_gfn0_missing_reason(const std::string& library_path) {
    return xtb_gfn0_missing_reason(library_path, environment_value("XTBPATH"),
                                   environment_value("XTBHOME"));
}

std::mutex& xtb_mutex() {
    // Leaked like the engine registry, so that a session released during static destruction
    // still finds the mutex.
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::shared_ptr<const XtbLibrary> XtbLibrary::Open(std::shared_ptr<RuntimeLibrary> runtime,
                                                   std::string& reason) {
    std::shared_ptr<XtbLibrary> library(new XtbLibrary());
    XtbFunctions& xtb = library->functions_;
    const std::string where = "libxtb at '" + runtime->GetPath() + "'";
    std::vector<std::string> missing;
    resolve(*runtime, "xtb_getAPIVersion", xtb.get_api_version, missing);
    if (!missing.empty()) {
        reason = where + " has no xtb_getAPIVersion, so it is not an xtb C API library";
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(xtb_mutex());
        library->api_version_ = xtb.get_api_version();
    }
    if (!xtb_api_version_supported(library->api_version_)) {
        reason = where + " reports C API " + xtb_api_version_text(library->api_version_) + " (" +
                 std::to_string(library->api_version_) + "); OEVibe supports only C API " +
                 xtb_api_version_text(XTB_SUPPORTED_API_VERSION) + " (xtb 6.7.x)";
        return nullptr;
    }
    resolve(*runtime, "xtb_newEnvironment", xtb.new_environment, missing);
    resolve(*runtime, "xtb_delEnvironment", xtb.del_environment, missing);
    resolve(*runtime, "xtb_checkEnvironment", xtb.check_environment, missing);
    resolve(*runtime, "xtb_getError", xtb.get_error, missing);
    resolve(*runtime, "xtb_setVerbosity", xtb.set_verbosity, missing);
    resolve(*runtime, "xtb_newMolecule", xtb.new_molecule, missing);
    resolve(*runtime, "xtb_delMolecule", xtb.del_molecule, missing);
    resolve(*runtime, "xtb_updateMolecule", xtb.update_molecule, missing);
    resolve(*runtime, "xtb_newCalculator", xtb.new_calculator, missing);
    resolve(*runtime, "xtb_delCalculator", xtb.del_calculator, missing);
    resolve(*runtime, "xtb_loadGFN0xTB", xtb.load_gfn0, missing);
    resolve(*runtime, "xtb_setAccuracy", xtb.set_accuracy, missing);
    resolve(*runtime, "xtb_singlepoint", xtb.singlepoint, missing);
    resolve(*runtime, "xtb_newResults", xtb.new_results, missing);
    resolve(*runtime, "xtb_delResults", xtb.del_results, missing);
    resolve(*runtime, "xtb_getEnergy", xtb.get_energy, missing);
    resolve(*runtime, "xtb_getGradient", xtb.get_gradient, missing);
    resolve(*runtime, "xtb_getDipole", xtb.get_dipole, missing);
    resolve(*runtime, "xtb_getCharges", xtb.get_charges, missing);
    resolve(*runtime, "xtb_getBondOrders", xtb.get_bond_orders, missing);
    resolve(*runtime, "xtb_getNao", xtb.get_nao, missing);
    resolve(*runtime, "xtb_getOrbitalEigenvalues", xtb.get_orbital_eigenvalues, missing);
    resolve(*runtime, "xtb_getOrbitalOccupations", xtb.get_orbital_occupations, missing);
    if (!missing.empty()) {
        reason = where + " lacks C API 1.0 entry points: " + join(missing);
        return nullptr;
    }
    library->runtime_ = std::move(runtime);
    return library;
}

const XtbFunctions& XtbLibrary::Functions() const { return functions_; }
const std::string& XtbLibrary::GetPath() const { return runtime_->GetPath(); }
int XtbLibrary::GetApiVersion() const { return api_version_; }

}  // namespace detail
}  // namespace OEVIBE
