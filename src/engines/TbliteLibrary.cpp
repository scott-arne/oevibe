#include "engines/TbliteLibrary.h"

#include <filesystem>
#include <system_error>

namespace OEVIBE {
namespace detail {

namespace {

/// One conda-meta record, from a file name of the form <name>-<version>-<build>.json.
struct CondaRecord {
    std::string name;
    std::string version;
    std::string build_number;  ///< The digits after the build string's last underscore.
};

std::optional<CondaRecord> parse_conda_record(const std::string& file_name) {
    const std::string suffix = ".json";
    if (file_name.size() <= suffix.size() ||
        file_name.compare(file_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return std::nullopt;
    }
    const std::string stem = file_name.substr(0, file_name.size() - suffix.size());
    const std::size_t build_dash = stem.rfind('-');
    if (build_dash == std::string::npos || build_dash == 0) {
        return std::nullopt;
    }
    const std::size_t version_dash = stem.rfind('-', build_dash - 1);
    if (version_dash == std::string::npos || version_dash == 0) {
        return std::nullopt;
    }
    CondaRecord record;
    record.name = stem.substr(0, version_dash);
    record.version = stem.substr(version_dash + 1, build_dash - version_dash - 1);
    const std::string build = stem.substr(build_dash + 1);
    const std::size_t underscore = build.rfind('_');
    record.build_number = underscore == std::string::npos ? build : build.substr(underscore + 1);
    return record;
}

template <typename Function>
void bind(const RuntimeLibrary& library, const char* name, Function& slot, std::string& missing) {
    slot = reinterpret_cast<Function>(library.Symbol(name));
    if (slot == nullptr) {
        missing += (missing.empty() ? "" : ", ") + std::string(name);
    }
}

}  // namespace

bool tblite_version_supported(int version) {
    return (version >= 600 && version <= 699) || (version >= 700 && version <= 799);
}

std::string tblite_version_text(int version) {
    return std::to_string(version / 10000) + "." + std::to_string(version / 100 % 100) + "." +
           std::to_string(version % 100);
}

std::optional<std::string> known_bad_tblite_pairing(const std::string& library_path) {
    namespace fs = std::filesystem;
    const fs::path prefix = fs::path(library_path).parent_path().parent_path();
    if (prefix.empty()) {
        return std::nullopt;
    }
    const fs::path records = prefix / "conda-meta";
    std::error_code error;
    if (!fs::is_directory(records, error)) {
        return std::nullopt;
    }
    std::string tblite_file;
    std::string dftd4_file;
    for (fs::directory_iterator entry(records, error), end; !error && entry != end;
         entry.increment(error)) {
        const std::string file_name = entry->path().filename().string();
        const std::optional<CondaRecord> record = parse_conda_record(file_name);
        if (!record) {
            continue;
        }
        if (record->name == "tblite" && record->version.rfind("0.6.", 0) == 0) {
            tblite_file = file_name;
        } else if (record->name == "dftd4" && record->version == "4.2.0" &&
                   record->build_number == "1") {
            dftd4_file = file_name;
        }
    }
    if (tblite_file.empty() || dftd4_file.empty()) {
        return std::nullopt;
    }
    return "the conda environment " + prefix.string() + " pairs " + tblite_file + " with " +
           dftd4_file +
           ", and with that pairing GFN2-xTB single points never return; install dftd4 4.2.0 "
           "build 0 (\"dftd4=4.2.0=*_0\") in that environment";
}

std::mutex& tblite_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string bind_tblite_api(const RuntimeLibrary& library, TbliteApi& api) {
    std::string missing;
    bind(library, "tblite_get_version", api.get_version, missing);
    bind(library, "tblite_new_context", api.new_context, missing);
    bind(library, "tblite_delete_context", api.delete_context, missing);
    bind(library, "tblite_check_context", api.check_context, missing);
    bind(library, "tblite_get_context_error", api.get_context_error, missing);
    bind(library, "tblite_set_context_verbosity", api.set_context_verbosity, missing);
    bind(library, "tblite_new_error", api.new_error, missing);
    bind(library, "tblite_delete_error", api.delete_error, missing);
    bind(library, "tblite_check_error", api.check_error, missing);
    bind(library, "tblite_clear_error", api.clear_error, missing);
    bind(library, "tblite_get_error", api.get_error, missing);
    bind(library, "tblite_new_structure", api.new_structure, missing);
    bind(library, "tblite_delete_structure", api.delete_structure, missing);
    bind(library, "tblite_update_structure_geometry", api.update_structure_geometry, missing);
    bind(library, "tblite_new_gfn2_calculator", api.new_gfn2_calculator, missing);
    bind(library, "tblite_new_gfn1_calculator", api.new_gfn1_calculator, missing);
    bind(library, "tblite_delete_calculator", api.delete_calculator, missing);
    bind(library, "tblite_set_calculator_accuracy", api.set_calculator_accuracy, missing);
    bind(library, "tblite_set_calculator_max_iter", api.set_calculator_max_iter, missing);
    bind(library, "tblite_set_calculator_temperature", api.set_calculator_temperature, missing);
    bind(library, "tblite_new_alpb_solvation_solvent", api.new_alpb_solvation_solvent, missing);
    bind(library, "tblite_calculator_push_back", api.calculator_push_back, missing);
    bind(library, "tblite_delete_container", api.delete_container, missing);
    bind(library, "tblite_new_result", api.new_result, missing);
    bind(library, "tblite_delete_result", api.delete_result, missing);
    bind(library, "tblite_get_singlepoint", api.get_singlepoint, missing);
    bind(library, "tblite_get_result_energy", api.get_result_energy, missing);
    bind(library, "tblite_get_result_gradient", api.get_result_gradient, missing);
    return missing;
}

}  // namespace detail
}  // namespace OEVIBE
