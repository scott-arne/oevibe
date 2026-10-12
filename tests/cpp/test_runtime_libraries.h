#ifndef OEVIBE_TEST_RUNTIME_LIBRARIES_H
#define OEVIBE_TEST_RUNTIME_LIBRARIES_H

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "oevibe/engines/RuntimeLibraries.h"

namespace OEVIBE {
namespace test {

/// Why engine-library tests skip (spec 12.2).
constexpr const char* TBLITE_SKIP_REASON =
    "OEVIBE_TBLITE_LIBRARY is unset or does not name an existing file";
constexpr const char* XTB_SKIP_REASON =
    "OEVIBE_XTB_LIBRARY is unset or does not name an existing file";

/// :returns: The value of ``variable`` when it names an existing file, else an empty string.
inline std::string library_from_environment(const char* variable) {
    const char* value = std::getenv(variable);
    if (value == nullptr || *value == '\0') {
        return "";
    }
    std::error_code error;
    return std::filesystem::is_regular_file(value, error) ? std::string(value) : std::string();
}

/// :returns: ``OEVIBE_TBLITE_LIBRARY`` when it names an existing file, else an empty string.
inline std::string tblite_library_path() {
    return library_from_environment("OEVIBE_TBLITE_LIBRARY");
}

/// :returns: ``OEVIBE_XTB_LIBRARY`` when it names an existing file, else an empty string.
inline std::string xtb_library_path() { return library_from_environment("OEVIBE_XTB_LIBRARY"); }

/// Sets or unsets one environment variable for a scope and restores its previous state.
class ScopedEnvironment {
public:
    /// :param value: The new value, or null to unset the variable.
    ScopedEnvironment(std::string name, const char* value) : name_(std::move(name)) {
        if (const char* previous = std::getenv(name_.c_str())) {
            previous_ = std::string(previous);
        }
        if (value != nullptr) {
            ::setenv(name_.c_str(), value, 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }
    ~ScopedEnvironment() {
        if (previous_) {
            ::setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    std::string name_;
    std::optional<std::string> previous_;
};

/// Sets the process-wide Python library directory for a scope and restores it afterwards.
class ScopedPythonLibraryDirectory {
public:
    explicit ScopedPythonLibraryDirectory(const std::string& directory)
        : previous_(get_python_library_directory()) {
        set_python_library_directory(directory);
    }
    ~ScopedPythonLibraryDirectory() { set_python_library_directory(previous_); }
    ScopedPythonLibraryDirectory(const ScopedPythonLibraryDirectory&) = delete;
    ScopedPythonLibraryDirectory& operator=(const ScopedPythonLibraryDirectory&) = delete;

private:
    std::string previous_;
};

}  // namespace test
}  // namespace OEVIBE

#endif  // OEVIBE_TEST_RUNTIME_LIBRARIES_H
