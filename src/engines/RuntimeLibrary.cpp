#include "engines/RuntimeLibrary.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "oevibe/engines/RuntimeLibraries.h"

namespace OEVIBE {

namespace {

std::mutex& python_directory_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string& python_directory() {
    static std::string directory;
    return directory;
}

std::string environment_value(const std::string& name) {
    const char* value = std::getenv(name.c_str());
    return value != nullptr ? std::string(value) : std::string();
}

std::string join_path(const std::string& directory, const std::string& name) {
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

}  // namespace

void set_python_library_directory(const std::string& directory) {
    const std::lock_guard<std::mutex> lock(python_directory_mutex());
    python_directory() = directory;
}

std::string get_python_library_directory() {
    const std::lock_guard<std::mutex> lock(python_directory_mutex());
    return python_directory();
}

namespace detail {

std::vector<std::string> library_file_names(const std::string& stem, const std::string& major) {
#ifdef __APPLE__
    return {"lib" + stem + "." + major + ".dylib", "lib" + stem + ".dylib"};
#else
    return {"lib" + stem + ".so." + major, "lib" + stem + ".so"};
#endif
}

std::vector<std::string> library_candidates(const LibrarySearch& search) {
    std::vector<std::string> candidates;
    const auto add = [&candidates](const std::string& candidate) {
        if (!candidate.empty() &&
            std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(candidate);
        }
    };
    const auto add_directory = [&](const std::string& directory) {
        if (directory.empty()) {
            return;
        }
        for (const std::string& name : search.file_names) {
            add(join_path(directory, name));
        }
    };
    add(search.explicit_path);
    if (!search.environment_variable.empty()) {
        add(environment_value(search.environment_variable));
    }
    const std::string conda_prefix = environment_value("CONDA_PREFIX");
    if (!conda_prefix.empty()) {
        add_directory(join_path(conda_prefix, "lib"));
    }
    add_directory(get_python_library_directory());
    for (const std::string& name : search.file_names) {
        add(name);
    }
    return candidates;
}

std::string image_path_of(const void* address) {
    Dl_info info{};
    if (address == nullptr || dladdr(address, &info) == 0 || info.dli_fname == nullptr) {
        return "";
    }
    return info.dli_fname;
}

std::shared_ptr<RuntimeLibrary> RuntimeLibrary::Load(const LibrarySearch& search,
                                                     std::string& error) {
    const std::vector<std::string> candidates = library_candidates(search);
    std::string tried;
    for (const std::string& candidate : candidates) {
        // RTLD_NOW reports a missing dependency here rather than at the first call, and
        // RTLD_LOCAL keeps the library's symbols from satisfying anyone else's lookups.
        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            error.clear();
            return std::shared_ptr<RuntimeLibrary>(new RuntimeLibrary(handle, candidate));
        }
        const char* message = dlerror();
        tried += "\n  " + candidate + ": " +
                 (message != nullptr ? std::string(message) : std::string("unknown dlopen error"));
    }
    error =
        "cannot load the " + search.engine + " library; " +
        (candidates.empty() ? std::string("no candidate path was configured") : "tried:" + tried);
    return nullptr;
}

RuntimeLibrary::RuntimeLibrary(void* handle, std::string path)
    : handle_(handle), path_(std::move(path)) {}

// Deliberately never calls dlclose: the library stays loaded for the life of the process.
// Unloading libtblite or libxtb would also unload the gfortran and OpenMP runtimes they pulled
// in, which is unsafe once those runtimes have started threads or registered handlers, and the
// engine registry and sessions released during static destruction may outlive this object.
RuntimeLibrary::~RuntimeLibrary() = default;

void* RuntimeLibrary::Symbol(const char* name) const { return dlsym(handle_, name); }

const std::string& RuntimeLibrary::GetPath() const { return path_; }

}  // namespace detail
}  // namespace OEVIBE
