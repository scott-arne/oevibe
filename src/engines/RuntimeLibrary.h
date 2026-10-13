#ifndef OEVIBE_ENGINES_RUNTIMELIBRARY_H
#define OEVIBE_ENGINES_RUNTIMELIBRARY_H

#include <memory>
#include <string>
#include <vector>

namespace OEVIBE {
namespace detail {

/// Where to look for one runtime-loaded engine library (spec 8.3).
struct LibrarySearch {
    std::string engine;                   ///< ``tblite`` or ``xtb``, for messages.
    std::string explicit_path;            ///< Tried first: the constructor or ``SetLibraryPath``.
    std::string environment_variable;     ///< ``OEVIBE_TBLITE_LIBRARY`` or ``OEVIBE_XTB_LIBRARY``.
    std::vector<std::string> file_names;  ///< Platform file names, most specific first.
};

/// :returns: The platform file names of ``lib<stem>`` with ABI major version ``major``, most
///     specific first: ``lib<stem>.<major>.dylib`` and ``lib<stem>.dylib`` on macOS,
///     ``lib<stem>.so.<major>`` and ``lib<stem>.so`` elsewhere.
std::vector<std::string> library_file_names(const std::string& stem, const std::string& major);

/// :returns: Every candidate in spec 8.3 order: explicit path, environment variable,
///     $CONDA_PREFIX/lib/<name>, <python library directory>/<name>, then bare <name> for the
///     system loader. Unset or empty sources are skipped and repeated candidates dropped.
std::vector<std::string> library_candidates(const LibrarySearch& search);

/// :returns: The file of the loaded image that contains ``address``, as the loader reports it
///     (with symbolic links resolved), or an empty string when the loader cannot tell.
std::string image_path_of(const void* address);

/// One shared library opened with ``dlopen``. The handle is never closed, so the library stays
/// loaded for the life of the process (see the destructor).
class RuntimeLibrary {
public:
    /// :returns: The first candidate that loads (RTLD_NOW | RTLD_LOCAL), or null with
    ///     ``error`` listing every candidate tried and its dlopen message.
    static std::shared_ptr<RuntimeLibrary> Load(const LibrarySearch& search, std::string& error);
    ~RuntimeLibrary();
    RuntimeLibrary(const RuntimeLibrary&) = delete;
    RuntimeLibrary& operator=(const RuntimeLibrary&) = delete;

    /// :returns: The symbol, or null when it is missing.
    void* Symbol(const char* name) const;
    /// :returns: The candidate that loaded, as it was passed to ``dlopen``.
    const std::string& GetPath() const;

private:
    RuntimeLibrary(void* handle, std::string path);

    void* handle_;
    std::string path_;
};

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_RUNTIMELIBRARY_H
