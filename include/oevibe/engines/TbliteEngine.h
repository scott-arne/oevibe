#ifndef OEVIBE_ENGINES_TBLITEENGINE_H
#define OEVIBE_ENGINES_TBLITEENGINE_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {

/// GFN2-xTB and GFN1-xTB through libtblite, which is loaded at runtime and never linked
/// (spec 8.3, 8.4). The library loads on the first call that needs it, so the Python package
/// can pass its prefix first; that first load decides availability for the instance's lifetime.
/// Every library call in the process is serialized by one mutex.
class TbliteEngine : public Engine {
public:
    /// :param library_path: Overrides discovery for this instance; empty uses SetLibraryPath's
    ///     value and then spec 8.3 discovery. The library loads lazily on first use.
    explicit TbliteEngine(std::string library_path = "");

    /// Default explicit path for instances built without one (the registry's built-in). It is
    /// read when an instance first loads, so it must be set before the engine is first queried.
    static void SetLibraryPath(const std::string& path);

    /// :returns: ``tblite``; never loads the library.
    std::string GetName() const override;
    /// :returns: The library version, for example ``0.6.0``, or an empty string when unavailable.
    std::string GetVersion() const override;
    bool IsAvailable() const override;
    /// :returns: Why the library cannot be used (every path tried, an unsupported version, or a
    ///     known-bad conda pairing), or an empty string when available.
    std::string GetUnavailableReason() const override;
    /// :returns: ``gfn2-xtb`` and ``gfn1-xtb``; never loads the library.
    std::vector<MethodCapabilities> GetMethods() const override;
    /// :returns: ``accuracy``, ``max_iterations``, and ``electronic_temperature``.
    std::vector<KeywordSpec> GetKeywordSpecs() const override;
    /// The session calls tblite only from its first ``Evaluate``, which builds the structure from
    /// the coordinates it is given, so a system's conformer order never decides a result (spec
    /// 6.4). That evaluation throws ``std::runtime_error`` with tblite's own message when the
    /// library rejects the structure or the solvent.
    /// :raises EngineUnavailableError: When the library cannot be used.
    /// :raises OptionsError: For settings that fail ``ValidateSettings`` or a solvent model
    ///     tblite does not provide.
    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override;

private:
    struct Loaded;

    /// :returns: The outcome of the first load; later calls return the same outcome.
    const Loaded& Load() const;
    /// Opens the first usable candidate, binds the C API, and checks the version and the
    /// conda pairing (spec 8.3, 8.4).
    static Loaded LoadFrom(const std::string& explicit_path);

    std::string library_path_;
    mutable std::once_flag load_once_;
    mutable std::shared_ptr<const Loaded> loaded_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_TBLITEENGINE_H
