#ifndef OEVIBE_ENGINES_XTBENGINE_H
#define OEVIBE_ENGINES_XTBENGINE_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {

namespace detail {
struct XtbEngineState;
}  // namespace detail

/// GFN0-xTB through libxtb's C API 1.0, loaded at runtime with ``dlopen`` (spec 8.3, 8.5).
///
/// ``gfn-ff`` is declared but unavailable: C API 1.0 keeps GFN-FF topology files in the process
/// working directory, which OEVibe never changes. The library loads on first use rather than
/// at construction, so the Python package can pass its library directory before discovery runs.
/// One process-wide mutex serializes every libxtb call, because the C API is not thread-safe.
class XtbEngine : public Engine {
public:
    /// :param library_path: Overrides discovery for this instance; empty uses SetLibraryPath's
    ///     value and then spec 8.3 discovery. The library loads lazily on first use.
    explicit XtbEngine(std::string library_path = "");
    /// Default explicit path for instances built without one (the registry's built-in). An
    /// instance reads it when it loads its library, so it must be set before first use.
    static void SetLibraryPath(const std::string& path);
    /// :returns: ``xtb``; never loads the library.
    std::string GetName() const override;
    /// :returns: ``C API 1.0.0``, since C API 1.0 reports no xtb release, or "" when the
    ///     engine is unavailable.
    std::string GetVersion() const override;
    bool IsAvailable() const override;
    /// :returns: Every library candidate tried and why it failed, or the rejected C API version
    ///     or missing entry points; "" when available.
    std::string GetUnavailableReason() const override;
    /// :returns: ``gfn0-xtb``, available when its parameter file is found, and ``gfn-ff``,
    ///     never available.
    std::vector<MethodCapabilities> GetMethods() const override;
    /// :returns: ``accuracy`` (DOUBLE, 1e-4 to 1e3) for both methods.
    std::vector<KeywordSpec> GetKeywordSpecs() const override;
    /// :raises EngineUnavailableError: When the library or the method is unavailable.
    /// :raises OptionsError: For a method the engine does not provide, or a solvent.
    /// :raises std::runtime_error: With libxtb's message when it rejects the system.
    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override;

private:
    /// :returns: The load outcome, computed on the first call and kept for the engine's life.
    const detail::XtbEngineState& Load() const;

    std::string library_path_;
    mutable std::mutex load_mutex_;
    mutable std::shared_ptr<const detail::XtbEngineState> state_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_XTBENGINE_H
