#ifndef OEVIBE_ENGINEREGISTRY_H
#define OEVIBE_ENGINEREGISTRY_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {

/// :returns: Whether the name is reserved for a built-in engine (``oeff``, ``tblite``, ``xtb``).
bool is_builtin_engine_name(const std::string& name);

/// Process-wide catalog of engines (spec 8.2).
class EngineRegistry {
public:
    /// The registry; built-in engines are registered on first use.
    static EngineRegistry& Instance();

    /// :raises OptionsError: For a null engine, an empty or built-in name, or a name already
    ///     registered while ``replace`` is false.
    void Register(std::shared_ptr<Engine> engine, bool replace = false);
    /// :returns: The engine, or null when no engine has the name.
    std::shared_ptr<Engine> Find(const std::string& name) const;
    /// :returns: Every engine in registration order.
    std::vector<std::shared_ptr<Engine>> List() const;
    /// Default engine for a method (spec 5.3).
    /// :raises EngineUnavailableError: When a built-in method's engine is not registered.
    /// :raises OptionsError: When no engine, or more than one, declares a non-built-in method.
    std::shared_ptr<Engine> ResolveForMethod(const Method& method) const;
    /// Registers a built-in engine; only ``detail::register_builtin_engines`` calls this.
    /// :raises OptionsError: For a null engine.
    void RegisterBuiltin(std::shared_ptr<Engine> engine);

private:
    EngineRegistry() = default;

    /// An engine with its name read once at registration. Engine code can be foreign (a Python
    /// engine needs the GIL), so nothing may call a virtual on an engine while ``mutex_`` is
    /// held: a thread that holds the GIL and waits for the mutex would deadlock against it.
    struct Entry {
        std::string name;
        std::shared_ptr<Engine> engine;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> engines_;
};

}  // namespace OEVIBE

#endif  // OEVIBE_ENGINEREGISTRY_H
