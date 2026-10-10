#include "oevibe/EngineRegistry.h"

#include <algorithm>

#include "engines/Builtins.h"
#include "oevibe/Error.h"

namespace OEVIBE {

bool is_builtin_engine_name(const std::string& name) {
    return name == "oeff" || name == "tblite" || name == "xtb";
}

EngineRegistry& EngineRegistry::Instance() {
    // Deliberately leaked: Python engines registered in Plan C must outlive every static
    // destructor that might still reach the registry during interpreter shutdown.
    static EngineRegistry* registry = [] {
        auto* created = new EngineRegistry();
        detail::register_builtin_engines(*created);
        return created;
    }();
    return *registry;
}

void EngineRegistry::Register(std::shared_ptr<Engine> engine, bool replace) {
    if (!engine) {
        throw OptionsError("cannot register a null engine");
    }
    const std::string name = engine->GetName();
    if (name.empty()) {
        throw OptionsError("engine names must not be empty");
    }
    if (is_builtin_engine_name(name)) {
        throw OptionsError("engine name '" + name + "' is reserved for a built-in engine");
    }
    // Both owners are declared before the lock so that an engine this call releases is destroyed
    // after the lock is released, on every path including an exception from ``push_back``; its
    // destructor can be foreign code (see ``EngineRegistry::Entry``).
    Entry incoming{name, std::move(engine)};
    std::shared_ptr<Engine> replaced;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find_if(engines_.begin(), engines_.end(),
                                 [&](const Entry& entry) { return entry.name == name; });
    if (it == engines_.end()) {
        engines_.push_back(std::move(incoming));
        return;
    }
    if (!replace) {
        throw OptionsError("an engine named '" + name + "' is already registered");
    }
    replaced = std::move(it->engine);
    it->engine = std::move(incoming.engine);
}

void EngineRegistry::RegisterBuiltin(std::shared_ptr<Engine> engine) {
    if (!engine) {
        throw OptionsError("cannot register a null engine");
    }
    std::string name = engine->GetName();
    // Built before the lock for the same reason as in ``Register``.
    Entry incoming{std::move(name), std::move(engine)};
    std::lock_guard<std::mutex> lock(mutex_);
    engines_.push_back(std::move(incoming));
}

std::shared_ptr<Engine> EngineRegistry::Find(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find_if(engines_.begin(), engines_.end(),
                                 [&](const Entry& entry) { return entry.name == name; });
    return it == engines_.end() ? nullptr : it->engine;
}

std::vector<std::shared_ptr<Engine>> EngineRegistry::List() const {
    std::vector<std::shared_ptr<Engine>> engines;
    std::lock_guard<std::mutex> lock(mutex_);
    engines.reserve(engines_.size());
    for (const Entry& entry : engines_) {
        engines.push_back(entry.engine);
    }
    return engines;
}

std::shared_ptr<Engine> EngineRegistry::ResolveForMethod(const Method& method) const {
    const std::string builtin = builtin_engine_for(method);
    if (!builtin.empty()) {
        std::shared_ptr<Engine> engine = Find(builtin);
        if (!engine) {
            throw EngineUnavailableError("method '" + method.GetName() + "' needs the '" + builtin +
                                         "' engine, which is not available");
        }
        return engine;
    }
    std::vector<std::shared_ptr<Engine>> matches;
    for (const auto& engine : List()) {
        if (find_method(*engine, method).has_value()) {
            matches.push_back(engine);
        }
    }
    if (matches.empty()) {
        throw OptionsError("no registered engine provides method '" + method.GetName() + "'");
    }
    if (matches.size() > 1) {
        throw OptionsError("more than one engine provides method '" + method.GetName() +
                           "'; choose one with SetEngine");
    }
    return matches.front();
}

}  // namespace OEVIBE
