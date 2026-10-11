#ifndef OEVIBE_ENGINES_BUILTINS_H
#define OEVIBE_ENGINES_BUILTINS_H

#include <memory>

namespace OEVIBE {

class Engine;
class EngineRegistry;

namespace detail {

/// The library's only route to ``EngineRegistry``'s built-in registration, which bypasses the
/// reserved-name and duplicate checks of ``Register``. It lives outside the public include tree
/// so that callers of the installed headers cannot claim a built-in engine name.
class BuiltinRegistration {
public:
    /// :raises OptionsError: For a null engine.
    static void Register(EngineRegistry& registry, std::shared_ptr<Engine> engine);
};

/// Registers every built-in engine into a freshly created registry.
void register_builtin_engines(EngineRegistry& registry);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_BUILTINS_H
