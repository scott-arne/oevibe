#ifndef OEVIBE_ENGINES_BUILTINS_H
#define OEVIBE_ENGINES_BUILTINS_H

namespace OEVIBE {

class EngineRegistry;

namespace detail {

/// Registers every built-in engine into a freshly created registry.
void register_builtin_engines(EngineRegistry& registry);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_BUILTINS_H
