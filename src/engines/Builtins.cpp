#include "engines/Builtins.h"

#include <memory>

#include "oevibe/EngineRegistry.h"
#include "oevibe/engines/OEFFEngine.h"

namespace OEVIBE {
namespace detail {

void register_builtin_engines(EngineRegistry& registry) {
    registry.RegisterBuiltin(std::make_shared<OEFFEngine>());
}

}  // namespace detail
}  // namespace OEVIBE
