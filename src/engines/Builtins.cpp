#include "engines/Builtins.h"

#include <memory>
#include <utility>

#include "oevibe/EngineRegistry.h"
#include "oevibe/engines/OEFFEngine.h"

namespace OEVIBE {
namespace detail {

void BuiltinRegistration::Register(EngineRegistry& registry, std::shared_ptr<Engine> engine) {
    registry.RegisterBuiltin(std::move(engine));
}

void register_builtin_engines(EngineRegistry& registry) {
    BuiltinRegistration::Register(registry, std::make_shared<OEFFEngine>());
}

}  // namespace detail
}  // namespace OEVIBE
