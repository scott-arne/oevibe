#include "engines/Builtins.h"

#include <memory>
#include <utility>

#include "oevibe/EngineRegistry.h"
#include "oevibe/engines/OEFFEngine.h"
#include "oevibe/engines/TbliteEngine.h"
#include "oevibe/engines/XtbEngine.h"

namespace OEVIBE {
namespace detail {

void BuiltinRegistration::Register(EngineRegistry& registry, std::shared_ptr<Engine> engine) {
    registry.RegisterBuiltin(std::move(engine));
}

void register_builtin_engines(EngineRegistry& registry) {
    BuiltinRegistration::Register(registry, std::make_shared<OEFFEngine>());
    // Both load their library on first use, never here, so the registry can be created before
    // the Python package passes its library directory, and a missing library only makes its
    // engine unavailable.
    BuiltinRegistration::Register(registry, std::make_shared<TbliteEngine>());
    BuiltinRegistration::Register(registry, std::make_shared<XtbEngine>());
}

}  // namespace detail
}  // namespace OEVIBE
