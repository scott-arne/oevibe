#include "engines/Builtins.h"

#include "oevibe/EngineRegistry.h"

namespace OEVIBE {
namespace detail {

void register_builtin_engines(EngineRegistry& registry) { static_cast<void>(registry); }

}  // namespace detail
}  // namespace OEVIBE
