#ifndef OEVIBE_CORE_LBFGS_H
#define OEVIBE_CORE_LBFGS_H

#include <vector>

#include "core/Minimizer.h"

namespace OEVIBE {
namespace detail {

/// Limited-memory BFGS (Nocedal, Math. Comp. 35, 773, 1980) with an Armijo backtracking line
/// search and the largest atomic displacement capped at the maximum trust radius. Frozen atoms
/// never move; exact constraints are not supported (spec 7.1).
MinimizerResult lbfgs_minimize(const ObjectiveFn& objective, std::vector<double> x0,
                               const std::vector<unsigned>& frozen,
                               const MinimizerSettings& settings);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_LBFGS_H
