#ifndef OEVIBE_CORE_OPTIMIZER_H
#define OEVIBE_CORE_OPTIMIZER_H

#include <string>
#include <vector>

#include "core/ConstraintMath.h"
#include "oevibe/Engine.h"
#include "oevibe/Options.h"
#include "oevibe/Result.h"

namespace OEVIBE {
namespace detail {

struct OptimizationProblem {
    std::vector<double> x0;                ///< Dense 3N, Angstrom.
    std::vector<unsigned> atomic_numbers;  ///< Dense order.
    ResolvedConstraintSet constraints;
    ResolvedOptimizerSettings settings;  ///< ``algorithm`` is ANC_RFO or LBFGS.
};

struct OptimizationOutcome {
    bool converged = false;
    std::string message;
    std::vector<double> x;       ///< Dense 3N, Angstrom.
    double engine_energy = 0.0;  ///< kcal/mol at ``x``, without restraint energy.
    OptimizationStats stats;
};

/// Minimizes engine energy plus restraint energy from ``x0`` (spec 7.1). Exceptions from the
/// session propagate; a malformed evaluation raises ``std::runtime_error`` naming the field.
OptimizationOutcome run_optimization(EngineSession& session, const OptimizationProblem& problem);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_OPTIMIZER_H
