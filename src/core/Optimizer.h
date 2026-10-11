#ifndef OEVIBE_CORE_OPTIMIZER_H
#define OEVIBE_CORE_OPTIMIZER_H

#include <memory>
#include <stdexcept>
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

/// A session failure or malformed evaluation during an optimization. ``what()`` is the failure's
/// message; the partial outcome holds the last iterate the minimizer accepted, so a failure
/// late in a run still reports the values computed before it (spec 7.5).
class OptimizationFailure : public std::runtime_error {
public:
    /// :param partial: Null when no evaluation succeeded before the failure.
    OptimizationFailure(const std::string& message,
                        std::shared_ptr<const OptimizationOutcome> partial);

    bool HasPartialOutcome() const;
    /// :raises std::logic_error: When there is no partial outcome.
    const OptimizationOutcome& GetPartialOutcome() const;

private:
    // Shared so that copying the exception cannot throw.
    std::shared_ptr<const OptimizationOutcome> partial_;
};

/// Minimizes engine energy plus restraint energy from ``x0`` (spec 7.1).
/// :raises UnsupportedError: Propagated unchanged from the session.
/// :raises OptimizationFailure: For any other exception from the session, or a malformed
///     evaluation, with a message naming the malformed field.
OptimizationOutcome run_optimization(EngineSession& session, const OptimizationProblem& problem);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_OPTIMIZER_H
