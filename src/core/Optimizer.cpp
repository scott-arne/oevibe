#include "core/Optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "core/AncRfo.h"
#include "core/Lbfgs.h"
#include "oevibe/Error.h"

namespace OEVIBE {
namespace detail {

namespace {

OptimizationStats stats_of(const MinimizerResult& result, unsigned evaluations,
                           double restraint_energy) {
    OptimizationStats stats;
    stats.iterations = result.iterations;
    stats.gradient_evaluations = evaluations;
    stats.converged = result.converged;
    stats.energy_change = result.energy_change;
    stats.gradient_norm = result.gradient_norm;
    stats.max_gradient_component = result.max_gradient_component;
    stats.max_distance_residual = result.max_distance_residual;
    stats.max_angle_residual = result.max_angle_residual;
    stats.restraint_energy = restraint_energy;
    return stats;
}

}  // namespace

OptimizationFailure::OptimizationFailure(const std::string& message,
                                         std::shared_ptr<const OptimizationOutcome> partial,
                                         std::optional<double> start_energy)
    : std::runtime_error(message), partial_(std::move(partial)), start_energy_(start_energy) {}

bool OptimizationFailure::HasPartialOutcome() const { return partial_ != nullptr; }

const OptimizationOutcome& OptimizationFailure::GetPartialOutcome() const {
    if (!partial_) {
        throw std::logic_error("the optimization failed before any evaluation succeeded");
    }
    return *partial_;
}

bool OptimizationFailure::HasStartEnergy() const { return start_energy_.has_value(); }

double OptimizationFailure::GetStartEnergy() const {
    if (!start_energy_) {
        throw std::logic_error("the optimization failure carries no start energy");
    }
    return *start_energy_;
}

OptimizationOutcome run_optimization(EngineSession& session, const OptimizationProblem& problem) {
    const auto num_atoms = static_cast<unsigned>(problem.atomic_numbers.size());
    // One-point cache: line searches and the final report revisit the last evaluated point.
    std::vector<double> cached_x;
    std::vector<double> cached_gradient;
    double cached_total = 0.0;
    double cached_engine = 0.0;
    double cached_restraint = 0.0;
    unsigned evaluations = 0;
    // The finite energy of a malformed first evaluation, kept as a single point keeps it.
    std::optional<double> start_energy;

    const ObjectiveFn objective = [&](const std::vector<double>& x,
                                      std::vector<double>& gradient) -> double {
        if (!cached_x.empty() && x == cached_x) {
            gradient = cached_gradient;
            return cached_total;
        }
        EvaluationRequest request;
        request.gradient = true;
        const Evaluation evaluation = session.Evaluate(x, request);
        ++evaluations;
        const std::string problem_text = validate_evaluation(evaluation, request, num_atoms);
        if (!problem_text.empty()) {
            if (evaluations == 1 && std::isfinite(evaluation.energy)) {
                start_energy = evaluation.energy;
            }
            throw std::runtime_error("engine returned a malformed Evaluation: " + problem_text);
        }
        gradient = evaluation.gradient;
        const double restraint = restraint_energy(problem.constraints.restraints, x, &gradient);
        cached_x = x;
        cached_gradient = gradient;
        cached_engine = evaluation.energy;
        cached_restraint = restraint;
        cached_total = evaluation.energy + restraint;
        return cached_total;
    };

    MinimizerSettings settings;
    settings.max_iterations = problem.settings.max_iterations;
    settings.energy_change = problem.settings.energy_change;
    settings.gradient_norm = problem.settings.gradient_norm;
    settings.initial_trust_radius = problem.settings.initial_trust_radius;
    settings.max_trust_radius = problem.settings.max_trust_radius;

    // The last iterate the minimizer accepted, with its energies, so that a failure later in
    // the run still reports the furthest geometry reached. The last evaluated point would not
    // do: it may be a trial the minimizer rejected.
    bool have_iterate = false;
    MinimizerResult last_iterate;
    double last_iterate_engine = 0.0;
    double last_iterate_restraint = 0.0;
    settings.observer = [&](const MinimizerResult& iterate) {
        // The minimizers report a point they have just evaluated, so this is a cache hit that
        // leaves the cached energies those of ``iterate.x``.
        std::vector<double> unused;
        objective(iterate.x, unused);
        last_iterate = iterate;
        last_iterate_engine = cached_engine;
        last_iterate_restraint = cached_restraint;
        have_iterate = true;
    };

    MinimizerResult result;
    try {
        if (problem.settings.algorithm == Algorithm::LBFGS) {
            result = lbfgs_minimize(objective, problem.x0, problem.constraints.frozen, settings);
        } else {
            AncRfoProblem anc;
            anc.atomic_numbers = problem.atomic_numbers;
            anc.frozen = problem.constraints.frozen;
            anc.constraints = problem.constraints.constraints;
            // A position restraint pins absolute position and orientation, so projecting out
            // rigid-body motion would drop part of the restraint gradient (spec 7.1).
            const std::vector<ResolvedRestraint>& restraints = problem.constraints.restraints;
            anc.rigid_invariant = std::none_of(restraints.begin(), restraints.end(),
                                               [](const ResolvedRestraint& restraint) {
                                                   return restraint.kind == RestraintKind::POSITION;
                                               });
            result = anc_rfo_minimize(objective, problem.x0, anc, settings);
        }
        // The cache may hold a rejected trial point; re-evaluating the final geometry is a
        // cache hit when it does not, and otherwise costs one evaluation.
        std::vector<double> final_gradient;
        objective(result.x, final_gradient);
    } catch (const UnsupportedError&) {
        throw;
    } catch (const std::exception& e) {
        std::shared_ptr<OptimizationOutcome> partial;
        if (have_iterate) {
            partial = std::make_shared<OptimizationOutcome>();
            partial->message = e.what();
            partial->x = last_iterate.x;
            partial->engine_energy = last_iterate_engine;
            partial->stats = stats_of(last_iterate, evaluations, last_iterate_restraint);
        }
        // An accepted iterate is the furthest point reached, so it supersedes the start energy.
        throw OptimizationFailure(e.what(), std::move(partial),
                                  have_iterate ? std::nullopt : start_energy);
    }

    OptimizationOutcome outcome;
    outcome.converged = result.converged;
    outcome.message = result.message;
    outcome.x = result.x;
    outcome.engine_energy = cached_engine;
    outcome.stats = stats_of(result, evaluations, cached_restraint);
    return outcome;
}

}  // namespace detail
}  // namespace OEVIBE
