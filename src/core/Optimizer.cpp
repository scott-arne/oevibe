#include "core/Optimizer.h"

#include <stdexcept>

#include "core/AncRfo.h"
#include "core/Lbfgs.h"

namespace OEVIBE {
namespace detail {

OptimizationOutcome run_optimization(EngineSession& session, const OptimizationProblem& problem) {
    const auto num_atoms = static_cast<unsigned>(problem.atomic_numbers.size());
    // One-point cache: line searches and the final report revisit the last evaluated point.
    std::vector<double> cached_x;
    std::vector<double> cached_gradient;
    double cached_total = 0.0;
    double cached_engine = 0.0;
    double cached_restraint = 0.0;
    unsigned evaluations = 0;

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

    MinimizerResult result;
    if (problem.settings.algorithm == Algorithm::LBFGS) {
        result = lbfgs_minimize(objective, problem.x0, problem.constraints.frozen, settings);
    } else {
        AncRfoProblem anc;
        anc.atomic_numbers = problem.atomic_numbers;
        anc.frozen = problem.constraints.frozen;
        anc.constraints = problem.constraints.constraints;
        result = anc_rfo_minimize(objective, problem.x0, anc, settings);
    }

    // The cache may hold a rejected trial point; re-evaluating the final geometry is a cache
    // hit when it does not, and otherwise costs one evaluation.
    std::vector<double> final_gradient;
    objective(result.x, final_gradient);

    OptimizationOutcome outcome;
    outcome.converged = result.converged;
    outcome.message = result.message;
    outcome.x = result.x;
    outcome.engine_energy = cached_engine;
    outcome.stats.iterations = result.iterations;
    outcome.stats.gradient_evaluations = evaluations;
    outcome.stats.converged = result.converged;
    outcome.stats.energy_change = result.energy_change;
    outcome.stats.gradient_norm = result.gradient_norm;
    outcome.stats.max_gradient_component = result.max_gradient_component;
    outcome.stats.max_distance_residual = result.max_distance_residual;
    outcome.stats.max_angle_residual = result.max_angle_residual;
    outcome.stats.restraint_energy = cached_restraint;
    return outcome;
}

}  // namespace detail
}  // namespace OEVIBE
