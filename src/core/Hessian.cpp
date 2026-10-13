#include "core/Hessian.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {
namespace detail {

namespace {

unsigned atoms_of(const std::vector<double>& x) {
    if (x.empty() || x.size() % 3 != 0) {
        throw OEVibeError("a Hessian needs 3N coordinates, got " + std::to_string(x.size()));
    }
    return static_cast<unsigned>(x.size() / 3);
}

Evaluation checked_evaluate(EngineSession& session, const std::vector<double>& x,
                            const EvaluationRequest& request, unsigned num_atoms) {
    Evaluation evaluation = session.Evaluate(x, request);
    const std::string problem = validate_evaluation(evaluation, request, num_atoms);
    if (!problem.empty()) {
        throw std::runtime_error("engine returned a malformed Evaluation: " + problem);
    }
    return evaluation;
}

/// Replaces each off-diagonal pair by its mean. Writing one computed mean to both entries makes
/// the matrix exactly symmetric, which the frequency analysis's eigensolver assumes.
void symmetrize(std::vector<double>& matrix, std::size_t dim) {
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            const double mean = 0.5 * (matrix[i * dim + j] + matrix[j * dim + i]);
            matrix[i * dim + j] = mean;
            matrix[j * dim + i] = mean;
        }
    }
}

}  // namespace

std::vector<double> finite_difference_hessian(EngineSession& session, const std::vector<double>& x,
                                              double step) {
    const unsigned num_atoms = atoms_of(x);
    if (!std::isfinite(step) || step <= 0.0) {
        throw OEVibeError(
            "the finite-difference step must be finite and greater than 0 Angstrom, got " +
            std::to_string(step));
    }
    const std::size_t dim = x.size();
    EvaluationRequest request;
    request.gradient = true;
    std::vector<double> hessian(dim * dim);
    std::vector<double> displaced = x;
    for (std::size_t k = 0; k < dim; ++k) {
        displaced[k] = x[k] + step;
        const double upper = displaced[k];
        const Evaluation plus = checked_evaluate(session, displaced, request, num_atoms);
        displaced[k] = x[k] - step;
        const double lower = displaced[k];
        const Evaluation minus = checked_evaluate(session, displaced, request, num_atoms);
        displaced[k] = x[k];
        // Rounding makes the displacement actually taken differ slightly from 2 * step.
        const double width = upper - lower;
        for (std::size_t j = 0; j < dim; ++j) {
            hessian[k * dim + j] = (plus.gradient[j] - minus.gradient[j]) / width;
        }
    }
    symmetrize(hessian, dim);
    return hessian;
}

std::vector<double> compute_hessian(EngineSession& session, const std::vector<double>& x,
                                    bool analytic, double step) {
    if (!analytic) {
        return finite_difference_hessian(session, x, step);
    }
    const unsigned num_atoms = atoms_of(x);
    EvaluationRequest request;
    request.hessian = true;
    std::vector<double> hessian = checked_evaluate(session, x, request, num_atoms).hessian;
    symmetrize(hessian, x.size());
    return hessian;
}

}  // namespace detail
}  // namespace OEVIBE
