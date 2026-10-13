#ifndef OEVIBE_CORE_HESSIAN_H
#define OEVIBE_CORE_HESSIAN_H

#include <vector>

#include "oevibe/Engine.h"

namespace OEVIBE {
namespace detail {

/// Central differences of analytic gradients: 6N evaluations at x +/- step along each
/// Cartesian component, symmetrized (spec 7.2). Returns 9N^2 values, row-major,
/// kcal/mol/Angstrom^2. Exceptions from the session propagate unchanged.
/// :param x: Dense 3N coordinates in Angstrom.
/// :param step: Displacement in Angstrom; ``Calculation`` passes the preset's ``fd_step``.
/// :raises OEVibeError: When ``x`` is empty or not a multiple of 3 long, or ``step`` is not
///     finite and greater than 0.
/// :raises std::runtime_error: For a malformed evaluation (message names the field).
std::vector<double> finite_difference_hessian(EngineSession& session, const std::vector<double>& x,
                                              double step);

/// The engine's analytic Hessian when ``analytic`` is true, otherwise finite differences. Both
/// are symmetrized, so every consumer receives an exactly symmetric matrix.
/// :raises OEVibeError: As ``finite_difference_hessian``; ``step`` is checked only when used.
/// :raises std::runtime_error: For a malformed evaluation (message names the field).
std::vector<double> compute_hessian(EngineSession& session, const std::vector<double>& x,
                                    bool analytic, double step);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_HESSIAN_H
