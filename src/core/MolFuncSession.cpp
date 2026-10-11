#include "core/MolFuncSession.h"

#include <stdexcept>
#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {
namespace detail {

MolFuncSession::MolFuncSession(std::unique_ptr<OEMolPotential::OEMolFunc1> function,
                               const MolecularSystem& system)
    : molecule_(system.GetMolecule()),
      function_(std::move(function)),
      num_atoms_(system.NumAtoms()) {
    const std::string title = molecule_.GetTitle();
    if (auto* force_field = dynamic_cast<OEMolPotential::OEForceField*>(function_.get())) {
        // sweep=false: a Sweep may renumber atoms, which would break the dense index mapping.
        if (!force_field->PrepMol(molecule_, false, true)) {
            throw UnsupportedError("OpenEye force-field preparation failed for '" + title + "'");
        }
        if (molecule_.NumAtoms() != num_atoms_ || molecule_.GetMaxAtomIdx() != num_atoms_) {
            throw UnsupportedError("OpenEye force-field preparation changed the atoms of '" +
                                   title + "'");
        }
    }
    if (!function_->Setup(molecule_)) {
        throw UnsupportedError("OpenEye potential setup failed for '" + title +
                               "'; parameters may be missing");
    }
    if (function_->NumVar() != 3 * num_atoms_) {
        throw UnsupportedError("OpenEye potential expects " + std::to_string(function_->NumVar()) +
                               " variables, not " + std::to_string(3 * num_atoms_));
    }
}

Evaluation MolFuncSession::Evaluate(const std::vector<double>& coords_angstrom,
                                    const EvaluationRequest& request) {
    const std::size_t nvar = 3 * static_cast<std::size_t>(num_atoms_);
    if (coords_angstrom.size() != nvar) {
        throw std::invalid_argument("expected " + std::to_string(nvar) + " coordinates, got " +
                                    std::to_string(coords_angstrom.size()));
    }
    if (request.properties != 0) {
        throw std::runtime_error("OpenEye potentials provide no properties");
    }
    Evaluation evaluation;
    // OpenEye accumulates into gradient and Hessian arrays, so every output starts at zero.
    if (request.hessian) {
        auto* second = dynamic_cast<OEMolPotential::OEMolFunc2*>(function_.get());
        if (second == nullptr) {
            throw std::runtime_error("this OpenEye potential has no analytic Hessian");
        }
        std::vector<double> gradient(nvar, 0.0);
        // OEMolFunc2 writes the lower triangle packed row by row, element (row, col <= row) at
        // row * (row + 1) / 2 + col, not a full matrix (measured: this reading matches finite
        // differences of the gradient, a full row-major reading does not). The engine contract
        // needs the full symmetric matrix, so it is expanded below. The header documents no
        // buffer size and only the first nvar * (nvar + 1) / 2 entries were observed written;
        // the full nvar * nvar allocation is kept so an undocumented larger write stays in bounds.
        std::vector<double> packed(nvar * nvar, 0.0);
        if (!(*second)(coords_angstrom.data(), packed.data(), gradient.data())) {
            throw std::runtime_error("OpenEye Hessian evaluation failed");
        }
        evaluation.hessian.assign(nvar * nvar, 0.0);
        for (std::size_t row = 0; row < nvar; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                const double value = packed[row * (row + 1) / 2 + col];
                evaluation.hessian[row * nvar + col] = value;
                evaluation.hessian[col * nvar + row] = value;
            }
        }
        if (request.gradient) {
            evaluation.gradient = std::move(gradient);
        }
        evaluation.energy = (*function_)(coords_angstrom.data());
    } else if (request.gradient) {
        evaluation.gradient.assign(nvar, 0.0);
        evaluation.energy = (*function_)(coords_angstrom.data(), evaluation.gradient.data());
    } else {
        evaluation.energy = (*function_)(coords_angstrom.data());
    }
    return evaluation;
}

}  // namespace detail
}  // namespace OEVIBE
