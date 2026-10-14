#include "oevibe/OEAdapters.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "core/EngineResolution.h"
#include "core/Hessian.h"
#include "core/InputPrep.h"
#include "core/MolFuncSession.h"
#include "oevibe/Error.h"
#include "oevibe/Preset.h"

namespace OEVIBE {

EngineMolFunc::EngineMolFunc(const CalculationOptions& options) : options_(options) {
    options_.Validate();
    detail::ResolvedEngine resolved = detail::resolve_engine(options_);
    engine_ = std::move(resolved.engine);
    capabilities_ = std::move(resolved.capabilities);
    settings_ = std::move(resolved.settings);
    fd_step_ = preset_values(settings_.preset).fd_step;
}

EngineMolFunc::EngineMolFunc(const CalculationOptions& options, std::shared_ptr<Engine> engine,
                             MethodCapabilities capabilities, EngineSettings settings,
                             double fd_step)
    : options_(options),
      engine_(std::move(engine)),
      capabilities_(std::move(capabilities)),
      settings_(std::move(settings)),
      fd_step_(fd_step) {}

EngineMolFunc::~EngineMolFunc() = default;

bool EngineMolFunc::Setup(const OEChem::OEMolBase& mol) {
    setup_error_.clear();
    // The session may hold references into the system, so it is released first everywhere.
    session_.reset();
    system_.reset();
    num_var_ = 0;
    caller_index_of_dense_.clear();
    cached_x_.clear();
    cached_gradient_.clear();
    cached_has_gradient_ = false;
    try {
        // Read from the caller's molecule before copying: OEMol's copy compacts index gaps, so
        // afterwards neither the copy nor the snapshot can say what the caller's indices were.
        const unsigned expected_atoms = mol.NumAtoms();
        const unsigned caller_max_index = mol.GetMaxAtomIdx();
        std::vector<unsigned> caller_indices;
        caller_indices.reserve(expected_atoms);
        for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
            caller_indices.push_back(atom->GetIdx());
        }
        // A private copy, because prepare_input modifies what it is given and Setup's argument
        // is const. MolFuncSession copies an OEMolBase into an OEMol the same way.
        molecule_ = OEChem::OEMol(mol);
        const detail::PreparedInput prepared = detail::prepare_input(molecule_, options_);
        if (molecule_.NumAtoms() != expected_atoms) {
            // The caller's coordinate arrays are sized from its own molecule, so a preparation
            // that adds atoms would make every index past the first new one read out of bounds.
            setup_error_ = "input preparation changed the atoms of '" +
                           std::string(mol.GetTitle()) + "'; add explicit hydrogens before Setup";
            return false;
        }
        system_ =
            std::make_unique<MolecularSystem>(molecule_, prepared.charge, prepared.multiplicity,
                                              detail::charge_policy(options_, capabilities_));
        engine_->ValidateSystem(*system_, settings_);
        session_ = engine_->CreateSession(*system_, settings_);
        if (!session_) {
            setup_error_ = "engine '" + engine_->GetName() + "' returned a null session";
            session_.reset();
            system_.reset();
            return false;
        }
        caller_index_of_dense_ = std::move(caller_indices);
        num_var_ = 3 * caller_max_index;
        return true;
    } catch (const std::exception& error) {
        setup_error_ = error.what();
        session_.reset();
        system_.reset();
        num_var_ = 0;
        caller_index_of_dense_.clear();
        return false;
    }
}

unsigned int EngineMolFunc::NumVar() const { return num_var_; }

const std::string& EngineMolFunc::GetSetupError() const { return setup_error_; }

std::vector<double> EngineMolFunc::ToDense(const double* x) const {
    std::vector<double> dense(3 * caller_index_of_dense_.size(), 0.0);
    for (std::size_t i = 0; i < caller_index_of_dense_.size(); ++i) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            dense[3 * i + axis] = x[3 * caller_index_of_dense_[i] + axis];
        }
    }
    return dense;
}

double EngineMolFunc::EvaluateCached(const double* x, bool need_gradient) {
    if (session_ == nullptr) {
        throw OEVibeError("EngineMolFunc was called before a successful Setup");
    }
    const bool same_geometry =
        cached_x_.size() == num_var_ && std::equal(cached_x_.begin(), cached_x_.end(), x);
    if (same_geometry && (!need_gradient || cached_has_gradient_)) {
        return cached_energy_;
    }
    EvaluationRequest request;
    request.gradient = need_gradient;
    const Evaluation evaluation = session_->Evaluate(ToDense(x), request);
    const std::string problem = validate_evaluation(
        evaluation, request, static_cast<unsigned>(caller_index_of_dense_.size()));
    if (!problem.empty()) {
        throw std::runtime_error("engine '" + engine_->GetName() + "' returned " + problem);
    }
    cached_x_.assign(x, x + num_var_);
    cached_energy_ = evaluation.energy;
    cached_gradient_ = evaluation.gradient;
    cached_has_gradient_ = need_gradient;
    return cached_energy_;
}

void EngineMolFunc::AccumulateGradient(double* g) const {
    for (std::size_t i = 0; i < caller_index_of_dense_.size(); ++i) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            g[3 * caller_index_of_dense_[i] + axis] += cached_gradient_[3 * i + axis];
        }
    }
}

double EngineMolFunc::operator()(const double* x) { return EvaluateCached(x, false); }

double EngineMolFunc::operator()(const double* x, double* g) {
    const double energy = EvaluateCached(x, true);
    AccumulateGradient(g);
    return energy;
}

bool EngineMolFunc::operator()(const double* x, double* h, double* g) {
    try {
        EvaluateCached(x, true);
        const std::vector<double> dense_hessian = detail::compute_hessian(
            *session_, ToDense(x), capabilities_.analytic_hessian, fd_step_);
        // Only the dense lower triangle: the matrix is symmetric and the mapping is injective,
        // so visiting both (u, v) and (v, u) would double every off-diagonal element.
        const std::size_t n = 3 * caller_index_of_dense_.size();
        for (std::size_t u = 0; u < n; ++u) {
            const std::size_t p = 3 * caller_index_of_dense_[u / 3] + (u % 3);
            for (std::size_t v = 0; v <= u; ++v) {
                const std::size_t q = 3 * caller_index_of_dense_[v / 3] + (v % 3);
                const std::size_t row = std::max(p, q);
                const std::size_t col = std::min(p, q);
                h[row * (row + 1) / 2 + col] += dense_hessian[u * n + v];
            }
        }
        AccumulateGradient(g);
        return true;
    } catch (const std::exception& error) {
        setup_error_ = error.what();
        return false;
    }
}

EngineMolFunc::FunctionValue EngineMolFunc::Evaluate(const std::vector<double>& x,
                                                     bool want_gradient) {
    if (x.size() != num_var_) {
        throw OEVibeError("expected " + std::to_string(num_var_) + " coordinates, got " +
                          std::to_string(x.size()));
    }
    FunctionValue value;
    value.energy = EvaluateCached(x.data(), want_gradient);
    if (want_gradient) {
        // Accumulating into a fresh zeroed array is this caller's assignment: every dense
        // position maps to a distinct variable, and the gaps keep the zero they start with.
        value.gradient.assign(num_var_, 0.0);
        AccumulateGradient(value.gradient.data());
    }
    return value;
}

OESystem::OEIterBase<OEOpt::OEFComponent>* EngineMolFunc::GetFComponents(const double*) {
    return nullptr;
}

EngineMolFunc* EngineMolFunc::CreateCopy() const {
    // Not the public constructor: it would resolve the options against the registry again, and a
    // copy taken after an engine was replaced under the same name, or after a second provider of
    // the method appeared, would not be a copy of this function.
    return new EngineMolFunc(options_, engine_, capabilities_, settings_, fd_step_);
}

MolFuncEngine::MolFuncEngine(std::string name, std::string version, Method method,
                             const OEMolPotential::OEMolFunc1& prototype,
                             bool requires_partial_charges)
    : name_(std::move(name)),
      version_(std::move(version)),
      method_(std::move(method)),
      prototype_(prototype.CreateCopy()),
      requires_partial_charges_(requires_partial_charges) {
    // Some OpenEye potentials (scaled and adaptor wrappers) return null from CreateCopy().
    if (!prototype_) {
        throw OptionsError("OpenEye potential for engine '" + name_ +
                           "' cannot be copied; MolFuncEngine needs a copyable potential");
    }
}

std::string MolFuncEngine::GetName() const { return name_; }
std::string MolFuncEngine::GetVersion() const { return version_; }
bool MolFuncEngine::IsAvailable() const { return true; }
std::string MolFuncEngine::GetUnavailableReason() const { return ""; }

std::vector<MethodCapabilities> MolFuncEngine::GetMethods() const {
    MethodCapabilities caps;
    caps.method = method_;
    caps.analytic_gradient = true;
    caps.analytic_hessian =
        dynamic_cast<const OEMolPotential::OEMolFunc2*>(prototype_.get()) != nullptr;
    caps.requires_partial_charges = requires_partial_charges_;
    caps.charge_from_topology = true;
    caps.force_field = true;
    return {caps};
}

std::shared_ptr<EngineSession> MolFuncEngine::CreateSession(const MolecularSystem& system,
                                                            const EngineSettings& settings) {
    static_cast<void>(settings);
    std::unique_ptr<OEMolPotential::OEMolFunc1> copy(prototype_->CreateCopy());
    if (!copy) {
        throw UnsupportedError("OpenEye potential for engine '" + name_ +
                               "' could not be copied for a session");
    }
    return std::make_shared<detail::MolFuncSession>(std::move(copy), system);
}

}  // namespace OEVIBE
