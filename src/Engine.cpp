#include "oevibe/Engine.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

std::string join(const std::vector<std::string>& parts) {
    std::string joined;
    for (const std::string& part : parts) {
        joined += joined.empty() ? "" : "; ";
        joined += part;
    }
    return joined;
}

bool parse_double(const std::string& text, double& value) {
    try {
        std::size_t used = 0;
        value = std::stod(text, &used);
        return used == text.size() && std::isfinite(value);
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_integer(const std::string& text, long long& value) {
    try {
        std::size_t used = 0;
        value = std::stoll(text, &used);
        return used == text.size();
    } catch (const std::exception&) {
        return false;
    }
}

std::string check_keyword_value(const KeywordSpec& spec, const std::string& value) {
    const std::string label = "keyword '" + spec.name + "'";
    double numeric = 0.0;
    switch (spec.type) {
        case KeywordType::BOOL:
            return (value == "true" || value == "false" || value == "1" || value == "0")
                       ? ""
                       : label + " expects true or false, got '" + value + "'";
        case KeywordType::INT: {
            long long parsed = 0;
            if (!parse_integer(value, parsed)) {
                return label + " expects an integer, got '" + value + "'";
            }
            numeric = static_cast<double>(parsed);
            break;
        }
        case KeywordType::DOUBLE:
            if (!parse_double(value, numeric)) {
                return label + " expects a finite number, got '" + value + "'";
            }
            break;
        case KeywordType::STRING:
            return "";
        default:
            return label + " has an invalid type";
    }
    if (spec.has_min && numeric < spec.min) {
        return label + " must be at least " + std::to_string(spec.min) + ", got '" + value + "'";
    }
    if (spec.has_max && numeric > spec.max) {
        return label + " must be at most " + std::to_string(spec.max) + ", got '" + value + "'";
    }
    return "";
}

bool all_finite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

std::string check_vector(bool requested, const std::vector<double>& values, std::size_t size,
                         const char* name) {
    if (!requested) {
        return values.empty() ? "" : std::string(name) + " was returned but not requested";
    }
    if (values.size() != size) {
        return std::string(name) + " has length " + std::to_string(values.size()) + ", expected " +
               std::to_string(size);
    }
    return all_finite(values) ? "" : std::string(name) + " is not finite";
}

bool wants(const EvaluationRequest& request, Property property) {
    return (request.properties & static_cast<unsigned>(property)) != 0;
}

}  // namespace

std::vector<KeywordSpec> Engine::GetKeywordSpecs() const { return {}; }

void Engine::ValidateSettings(const EngineSettings& settings) const {
    std::vector<std::string> problems;
    if (!find_method(*this, settings.method).has_value()) {
        problems.push_back("engine '" + GetName() + "' does not provide method '" +
                           settings.method.GetName() + "'");
    }
    const std::vector<KeywordSpec> specs = GetKeywordSpecs();
    for (const auto& [name, value] : settings.keywords) {
        // Init-capture because capturing a structured binding is a C++20 extension.
        const auto it = std::find_if(
            specs.begin(), specs.end(),
            [&keyword = name](const KeywordSpec& spec) { return spec.name == keyword; });
        if (it == specs.end()) {
            problems.push_back("unknown keyword '" + name + "' for engine '" + GetName() + "'");
            continue;
        }
        if (!it->methods.empty() && std::find(it->methods.begin(), it->methods.end(),
                                              settings.method) == it->methods.end()) {
            problems.push_back("keyword '" + name + "' does not apply to method '" +
                               settings.method.GetName() + "'");
        }
        const std::string problem = check_keyword_value(*it, value);
        if (!problem.empty()) {
            problems.push_back(problem);
        }
    }
    if (!problems.empty()) {
        throw OptionsError(join(problems));
    }
}

void Engine::ValidateSystem(const MolecularSystem& system, const EngineSettings& settings) const {
    const std::optional<MethodCapabilities> caps = find_method(*this, settings.method);
    if (!caps.has_value()) {
        throw InputError("engine '" + GetName() + "' does not provide method '" +
                         settings.method.GetName() + "'");
    }
    std::vector<std::string> problems;
    if (!caps->elements.empty()) {
        for (unsigned z : system.GetAtomicNumbers()) {
            if (std::find(caps->elements.begin(), caps->elements.end(), z) ==
                caps->elements.end()) {
                problems.push_back("element " + std::to_string(z) + " is not supported by '" +
                                   settings.method.GetName() + "'");
                break;
            }
        }
    }
    if (system.GetMultiplicity() != 1 && !caps->open_shell) {
        problems.push_back("method '" + settings.method.GetName() +
                           "' is closed-shell only, but the multiplicity is " +
                           std::to_string(system.GetMultiplicity()));
    }
    if (caps->charge_from_topology && system.GetCharge() != system.GetFormalChargeSum()) {
        problems.push_back("the total charge " + std::to_string(system.GetCharge()) +
                           " differs from the formal-charge sum " +
                           std::to_string(system.GetFormalChargeSum()) + "; method '" +
                           settings.method.GetName() +
                           "' takes its charge from the atoms' formal charges");
    }
    if (!problems.empty()) {
        throw InputError(join(problems));
    }
}

std::optional<MethodCapabilities> find_method(const Engine& engine, const Method& method) {
    for (const MethodCapabilities& caps : engine.GetMethods()) {
        if (caps.method == method) {
            return caps;
        }
    }
    return std::nullopt;
}

std::string validate_evaluation(const Evaluation& evaluation, const EvaluationRequest& request,
                                unsigned num_atoms) {
    const std::size_t n = num_atoms;
    if (!std::isfinite(evaluation.energy)) {
        return "energy is not finite";
    }
    for (const std::string& problem :
         {check_vector(request.gradient, evaluation.gradient, 3 * n, "gradient"),
          check_vector(request.hessian, evaluation.hessian, 9 * n * n, "hessian"),
          check_vector(wants(request, Property::CHARGES), evaluation.charges, n, "charges"),
          check_vector(wants(request, Property::DIPOLE), evaluation.dipole, 3, "dipole")}) {
        if (!problem.empty()) {
            return problem;
        }
    }
    if (wants(request, Property::ORBITAL_ENERGIES)) {
        const auto& orbitals = evaluation.orbital_energies;
        if (orbitals.empty() || !all_finite(orbitals) ||
            !std::is_sorted(orbitals.begin(), orbitals.end())) {
            return "orbital energies must be present, finite, and ascending";
        }
        if (evaluation.homo_index < -1 ||
            evaluation.homo_index >= static_cast<int>(orbitals.size())) {
            return "orbital HOMO index is out of range";
        }
    } else if (!evaluation.orbital_energies.empty()) {
        return "orbital energies were returned but not requested";
    }
    if (wants(request, Property::BOND_ORDERS)) {
        for (const BondOrder& bond : evaluation.bond_orders) {
            if (bond.i >= bond.j || bond.j >= num_atoms || !std::isfinite(bond.order)) {
                return "bond order entries need i < j < N and a finite order";
            }
        }
    } else if (!evaluation.bond_orders.empty()) {
        return "bond orders were returned but not requested";
    }
    return "";
}

}  // namespace OEVIBE
