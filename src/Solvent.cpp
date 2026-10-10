#include "oevibe/Solvent.h"

#include <cctype>

#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

void require_model(SolventModel model) {
    const auto raw = static_cast<long long>(model);
    if (raw < 0 || raw > static_cast<long long>(SolventModel::SHEFFIELD)) {
        throw OptionsError("SolventModel must be one of its enumerators, got " +
                           std::to_string(raw));
    }
}

}  // namespace

std::string solvent_model_name(SolventModel model) {
    require_model(model);
    switch (model) {
        case SolventModel::AUTO:
            return "auto";
        case SolventModel::ALPB:
            return "alpb";
        case SolventModel::GBSA:
            return "gbsa";
        case SolventModel::SHEFFIELD:
            return "sheffield";
    }
    return "auto";
}

Solvent::Solvent() = default;

Solvent::Solvent(const std::string& name, SolventModel model) {
    require_model(model);
    std::string lowered;
    for (char c : name) {
        const auto u = static_cast<unsigned char>(c);
        if (std::isspace(u) != 0 || std::iscntrl(u) != 0) {
            throw OptionsError("Solvent name '" + name + "' must not contain whitespace");
        }
        lowered.push_back(static_cast<char>(std::tolower(u)));
    }
    if (lowered.empty()) {
        throw OptionsError("Solvent name must not be empty; use Solvent() for gas phase");
    }
    name_ = lowered;
    model_ = model;
}

bool Solvent::IsGasPhase() const { return name_.empty(); }
const std::string& Solvent::GetName() const { return name_; }
SolventModel Solvent::GetModel() const { return model_; }

std::string Solvent::GetTag() const {
    if (IsGasPhase()) {
        return "gas";
    }
    if (model_ == SolventModel::AUTO) {
        throw OptionsError("solvent '" + name_ + "' has no resolved model yet");
    }
    return solvent_model_name(model_) + "-" + name_;
}

Solvent Solvent::WithModel(SolventModel model) const {
    if (IsGasPhase()) {
        return *this;
    }
    require_model(model);
    Solvent copy = *this;
    copy.model_ = model;
    return copy;
}

bool Solvent::operator==(const Solvent& other) const {
    return name_ == other.name_ && model_ == other.model_;
}

}  // namespace OEVIBE
