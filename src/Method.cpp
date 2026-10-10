#include "oevibe/Method.h"

#include <cctype>
#include <map>

#include "oevibe/Error.h"

namespace OEVIBE {

namespace {

bool is_method_char(char c) {
    const auto u = static_cast<unsigned char>(c);
    return std::islower(u) != 0 || std::isdigit(u) != 0 || c == '-' || c == '+' || c == '/' ||
           c == '.';
}

}  // namespace

Method::Method() : name_(GFN2_XTB) {}

Method::Method(const std::string& name) {
    std::string lowered;
    lowered.reserve(name.size());
    for (char c : name) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lowered.empty()) {
        throw OptionsError("Method name must not be empty");
    }
    for (char c : lowered) {
        if (!is_method_char(c)) {
            throw OptionsError("Method name '" + name + "' contains an invalid character");
        }
    }
    name_ = lowered;
}

const std::string& Method::GetName() const { return name_; }

bool Method::operator==(const Method& other) const { return name_ == other.name_; }
bool Method::operator!=(const Method& other) const { return name_ != other.name_; }
bool Method::operator<(const Method& other) const { return name_ < other.name_; }

std::string builtin_engine_for(const Method& method) {
    static const std::map<std::string, std::string> ROUTING = {
        {GFN2_XTB, "tblite"}, {GFN1_XTB, "tblite"}, {GFN0_XTB, "xtb"}, {GFN_FF, "xtb"},
        {MMFF94, "oeff"},     {MMFF94S, "oeff"},    {SAGE, "oeff"},    {PARSLEY, "oeff"},
    };
    const auto it = ROUTING.find(method.GetName());
    return it == ROUTING.end() ? std::string() : it->second;
}

}  // namespace OEVIBE
