#ifndef OEVIBE_METHOD_H
#define OEVIBE_METHOD_H

#include <string>

namespace OEVIBE {

// Built-in method names (spec 5.3).
constexpr const char* GFN2_XTB = "gfn2-xtb";
constexpr const char* GFN1_XTB = "gfn1-xtb";
constexpr const char* GFN0_XTB = "gfn0-xtb";
constexpr const char* GFN_FF = "gfn-ff";
constexpr const char* MMFF94 = "mmff94";
constexpr const char* MMFF94S = "mmff94s";
constexpr const char* SAGE = "sage";
constexpr const char* PARSLEY = "parsley";

/// Level of theory, identified by a canonical lowercase name (spec 5.3).
class Method {
public:
    /// The default method, ``gfn2-xtb``.
    Method();
    /// :param name: Case-insensitive; stored lowercase.
    /// :raises OptionsError: When empty or containing characters other than letters, digits,
    ///     ``-``, ``+``, ``/``, and ``.``.
    explicit Method(const std::string& name);

    const std::string& GetName() const;

    bool operator==(const Method& other) const;
    bool operator!=(const Method& other) const;
    bool operator<(const Method& other) const;

private:
    std::string name_;
};

/// :returns: The engine a built-in method routes to by default, or an empty string for a
///     method that is not built in (spec 5.3).
std::string builtin_engine_for(const Method& method);

}  // namespace OEVIBE

#endif  // OEVIBE_METHOD_H
