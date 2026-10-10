#ifndef OEVIBE_PROPERTY_H
#define OEVIBE_PROPERTY_H

#include <string>

namespace OEVIBE {

/// Optional per-conformer properties, stored as a bit set (spec 5.5).
enum class Property : unsigned {
    CHARGES = 1u,
    DIPOLE = 2u,
    ORBITAL_ENERGIES = 4u,
    BOND_ORDERS = 8u,
};

constexpr unsigned ALL_PROPERTIES = 15u;

/// One Wiberg bond order between atoms ``i < j``: dense indices inside the engine contract,
/// OpenEye atom indices in results.
struct BondOrder {
    unsigned i = 0;
    unsigned j = 0;
    double order = 0.0;
};

/// :returns: Comma-separated names of the set bits, for messages.
inline std::string property_names(unsigned properties) {
    static const char* const NAMES[] = {"charges", "dipole", "orbital_energies", "bond_orders"};
    std::string joined;
    for (unsigned bit = 0; bit < 4; ++bit) {
        if ((properties & (1u << bit)) != 0) {
            joined += joined.empty() ? "" : ", ";
            joined += NAMES[bit];
        }
    }
    return joined;
}

}  // namespace OEVIBE

#endif  // OEVIBE_PROPERTY_H
