#ifndef OEVIBE_UNITS_H
#define OEVIBE_UNITS_H

namespace OEVIBE {
namespace Units {

// CODATA 2018 values (spec 3.1). Hartree and Bohr appear only inside engine adapters.
constexpr double HARTREE_TO_KCAL = 627.5094740631;
constexpr double HARTREE_TO_EV = 27.211386245988;
constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;
constexpr double HARTREE_PER_BOHR_TO_KCAL_PER_ANGSTROM = HARTREE_TO_KCAL / BOHR_TO_ANGSTROM;
constexpr double GAS_CONSTANT_KCAL = 1.98720425864083e-3;  ///< kcal/(mol K)
constexpr double EBOHR_TO_DEBYE = 2.541746473;
constexpr double PI = 3.14159265358979323846;
constexpr double DEGREES_TO_RADIANS = PI / 180.0;
constexpr double RADIANS_TO_DEGREES = 180.0 / PI;

}  // namespace Units
}  // namespace OEVIBE

#endif  // OEVIBE_UNITS_H
