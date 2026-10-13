#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "core/Thermochemistry.h"
#include "oevibe/Error.h"
#include "oevibe/Result.h"

using namespace OEVIBE;

namespace {

// CODATA 2018, restated so the expected values do not reuse the implementation's constants.
constexpr double TEST_PI = 3.14159265358979323846;
constexpr double H = 6.62607015e-34;       // J s
constexpr double KB = 1.380649e-23;        // J/K
constexpr double NA = 6.02214076e23;       // 1/mol
constexpr double C_CM = 2.99792458e10;     // cm/s
constexpr double AMU = 1.66053906660e-27;  // kg
constexpr double CAL = 4.184;              // J
constexpr double R_CAL = KB * NA / CAL;    // cal/(mol K)
constexpr double R_KCAL = R_CAL / 1000.0;  // kcal/(mol K)
constexpr double T298 = 298.15;
constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();
constexpr double INFINITE = std::numeric_limits<double>::infinity();

// Water's MMFF94 principal moments (amu Angstrom^2), the reference values of test_frequencies.cpp,
// and its mass.
const std::vector<double> WATER_MOMENTS = {0.637488, 1.175023, 1.812511};
constexpr double WATER_MASS = 2 * 1.00794 + 15.9994;
const std::vector<double> NO_MOMENTS = {0.0, 0.0, 0.0};

Thermochemistry thermo(const std::vector<double>& frequencies,
                       const std::vector<double>& moments = NO_MOMENTS, double mass = WATER_MASS,
                       unsigned symmetry = 1, bool linear = false, unsigned multiplicity = 1,
                       double temperature = T298, double pressure = 1.0, double cutoff = 100.0) {
    return detail::compute_thermochemistry(-10.0, frequencies, moments, mass, symmetry, linear,
                                           multiplicity, temperature, pressure, cutoff);
}

/// Rotational constant h / (8 pi^2 c I) in cm^-1 of a moment in amu Angstrom^2.
double rotational_constant(double moment) {
    return H / (8.0 * TEST_PI * TEST_PI * C_CM * moment * AMU * 1e-20);
}

/// x = h c nu / (k T) for ``wavenumber`` cm^-1.
double reduced(double wavenumber, double temperature = T298) {
    return H * C_CM * wavenumber / (KB * temperature);
}

/// Harmonic entropy in cal/(mol K).
double harmonic_entropy(double wavenumber) {
    const double x = reduced(wavenumber);
    return R_CAL * (x / (std::exp(x) - 1.0) - std::log(1.0 - std::exp(-x)));
}

/// Free-rotor entropy in cal/(mol K) (Grimme, Chem. Eur. J. 18, 9955, 2012, eqs. 4-6).
double free_rotor_entropy(double wavenumber) {
    const double mu = H / (8.0 * TEST_PI * TEST_PI * C_CM * wavenumber);
    const double average = 1e-44;
    const double effective = mu * average / (mu + average);
    return R_CAL *
           (0.5 + 0.5 * std::log(8.0 * std::pow(TEST_PI, 3) * effective * KB * T298 / (H * H)));
}

/// Harmonic vibrational enthalpy, ZPE included, in kcal/mol.
double harmonic_enthalpy(double wavenumber) {
    const double zpe = 0.5 * H * C_CM * wavenumber * NA / CAL / 1000.0;
    return zpe + R_KCAL * T298 * reduced(wavenumber) / (std::exp(reduced(wavenumber)) - 1.0);
}

}  // namespace

// Literature: S(Ar, g, 298.15 K, 1 bar) = 154.846 J/(mol K) (CODATA Key Values for
// Thermodynamics, Cox, Wagman, and Medvedev, 1989); at 1 atm, 154.846 - R ln(1.01325) =
// 154.7366 J/(mol K) = 36.9829 cal/(mol K). The table's uncertainty is 0.003 J/(mol K).
TEST(ThermochemistryTest, MonatomicEntropyIsSackurTetrode) {
    const Thermochemistry argon = thermo({}, NO_MOMENTS, 39.948);
    EXPECT_NEAR(argon.entropy, 36.9829, 2e-3);
    EXPECT_DOUBLE_EQ(argon.zpe, 0.0);
    // E_trans + PV = 5/2 RT.
    EXPECT_NEAR(argon.enthalpy - argon.electronic_energy, 2.5 * R_KCAL * T298, 1e-12);
    const Thermochemistry compressed = thermo({}, NO_MOMENTS, 39.948, 1, false, 1, T298, 2.0);
    EXPECT_NEAR(argon.entropy - compressed.entropy, R_CAL * std::log(2.0), 1e-9);
}

TEST(ThermochemistryTest, NonlinearRotorEntropyAndEnergy) {
    const double t = T298;
    // q_rot = sqrt(pi) / sigma * sqrt((k T / h c)^3 / (A B C)), S_rot = R (ln q_rot + 3/2).
    const double kt_hc = KB * t / (H * C_CM);
    const double abc = rotational_constant(WATER_MOMENTS[0]) *
                       rotational_constant(WATER_MOMENTS[1]) *
                       rotational_constant(WATER_MOMENTS[2]);
    const double expected =
        R_CAL * (std::log(std::sqrt(TEST_PI) / 2.0 * std::sqrt(std::pow(kt_hc, 3) / abc)) + 1.5);
    const Thermochemistry rotor = thermo({}, WATER_MOMENTS, WATER_MASS, 2);
    const Thermochemistry atom = thermo({}, NO_MOMENTS, WATER_MASS);
    EXPECT_NEAR(rotor.entropy - atom.entropy, expected, 1e-9);
    EXPECT_NEAR(rotor.enthalpy - atom.enthalpy, 1.5 * R_KCAL * t, 1e-12);
    const Thermochemistry unsymmetric = thermo({}, WATER_MOMENTS, WATER_MASS, 1);
    EXPECT_NEAR(unsymmetric.entropy - rotor.entropy, R_CAL * std::log(2.0), 1e-9);
}

TEST(ThermochemistryTest, LinearRotorEntropyAndEnergy) {
    // CO2 at its MMFF94 minimum: I = 2 m_O r^2 with r = 1.1970 Angstrom.
    const double moment = 2.0 * 15.9994 * 1.1970 * 1.1970;
    const std::vector<double> moments = {0.0, moment, moment};
    const double mass = 12.0107 + 2.0 * 15.9994;
    const double expected =
        R_CAL * (std::log(KB * T298 / (2.0 * H * C_CM * rotational_constant(moment))) + 1.0);
    const Thermochemistry rotor = thermo({}, moments, mass, 2, true);
    const Thermochemistry atom = thermo({}, NO_MOMENTS, mass);
    EXPECT_NEAR(rotor.entropy - atom.entropy, expected, 1e-9);
    EXPECT_NEAR(rotor.enthalpy - atom.enthalpy, R_KCAL * T298, 1e-12);
}

TEST(ThermochemistryTest, HighFrequencyModeIsZeroPointEnergyOnly) {
    const Thermochemistry none = thermo({});
    const Thermochemistry stiff = thermo({3000.0});
    // ZPE = N_A h c nu / 2 = 4.288715 kcal/mol for 3000 cm^-1.
    EXPECT_NEAR(stiff.zpe, 0.5 * H * C_CM * 3000.0 * NA / CAL / 1000.0, 1e-12);
    EXPECT_NEAR(stiff.zpe, 4.288715, 1e-6);
    EXPECT_NEAR(stiff.enthalpy - none.enthalpy, harmonic_enthalpy(3000.0), 1e-12);
    EXPECT_GT(stiff.entropy, none.entropy);
    EXPECT_LT(stiff.entropy - none.entropy, 1e-4);
}

TEST(ThermochemistryTest, QuasiRrhoInterpolatesFreeRotorAndHarmonicEntropy) {
    const Thermochemistry none = thermo({});
    for (double nu : {1.0, 30.0, 100.0, 400.0, 3000.0}) {
        const double weight = 1.0 / (1.0 + std::pow(100.0 / nu, 4));
        const double expected =
            weight * harmonic_entropy(nu) + (1.0 - weight) * free_rotor_entropy(nu);
        EXPECT_NEAR(thermo({nu}).entropy - none.entropy, expected, 1e-9) << nu << " cm^-1";
    }
    // At the cutoff the weight is exactly one half.
    EXPECT_NEAR(thermo({100.0}).entropy - none.entropy,
                0.5 * (harmonic_entropy(100.0) + free_rotor_entropy(100.0)), 1e-9);
    // Far below the cutoff the free rotor dominates; far above it the harmonic oscillator does.
    EXPECT_NEAR(thermo({1.0}).entropy - none.entropy, free_rotor_entropy(1.0), 1e-6);
    EXPECT_NEAR(thermo({3000.0}).entropy - none.entropy, harmonic_entropy(3000.0), 1e-5);
    // Enthalpy stays harmonic for every real mode, however soft.
    EXPECT_NEAR(thermo({10.0}).enthalpy - none.enthalpy, harmonic_enthalpy(10.0), 1e-12);
    // A different cutoff changes the weight.
    const double weight50 = 1.0 / (1.0 + std::pow(50.0 / 100.0, 4));
    const Thermochemistry custom =
        thermo({100.0}, NO_MOMENTS, WATER_MASS, 1, false, 1, T298, 1.0, 50.0);
    EXPECT_NEAR(custom.entropy - none.entropy,
                weight50 * harmonic_entropy(100.0) + (1.0 - weight50) * free_rotor_entropy(100.0),
                1e-9);
}

TEST(ThermochemistryTest, ImaginaryModesAreExcludedButKept) {
    const Thermochemistry real = thermo({1500.0});
    const Thermochemistry with_imaginary = thermo({-200.0, 1500.0});
    EXPECT_DOUBLE_EQ(with_imaginary.zpe, real.zpe);
    EXPECT_DOUBLE_EQ(with_imaginary.enthalpy, real.enthalpy);
    EXPECT_DOUBLE_EQ(with_imaginary.entropy, real.entropy);
    EXPECT_EQ(with_imaginary.frequencies, (std::vector<double>{-200.0, 1500.0}));
}

TEST(ThermochemistryTest, DoubletAddsElectronicEntropy) {
    const Thermochemistry singlet = thermo({1500.0});
    const Thermochemistry doublet = thermo({1500.0}, NO_MOMENTS, WATER_MASS, 1, false, 2);
    // R ln 2 = 1.377425 cal/(mol K).
    EXPECT_NEAR(doublet.entropy - singlet.entropy, 1.377425, 1e-6);
    EXPECT_NEAR(doublet.entropy - singlet.entropy, R_CAL * std::log(2.0), 1e-9);
    EXPECT_NEAR(doublet.free_energy - singlet.free_energy, -T298 * R_KCAL * std::log(2.0), 1e-9);
    EXPECT_DOUBLE_EQ(doublet.enthalpy, singlet.enthalpy);
}

TEST(ThermochemistryTest, TotalsFollowTheConvention) {
    const std::vector<double> frequencies = {-50.0, 80.0, 1600.0, 3700.0};
    const Thermochemistry result = detail::compute_thermochemistry(
        -12.5, frequencies, WATER_MOMENTS, WATER_MASS, 2, false, 3, 350.0, 1.5, 120.0);
    double vibrational = 0.0;  // ZPE plus thermal, harmonic, real modes
    double zpe = 0.0;
    for (double nu : {80.0, 1600.0, 3700.0}) {
        const double x = reduced(nu, 350.0);
        zpe += 0.5 * H * C_CM * nu * NA / CAL / 1000.0;
        vibrational +=
            0.5 * H * C_CM * nu * NA / CAL / 1000.0 + R_KCAL * 350.0 * x / (std::exp(x) - 1.0);
    }
    const double rt = R_KCAL * 350.0;
    EXPECT_NEAR(result.zpe, zpe, 1e-12);
    // H = E + ZPE + E_vib + E_trans (3/2 RT) + E_rot (3/2 RT) + PV (RT).
    EXPECT_NEAR(result.enthalpy, -12.5 + vibrational + 4.0 * rt, 1e-10);
    EXPECT_NEAR(result.free_energy, result.enthalpy - 350.0 * result.entropy / 1000.0, 1e-12);
    EXPECT_NEAR(result.free_energy_correction, result.free_energy + 12.5, 1e-12);
    EXPECT_DOUBLE_EQ(result.electronic_energy, -12.5);
    EXPECT_DOUBLE_EQ(result.temperature, 350.0);
    EXPECT_DOUBLE_EQ(result.pressure, 1.5);
    EXPECT_EQ(result.frequencies, frequencies);
    EXPECT_EQ(result.principal_moments, WATER_MOMENTS);
    EXPECT_DOUBLE_EQ(result.total_mass, WATER_MASS);
    EXPECT_EQ(result.symmetry_number, 2u);
    EXPECT_FALSE(result.linear);
    EXPECT_EQ(result.multiplicity, 3u);
    EXPECT_DOUBLE_EQ(result.quasi_rrho_cutoff, 120.0);
}

TEST(ThermochemistryTest, RecomputeAtAnotherTemperatureMatchesAFreshCompute) {
    const std::vector<double> frequencies = {-30.0, 45.0, 900.0, 1700.0, 3100.0};
    const Thermochemistry stored = detail::compute_thermochemistry(
        -7.0, frequencies, WATER_MOMENTS, WATER_MASS, 2, false, 2, T298, 2.0, 100.0);
    const Thermochemistry recomputed = detail::recompute_thermochemistry(stored, 400.0);
    const Thermochemistry fresh = detail::compute_thermochemistry(
        -7.0, frequencies, WATER_MOMENTS, WATER_MASS, 2, false, 2, 400.0, 2.0, 100.0);
    EXPECT_DOUBLE_EQ(recomputed.temperature, 400.0);
    EXPECT_DOUBLE_EQ(recomputed.pressure, 2.0);
    EXPECT_DOUBLE_EQ(recomputed.zpe, fresh.zpe);
    EXPECT_DOUBLE_EQ(recomputed.enthalpy, fresh.enthalpy);
    EXPECT_DOUBLE_EQ(recomputed.entropy, fresh.entropy);
    EXPECT_DOUBLE_EQ(recomputed.free_energy, fresh.free_energy);
    EXPECT_DOUBLE_EQ(recomputed.free_energy_correction, fresh.free_energy_correction);
    EXPECT_NE(recomputed.free_energy, stored.free_energy);
}

TEST(ThermochemistryTest, InvalidInputsRaise) {
    const std::vector<double> f = {1500.0};
    const std::vector<double>& m = WATER_MOMENTS;
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, 0.0), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, -1.0), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, NAN_VALUE), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, T298, 0.0), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, T298, NAN_VALUE), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, T298, 1.0, 0.0), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 1, T298, 1.0, INFINITE), OEVibeError);
    EXPECT_THROW(thermo(f, m, 0.0), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 0), OEVibeError);
    EXPECT_THROW(thermo(f, m, WATER_MASS, 1, false, 0), OEVibeError);
    EXPECT_THROW(thermo(f, {1.0, 2.0}), OEVibeError);
    EXPECT_THROW(thermo(f, {-0.5, 1.0, 2.0}), OEVibeError);
    EXPECT_THROW(thermo({NAN_VALUE}), OEVibeError);
    EXPECT_THROW(thermo(f, NO_MOMENTS, WATER_MASS, 1, true), OEVibeError);
    EXPECT_THROW(thermo(f, {0.0, 1.0, 1.0}, WATER_MASS, 1, false), OEVibeError);
    EXPECT_THROW(
        detail::compute_thermochemistry(NAN_VALUE, f, m, WATER_MASS, 1, false, 1, T298, 1.0, 100.0),
        OEVibeError);
}
