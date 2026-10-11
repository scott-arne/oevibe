#ifndef OEVIBE_TEST_SUPPORT_H
#define OEVIBE_TEST_SUPPORT_H

#include <oechem.h>
#include <oeomega2.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "oevibe/Engine.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"

namespace OEVIBE {
namespace test {

/// SMILES to a 3D OEMol with explicit hydrogens and up to ``max_confs`` OMEGA conformers.
inline OEChem::OEMol make_conformers(const std::string& smiles, unsigned max_confs = 1) {
    OEChem::OEMol mol;
    if (!OEChem::OESmilesToMol(mol, smiles)) {
        throw std::runtime_error("cannot parse SMILES " + smiles);
    }
    OEConfGen::OEOmegaOptions options;
    options.SetMaxConfs(max_confs);
    OEConfGen::OEOmega omega(options);
    if (omega.Build(mol) != OEConfGen::OEOmegaReturnCode::Success) {
        throw std::runtime_error("OMEGA failed for " + smiles);
    }
    return mol;
}

/// Methane whose atom indices have a gap: C is 0, a deleted helium was 1, hydrogens are 2-5.
inline OEChem::OEMol make_gapped_methane() {
    OEChem::OEMol mol;
    OEChem::OEAtomBase* carbon = mol.NewAtom(6);
    OEChem::OEAtomBase* dummy = mol.NewAtom(2);
    std::vector<OEChem::OEAtomBase*> hydrogens;
    for (int k = 0; k < 4; ++k) {
        OEChem::OEAtomBase* hydrogen = mol.NewAtom(1);
        mol.NewBond(carbon, hydrogen, 1);
        hydrogens.push_back(hydrogen);
    }
    mol.DeleteAtom(dummy);
    const double positions[4][3] = {{0.629, 0.629, 0.629},
                                    {-0.629, -0.629, 0.629},
                                    {-0.629, 0.629, -0.629},
                                    {0.629, -0.629, -0.629}};
    std::vector<double> coords(3 * mol.GetMaxAtomIdx(), 0.0);
    for (int k = 0; k < 4; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
            coords[3 * hydrogens[k]->GetIdx() + axis] = positions[k][axis];
        }
    }
    mol.SetCoords(coords.data());
    mol.SetDimension(3);
    return mol;
}

/// Spring network over all atom pairs, E = 1/2 k sum (r_ij - r0_ij)^2 in kcal/mol with r0 from
/// ``reference``. Its minima are rigid motions of the reference, so it is rotation invariant
/// like a real potential.
inline double spring_energy(const std::vector<double>& reference, const std::vector<double>& x,
                            double force_constant, std::vector<double>* gradient) {
    const std::size_t n = x.size() / 3;
    if (gradient != nullptr) {
        gradient->assign(x.size(), 0.0);
    }
    double energy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double d[3];
            double d0[3];
            for (int a = 0; a < 3; ++a) {
                d[a] = x[3 * i + a] - x[3 * j + a];
                d0[a] = reference[3 * i + a] - reference[3 * j + a];
            }
            const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            const double r0 = std::sqrt(d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2]);
            const double deviation = r - r0;
            energy += 0.5 * force_constant * deviation * deviation;
            if (gradient != nullptr && r > 0.0) {
                for (int a = 0; a < 3; ++a) {
                    const double g = force_constant * deviation * d[a] / r;
                    (*gradient)[3 * i + a] += g;
                    (*gradient)[3 * j + a] -= g;
                }
            }
        }
    }
    return energy;
}

/// A non-planar six-atom geometry in Angstrom for numerical tests.
inline std::vector<double> sample_geometry() {
    return {0.000, 0.000, 0.000, 1.530,  0.000,  0.000, 2.040, 1.440,  0.000,
            3.570, 1.440, 0.420, -0.510, -0.880, 0.510, 1.980, -0.700, -0.880};
}

/// Deterministic perturbation so tests start away from the reference.
inline std::vector<double> perturbed(std::vector<double> x, double amplitude) {
    for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] += amplitude * std::sin(1.7 * static_cast<double>(k) + 0.3);
    }
    return x;
}

enum class FakeMode { HARMONIC, THROW_UNSUPPORTED, THROW_RUNTIME, MALFORMED_GRADIENT };

/// Session of ``FakeEngine``: spring network around conformer 0, plus fixed properties.
class FakeSession : public EngineSession {
public:
    FakeSession(std::vector<double> reference, FakeMode mode, double force_constant,
                std::vector<std::pair<unsigned, unsigned>> bonds)
        : reference_(std::move(reference)),
          mode_(mode),
          force_constant_(force_constant),
          bonds_(std::move(bonds)) {}

    Evaluation Evaluate(const std::vector<double>& coords,
                        const EvaluationRequest& request) override {
        if (mode_ == FakeMode::THROW_RUNTIME) {
            throw std::runtime_error("fake engine failure");
        }
        Evaluation evaluation;
        evaluation.energy = spring_energy(reference_, coords, force_constant_,
                                          request.gradient ? &evaluation.gradient : nullptr);
        if (mode_ == FakeMode::MALFORMED_GRADIENT && request.gradient) {
            evaluation.gradient.pop_back();
        }
        const std::size_t n = coords.size() / 3;
        if ((request.properties & static_cast<unsigned>(Property::CHARGES)) != 0) {
            // Coordinate-dependent so different conformers produce different charges.
            for (std::size_t i = 0; i < n; ++i) {
                evaluation.charges.push_back(0.001 * coords[3 * i]);
            }
        }
        if ((request.properties & static_cast<unsigned>(Property::DIPOLE)) != 0) {
            evaluation.dipole = {0.0, 0.0, 1.0};
        }
        if ((request.properties & static_cast<unsigned>(Property::ORBITAL_ENERGIES)) != 0) {
            evaluation.orbital_energies = {-12.0, -9.0, 1.0, 3.0};
            evaluation.homo_index = 1;
        }
        if ((request.properties & static_cast<unsigned>(Property::BOND_ORDERS)) != 0) {
            for (const auto& bond : bonds_) {
                evaluation.bond_orders.push_back({bond.first, bond.second, 1.0});
            }
        }
        return evaluation;
    }

private:
    std::vector<double> reference_;
    FakeMode mode_;
    double force_constant_;
    std::vector<std::pair<unsigned, unsigned>> bonds_;
};

inline MethodCapabilities fake_capabilities(const std::string& method) {
    MethodCapabilities capabilities;
    capabilities.method = Method(method);
    capabilities.properties = ALL_PROPERTIES;
    capabilities.solvent_models = {SolventModel::ALPB};
    capabilities.open_shell = true;
    return capabilities;
}

/// Configurable engine for tests; the public members are knobs the tests set directly.
class FakeEngine : public Engine {
public:
    explicit FakeEngine(
        std::string name,
        std::vector<MethodCapabilities> method_list = {fake_capabilities("fake-method")})
        : methods(std::move(method_list)), name_(std::move(name)) {}

    std::string GetName() const override { return name_; }
    std::string GetVersion() const override { return "1.0"; }
    bool IsAvailable() const override { return available; }
    std::string GetUnavailableReason() const override { return unavailable_reason; }
    std::vector<MethodCapabilities> GetMethods() const override { return methods; }
    std::vector<KeywordSpec> GetKeywordSpecs() const override { return keyword_specs; }

    std::shared_ptr<EngineSession> CreateSession(const MolecularSystem& system,
                                                 const EngineSettings& settings) override {
        ++sessions_created;
        last_settings = settings;
        if (mode == FakeMode::THROW_UNSUPPORTED) {
            throw UnsupportedError("fake engine cannot handle this system");
        }
        std::vector<std::pair<unsigned, unsigned>> bonds;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = system.GetMolecule().GetBonds(); bond;
             ++bond) {
            bonds.emplace_back(std::min(bond->GetBgnIdx(), bond->GetEndIdx()),
                               std::max(bond->GetBgnIdx(), bond->GetEndIdx()));
        }
        return std::make_shared<FakeSession>(
            reference.empty() ? system.GetConformerCoordinates(0) : reference, mode, force_constant,
            bonds);
    }

    bool available = true;
    std::string unavailable_reason;
    std::vector<MethodCapabilities> methods;
    std::vector<KeywordSpec> keyword_specs;
    FakeMode mode = FakeMode::HARMONIC;
    double force_constant = 50.0;
    std::vector<double> reference;
    int sessions_created = 0;
    EngineSettings last_settings;

private:
    std::string name_;
};

/// Registers (or replaces) a fake engine under ``name`` in the process-wide registry.
inline std::shared_ptr<FakeEngine> register_fake(const std::string& name) {
    auto engine = std::make_shared<FakeEngine>(name);
    EngineRegistry::Instance().Register(engine, true);
    return engine;
}

}  // namespace test
}  // namespace OEVIBE

#endif  // OEVIBE_TEST_SUPPORT_H
