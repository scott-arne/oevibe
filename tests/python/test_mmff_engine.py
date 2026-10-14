"""The Python MMFF engine must reproduce the C++ oeff engine (spec 8.7, spec 12.2)."""

import numpy as np
import pytest
from mmff_engine import ENGINE_NAME, MMFFEngine
from openeye import oechem, oeomega

import oevibe


@pytest.fixture(scope="module", autouse=True)
def registered():
    oevibe.register_engine(MMFFEngine(), replace=True)


def options(engine, method="mmff94s"):
    opts = oevibe.CalculationOptions()
    opts.SetEngine(engine)
    opts.SetMethod(oevibe.Method(method))
    return opts


def coordinates(result):
    return np.array(list(result.GetCoordinates()), dtype=np.float64).reshape(-1, 3)


@pytest.mark.parametrize("method", ["mmff94", "mmff94s"])
def test_single_point_energy_matches_the_cpp_engine(butanol_conformers, method):
    python_results = oevibe.Calculation(options(ENGINE_NAME, method)).Energy(
        oechem.OEMol(butanol_conformers)
    )
    cpp_results = oevibe.Calculation(options("oeff", method)).Energy(
        oechem.OEMol(butanol_conformers)
    )
    for python, cpp in zip(python_results, cpp_results, strict=True):
        assert python.GetStatus() == oevibe.ConformerStatus.OK, python.GetMessage()
        assert python.GetEnergy() == pytest.approx(cpp.GetEnergy(), abs=1e-9)


def test_gradient_matches_the_cpp_engine(butanol_conformers):
    python = oevibe.Calculation(options(ENGINE_NAME)).Gradient(
        oechem.OEMol(butanol_conformers)
    )
    cpp = oevibe.Calculation(options("oeff")).Gradient(oechem.OEMol(butanol_conformers))
    assert np.allclose(
        np.array(list(python.Get(0).GetGradient())),
        np.array(list(cpp.Get(0).GetGradient())),
        atol=1e-9,
    )


def test_optimization_reproduces_the_cpp_engine(butanol_conformers):
    """The spec 8.7 acceptance check for the director path, units, and index mapping."""
    python = oevibe.Calculation(options(ENGINE_NAME)).Optimize(
        oechem.OEMol(butanol_conformers), oevibe.OptimizerOptions()
    )
    cpp = oevibe.Calculation(options("oeff")).Optimize(
        oechem.OEMol(butanol_conformers), oevibe.OptimizerOptions()
    )
    for from_python, from_cpp in zip(python, cpp, strict=True):
        assert from_python.GetStatus() == oevibe.ConformerStatus.OK, (
            from_python.GetMessage()
        )
        assert from_cpp.GetStatus() == oevibe.ConformerStatus.OK, from_cpp.GetMessage()
        assert from_python.GetEnergy() == pytest.approx(from_cpp.GetEnergy(), abs=1e-6)
        assert np.allclose(coordinates(from_python), coordinates(from_cpp), atol=1e-5)
        assert (
            from_python.GetOptimizationStats().iterations
            == from_cpp.GetOptimizationStats().iterations
        )


def test_parity_holds_for_a_molecule_with_index_gaps():
    """Spec 8.7 names index mapping as one of the things this engine proves.

    A molecule with a deleted atom has an OpenEye layout with a gap, while the snapshot the
    engine sees is dense. Agreement with the C++ engine means both sides mapped it the same way.
    """

    def make_gapped():
        """A fresh molecule whose OpenEye indices have a hole.

        Built from scratch each time rather than copied: ``oechem.OEMol(mol)`` renumbers the
        atoms and compacts the gap away, which would leave this test proving nothing. OMEGA is
        deterministic for the same SMILES and options, so the two molecules are identical.
        """
        fresh = oechem.OEMol()
        assert oechem.OESmilesToMol(fresh, "CCOC")
        omega_options = oeomega.OEOmegaOptions()
        omega_options.SetMaxConfs(1)
        assert (
            oeomega.OEOmega(omega_options).Build(fresh)
            == oeomega.OEOmegaReturnCode_Success
        )
        # A helium added and deleted leaves one unused atom index behind.
        fresh.DeleteAtom(fresh.NewAtom(2))
        assert fresh.GetMaxAtomIdx() > fresh.NumAtoms(), "the gap was compacted away"
        return fresh

    python_result = (
        oevibe.Calculation(options(ENGINE_NAME)).Energy(make_gapped()).Get(0)
    )
    cpp_result = oevibe.Calculation(options("oeff")).Energy(make_gapped()).Get(0)
    assert python_result.GetStatus() == oevibe.ConformerStatus.OK, (
        python_result.GetMessage()
    )
    assert cpp_result.GetStatus() == oevibe.ConformerStatus.OK, cpp_result.GetMessage()
    assert python_result.GetEnergy() == pytest.approx(cpp_result.GetEnergy(), abs=1e-9)


def test_gradient_parity_holds_when_the_first_atom_index_is_unused():
    """The gap above trails the real atoms, so dense and OpenEye indices still agree there.

    Here the gap comes first, which shifts every dense index by one against the caller's layout:
    the engine's coordinates and the OpenEye-layout gradient it comes back as must both follow.
    """

    def make_leading_gap():
        source = oechem.OEMol()
        assert oechem.OESmilesToMol(source, "CCOC")
        omega_options = oeomega.OEOmegaOptions()
        omega_options.SetMaxConfs(1)
        assert (
            oeomega.OEOmega(omega_options).Build(source)
            == oeomega.OEOmegaReturnCode_Success
        )
        # Adding the source after a placeholder puts its atoms at indices 1..N; deleting the
        # placeholder then leaves index 0 unused.
        fresh = oechem.OEMol()
        placeholder = fresh.NewAtom(2)
        oechem.OEAddMols(fresh, source)
        fresh.DeleteAtom(placeholder)
        assert min(atom.GetIdx() for atom in fresh.GetAtoms()) == 1, (
            "the gap was compacted away"
        )
        return fresh

    gapped = make_leading_gap()
    python_result = oevibe.Calculation(options(ENGINE_NAME)).Gradient(gapped).Get(0)
    cpp_result = oevibe.Calculation(options("oeff")).Gradient(make_leading_gap()).Get(0)
    assert python_result.GetStatus() == oevibe.ConformerStatus.OK, (
        python_result.GetMessage()
    )
    assert cpp_result.GetStatus() == oevibe.ConformerStatus.OK, cpp_result.GetMessage()
    python_gradient = np.array(list(python_result.GetGradient()))
    cpp_gradient = np.array(list(cpp_result.GetGradient()))
    # OpenEye layout: three entries for every index up to GetMaxAtomIdx, the unused first one zero.
    assert len(python_gradient) == 3 * gapped.GetMaxAtomIdx()
    assert not python_gradient[:3].any()
    assert np.abs(python_gradient[3:]).max() > 1.0, (
        "a near-zero gradient would prove nothing"
    )
    assert np.allclose(python_gradient, cpp_gradient, atol=1e-9)


def test_frequencies_through_the_python_engine(butanol_conformers):
    """Finite-difference Hessians drive hundreds of director callbacks in one session."""
    thermo = oevibe.ThermoOptions()
    results = oevibe.Calculation(options(ENGINE_NAME)).Frequencies(
        oechem.OEMol(butanol_conformers), thermo
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.OK, results.Get(
        0
    ).GetMessage()
    frequencies = list(results.Get(0).GetThermochemistry().frequencies)
    assert len(frequencies) == 3 * butanol_conformers.NumAtoms() - 6
    # The C++ engine has an analytic Hessian where this one is differenced from gradients, so the
    # two agree only to finite-difference accuracy (measured: 0.2 cm^-1 at worst). A units or
    # layout error moves modes by far more than the 1 cm^-1 allowed here.
    cpp = oevibe.Calculation(options("oeff")).Frequencies(
        oechem.OEMol(butanol_conformers), thermo
    )
    assert cpp.Get(0).GetStatus() == oevibe.ConformerStatus.OK, cpp.Get(0).GetMessage()
    assert np.allclose(
        frequencies, list(cpp.Get(0).GetThermochemistry().frequencies), atol=1.0
    )
