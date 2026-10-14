"""OpenEye potentials as OEVibe engines (spec 8.8)."""

import gc

import numpy as np
import pytest
from openeye import oechem, oeff

import oevibe

METHOD = "mmff94s-molfunc"


@pytest.fixture
def mmff_engine():
    engine = oevibe.MolFuncEngine(
        "oeff-molfunc", "1.0", oevibe.Method(METHOD), oeff.OEMMFF(True)
    )
    oevibe.register_engine(engine, replace=True)
    return engine


def molfunc_options():
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method(METHOD))
    return options


def oeff_options():
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94s"))
    return options


def test_the_engine_resolves_without_naming_it(butanol_conformers, mmff_engine):
    """Only one engine declares the method, so ResolveForMethod must find it (spec 5.3)."""
    calculation = oevibe.Calculation(molfunc_options())
    assert calculation.GetEngineName() == "oeff-molfunc"


def test_single_point_matches_the_builtin_oeff_engine(butanol_conformers, mmff_engine):
    through_molfunc = oevibe.Calculation(molfunc_options()).Energy(
        oechem.OEMol(butanol_conformers)
    )
    through_oeff = oevibe.Calculation(oeff_options()).Energy(
        oechem.OEMol(butanol_conformers)
    )
    for mine, theirs in zip(through_molfunc, through_oeff, strict=True):
        assert mine.GetStatus() == oevibe.ConformerStatus.OK, mine.GetMessage()
        assert mine.GetEnergy() == pytest.approx(theirs.GetEnergy(), abs=1e-9)


def test_optimization_matches_the_builtin_oeff_engine(butanol_conformers, mmff_engine):
    mine = oevibe.Calculation(molfunc_options()).Optimize(
        oechem.OEMol(butanol_conformers), oevibe.OptimizerOptions()
    )
    theirs = oevibe.Calculation(oeff_options()).Optimize(
        oechem.OEMol(butanol_conformers), oevibe.OptimizerOptions()
    )
    for from_molfunc, from_oeff in zip(mine, theirs, strict=True):
        assert from_molfunc.GetEnergy() == pytest.approx(
            from_oeff.GetEnergy(), abs=1e-6
        )


def test_the_engine_survives_the_caller_dropping_the_potential(butanol_conformers):
    """The C++ constructor takes a CreateCopy() of the prototype, so this must still work."""
    potential = oeff.OEMMFF(True)
    engine = oevibe.MolFuncEngine(
        "oeff-molfunc-dropped", "1.0", oevibe.Method("mmff94s-dropped"), potential
    )
    oevibe.register_engine(engine, replace=True)
    del potential
    gc.collect()
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94s-dropped"))
    results = oevibe.Calculation(options).Energy(oechem.OEMol(butanol_conformers))
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.OK


def test_the_declared_capabilities_follow_spec_8_8(mmff_engine):
    methods = list(mmff_engine.GetMethods())
    assert len(methods) == 1
    capabilities = methods[0]
    assert capabilities.method.GetName() == METHOD
    assert capabilities.analytic_gradient
    assert capabilities.analytic_hessian  # OEMMFF is an OEMolFunc2
    assert capabilities.charge_from_topology
    assert list(capabilities.solvent_models) == []
    assert capabilities.properties == 0


def test_a_non_potential_is_rejected():
    with pytest.raises(TypeError):
        oevibe.MolFuncEngine("bad", "1.0", oevibe.Method("nope"), object())


def test_a_python_defined_potential_is_rejected():
    """Spec 8.8 excludes Python-defined OEMolFunc1 subclasses from v1."""

    class MyPotential(oeff.OEMolFunc1):
        def __init__(self):
            oeff.OEMolFunc1.__init__(self)

        def CreateCopy(self):
            copy = MyPotential()
            copy.__disown__()
            return copy

        def Setup(self, mol):
            return True

        def NumVar(self):
            return 0

        def GetFComponents(self, coords):
            return None

        def __call__(self, *args):
            return 0.0

    with pytest.raises(TypeError, match="PythonEngine"):
        oevibe.MolFuncEngine("py", "1.0", oevibe.Method("nope"), MyPotential())


def test_the_raw_constructor_rejects_an_unadjusted_proxy():
    """The pointer inside an OEMMFF proxy is 48 bytes away from its OEMolFunc1 subobject."""
    from oevibe.oevibe import _MolFuncEngine

    with pytest.raises(TypeError):
        _MolFuncEngine("raw", "1.0", oevibe.Method("nope"), oeff.OEMMFF(True))


def test_frequencies_use_the_analytic_hessian(butanol_conformers, mmff_engine):
    results = oevibe.Calculation(molfunc_options()).Frequencies(
        oechem.OEMol(butanol_conformers), oevibe.ThermoOptions()
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.OK, results.Get(
        0
    ).GetMessage()
    frequencies = np.array(list(results.Get(0).GetThermochemistry().frequencies))
    assert len(frequencies) == 3 * butanol_conformers.NumAtoms() - 6
