"""The frequencies verb with the OpenEye force-field engine."""

import gc
import math

import pytest
from openeye import oechem

import oevibe

_GAS_CONSTANT_KCAL = 1.98720425864083e-3  # kcal/(mol K), spec 3.1
_OK = oevibe.ConformerStatus.OK


def _boltzmann(values, temperature):
    lowest = min(values.values())
    rt = _GAS_CONSTANT_KCAL * temperature
    factors = {key: math.exp(-(value - lowest) / rt) for key, value in values.items()}
    total = sum(factors.values())
    return {key: factor / total for key, factor in factors.items()}


def _free_energies(results):
    return {r.GetConformerIndex(): r.GetThermochemistry().free_energy for r in results}


def test_frequencies_attach_modes_and_thermochemistry(butanol_conformers):
    results = oevibe.frequencies(butanol_conformers, method="mmff94")
    num_atoms = butanol_conformers.NumAtoms()
    assert len(results) == butanol_conformers.NumConfs()
    for result in results:
        assert result.GetStatus() == _OK, result.GetMessage()
        frequencies = result.GetFrequencies()
        assert isinstance(frequencies, tuple)
        assert len(frequencies) == 3 * num_atoms - 6
        assert len(result.GetNormalModes()) == len(frequencies) * 3 * num_atoms
        assert len(result.GetHessian()) == (3 * num_atoms) ** 2
        thermo = result.GetThermochemistry()
        assert thermo.temperature == 298.15
        assert thermo.pressure == 1.0
        assert thermo.frequencies == frequencies
        assert len(thermo.principal_moments) == 3
        assert thermo.electronic_energy == result.GetEnergy()
        assert thermo.zpe > 0.0
        assert thermo.free_energy_correction == pytest.approx(
            thermo.free_energy - thermo.electronic_energy
        )
    resolved = results[0].GetProvenance().GetResolvedOptions()
    assert resolved["task"] == "frequencies"
    assert resolved["hessian"] == "analytic"


def test_boltzmann_weights_use_free_energies_at_each_temperature(butanol_conformers):
    results = oevibe.frequencies(butanol_conformers, method="mmff94")
    assert all(r.GetStatus() == _OK for r in results)
    weights = dict(results.GetBoltzmannWeights(298.15))
    assert weights == pytest.approx(_boltzmann(_free_energies(results), 298.15))

    # The written-back geometries analyzed directly at 600 K give each conformer's free energy
    # there, which the stored thermochemistry must reproduce when recomputed at 600 K.
    hot = oevibe.frequencies(
        butanol_conformers, method="mmff94", optimize=False, temperature=600.0
    )
    assert all(r.GetThermochemistry().temperature == 600.0 for r in hot)
    expected = _boltzmann(_free_energies(hot), 600.0)
    assert dict(results.GetBoltzmannWeights(600.0)) == pytest.approx(expected, abs=1e-4)
    assert sum(results.GetBoltzmannWeights(600.0).values()) == pytest.approx(1.0)


def test_results_are_read_only(butanol_conformers):
    results = oevibe.frequencies(butanol_conformers, method="mmff94")
    thermo = results[0].GetThermochemistry()
    with pytest.raises(AttributeError):
        thermo.zpe = 0.0
    with pytest.raises(AttributeError):
        thermo.frequencies = (1.0,)
    # Vector members come back as tuples, not as views that could edit the result.
    assert isinstance(thermo.frequencies, tuple)
    assert isinstance(thermo.principal_moments, tuple)
    for name in ("SetHessian", "SetFrequencies", "SetThermochemistry"):
        assert not hasattr(results[0], name)
    zpe = thermo.zpe
    del results
    gc.collect()
    # The view keeps its result set alive.
    assert thermo.zpe == zpe


def test_tags_are_written(butanol_conformers):
    results = oevibe.frequencies(butanol_conformers, method="mmff94")
    for conf, result in zip(butanol_conformers.GetConfs(), results, strict=True):
        thermo = result.GetThermochemistry()
        for quantity, value in (
            ("ZPE", thermo.zpe),
            ("Enthalpy", thermo.enthalpy),
            ("FreeEnergy", thermo.free_energy),
        ):
            tag = f"OEVibe.mmff94.gas.{quantity}"
            assert oechem.OEHasSDData(conf, tag)
            # SD tags carry six decimals.
            assert float(oechem.OEGetSDData(conf, tag)) == pytest.approx(
                value, abs=1e-6
            )


def test_optimize_false_leaves_the_coordinates(butanol_conformers):
    before = {conf.GetIdx(): conf.GetCoords() for conf in butanol_conformers.GetConfs()}
    results = oevibe.frequencies(
        butanol_conformers, method="mmff94", optimize=False, pressure=2.0
    )
    for conf in butanol_conformers.GetConfs():
        assert conf.GetCoords() == before[conf.GetIdx()]
    assert all(r.GetThermochemistry().pressure == 2.0 for r in results)
    assert not results[0].HasCoordinates()
    resolved = results[0].GetProvenance().GetResolvedOptions()
    assert resolved["thermo.optimize_first"] == "false"


def test_invalid_options_are_option_errors(butanol_conformers):
    with pytest.raises(oevibe.OptionsError, match="(?i)temperature"):
        oevibe.frequencies(butanol_conformers, method="mmff94", temperature=-1.0)
    with pytest.raises(oevibe.OptionsError, match="(?i)temperature"):
        oevibe.frequencies(butanol_conformers, method="mmff94", temperature="hot")
    with pytest.raises(oevibe.OptionsError, match="(?i)pressure"):
        oevibe.frequencies(butanol_conformers, method="mmff94", pressure=float("nan"))
    with pytest.raises(oevibe.OptionsError, match="multiplicity"):
        oevibe.frequencies(butanol_conformers, method="mmff94", multiplicity=-1)


def test_thermo_options_reject_constraints(butanol_conformers):
    constraints = oevibe.ConstraintSet()
    constraints.AddDistance(0, 1)
    optimizer = oevibe.OptimizerOptions()
    optimizer.SetConstraints(constraints)
    thermo = oevibe.ThermoOptions()
    thermo.SetOptimizerOptions(optimizer)
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94"))
    with pytest.raises(oevibe.OptionsError):
        oevibe.Calculation(options).Frequencies(butanol_conformers, thermo)
    assert thermo.GetOptimizerOptions().GetConstraints().HasExactConstraints()
