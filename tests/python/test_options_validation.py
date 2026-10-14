"""Every invalid verb option is an OptionsError, and value types are hashable."""

import pytest

import oevibe
from oevibe import _verbs


@pytest.mark.parametrize("value", [99, -1, 2.5, True, "quick", None])
def test_an_invalid_preset_is_an_options_error(value):
    with pytest.raises(oevibe.OptionsError):
        _verbs._preset(value)


@pytest.mark.parametrize("value", [99, -1, True, "fastest", None])
def test_an_invalid_algorithm_is_an_options_error(value):
    with pytest.raises(oevibe.OptionsError):
        _verbs._algorithm(value)


@pytest.mark.parametrize("value", [99, -1, True, "charge", None])
def test_an_invalid_property_is_an_options_error(value):
    with pytest.raises(oevibe.OptionsError):
        _verbs._properties([value])


def test_valid_presets_and_properties_still_resolve():
    assert _verbs._preset("Thorough") == _verbs._preset(oevibe.Preset.THOROUGH)
    assert _verbs._algorithm("ANC_RFO") == _verbs._algorithm(oevibe.Algorithm.ANC_RFO)
    mask = _verbs._properties(["charges", oevibe.Property.DIPOLE])
    assert mask == int(oevibe.Property.CHARGES) | int(oevibe.Property.DIPOLE)


@pytest.mark.parametrize("value", [True, False, "2", None])
def test_a_boolean_or_string_is_not_a_number(value):
    with pytest.raises(oevibe.OptionsError):
        _verbs._checked_float("temperature", value)


@pytest.mark.parametrize("value", [True, 1.7, "3", None])
def test_a_boolean_or_fractional_value_is_not_an_integer(value):
    with pytest.raises(oevibe.OptionsError):
        _verbs._checked_int("max_iterations", value, 1, 1000)


def test_verbs_report_an_invalid_preset_as_an_options_error(butanol_conformers):
    with pytest.raises(oevibe.OptionsError):
        oevibe.energy(butanol_conformers, method="mmff94s", preset=99)


def test_method_and_solvent_are_hashable():
    methods = {
        oevibe.Method("mmff94s"),
        oevibe.Method("mmff94s"),
        oevibe.Method("gfn2-xtb"),
    }
    assert len(methods) == 2
    solvents = {
        oevibe.Solvent("water"),
        oevibe.Solvent("water"),
        oevibe.Solvent("water", oevibe.SolventModel.ALPB),
        oevibe.Solvent(),
    }
    assert len(solvents) == 3
    assert hash(oevibe.Method("MMFF94S")) == hash(oevibe.Method("mmff94s"))
