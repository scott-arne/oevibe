"""Every invalid verb option is an OptionsError, and value types are hashable."""

from fractions import Fraction

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


def test_a_property_bit_set_is_told_to_pass_each_property_separately():
    # MethodCapabilities.properties and EngineSettings.properties are documented bit sets, so a
    # caller can hold 7 without ever having built it, and has to be told how to pass it on.
    mask = (
        int(oevibe.Property.CHARGES)
        | int(oevibe.Property.DIPOLE)
        | int(oevibe.Property.ORBITAL_ENERGIES)
    )
    assert mask == 7
    with pytest.raises(oevibe.OptionsError) as raised:
        _verbs._properties([mask])
    message = str(raised.value)
    assert (
        "each passed as its own value rather than combined into one bit set" in message
    )
    # The remedy is added to the existing list, not substituted for it.
    assert "unknown property 7; expected one of ['bond_orders'" in message


@pytest.mark.parametrize("value", ["quick", None, 99])
def test_an_invalid_preset_lists_the_presets_whatever_its_type(value):
    with pytest.raises(oevibe.OptionsError) as raised:
        _verbs._preset(value)
    assert "fast, standard, or thorough" in str(raised.value)


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


@pytest.mark.parametrize(
    "value",
    [
        # Spelled as ids because pytest builds a parameter id by str()ing the value, and
        # Python refuses to render a 10001-digit integer.
        pytest.param(10**10000, id="huge-int"),
        pytest.param(-(10**10000), id="huge-negative-int"),
        pytest.param(Fraction(10**10000, 3), id="huge-fraction"),
    ],
)
def test_a_real_too_large_for_a_double_is_an_options_error(value):
    # numbers.Real admits values no double can hold -- a big int, a big Fraction -- and float()
    # raises OverflowError rather than TypeError on those, so the guard above is not enough.
    with pytest.raises(oevibe.OptionsError):
        _verbs._checked_float("temperature", value)


def test_a_verb_reports_an_unrepresentable_temperature_as_an_options_error(
    butanol_conformers,
):
    with pytest.raises(oevibe.OptionsError):
        oevibe.frequencies(butanol_conformers, method="mmff94", temperature=10**10000)


def test_a_hashed_method_key_survives_its_options_changing():
    # GetMethod hands back a copy, so a Method already used as a key keeps the value it was
    # hashed under. A borrowed view into the options' storage would rehash when the options
    # were given another method, and the set entry would become unreachable.
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94s"))
    key = options.GetMethod()
    keys = {key}
    before = hash(key)
    options.SetMethod(oevibe.Method("gfn2-xtb"))

    assert key.GetName() == "mmff94s"
    assert hash(key) == before
    assert key in keys


def test_a_hashed_solvent_key_survives_its_options_changing():
    options = oevibe.CalculationOptions()
    options.SetSolvent(oevibe.Solvent("water", oevibe.SolventModel.ALPB))
    key = options.GetSolvent()
    keys = {key}
    before = hash(key)
    options.SetSolvent(oevibe.Solvent("octanol", oevibe.SolventModel.GBSA))

    assert key.GetName() == "water"
    assert key.GetModel() == oevibe.SolventModel.ALPB
    assert hash(key) == before
    assert key in keys


class _RenamedMethod(oevibe.Method):
    """Reports a name other than the one it stores, which is what ``__eq__`` compares."""

    def GetName(self):
        return "renamed-" + super().GetName()


class _RenamedSolvent(oevibe.Solvent):
    """Reports a name other than the one it stores, which is what ``__eq__`` compares."""

    def GetName(self):
        return "renamed-" + super().GetName()


def test_a_method_hashes_on_what_equality_compares():
    # __eq__ is C++ operator==, which reads the stored name. A __hash__ routed through the
    # overridable GetName() would disagree with it for a subclass that renames itself, so two
    # objects that compare equal would fail to find each other in a set.
    base = oevibe.Method("mmff94s")
    renamed = _RenamedMethod("mmff94s")

    assert renamed == base
    assert hash(renamed) == hash(base)
    assert base in {renamed}
    assert renamed in {base}


def test_a_solvent_hashes_on_what_equality_compares():
    base = oevibe.Solvent("water", oevibe.SolventModel.ALPB)
    renamed = _RenamedSolvent("water", oevibe.SolventModel.ALPB)

    assert renamed == base
    assert hash(renamed) == hash(base)
    assert base in {renamed}
    assert renamed in {base}
