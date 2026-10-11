"""One-call verbs over :class:`Calculation` (spec section 5.1)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from . import oevibe as _core

_PRESETS = {
    "fast": _core.Preset_FAST,
    "standard": _core.Preset_STANDARD,
    "thorough": _core.Preset_THOROUGH,
}
_PROPERTIES = {
    "charges": _core.Property_CHARGES,
    "dipole": _core.Property_DIPOLE,
    "orbital_energies": _core.Property_ORBITAL_ENERGIES,
    "bond_orders": _core.Property_BOND_ORDERS,
}


def _preset(value: Any) -> int:
    if isinstance(value, str):
        try:
            return _PRESETS[value.lower()]
        except KeyError:
            raise _core.OptionsError(
                f"unknown preset {value!r}; expected fast, standard, or thorough"
            ) from None
    return int(value)


def _solvent(value: Any) -> Any:
    if value is None:
        return _core.Solvent()
    if isinstance(value, str):
        return _core.Solvent(value)
    return value


def _properties(values: Iterable[Any]) -> int:
    mask = 0
    for value in values:
        if isinstance(value, str):
            try:
                mask |= _PROPERTIES[value.lower()]
            except KeyError:
                raise _core.OptionsError(
                    f"unknown property {value!r}; expected one of {sorted(_PROPERTIES)}"
                ) from None
        else:
            mask |= int(value)
    return mask


def _keyword_text(value: Any) -> str:
    # Engine keyword schemas parse "true"/"false"; Python's str(True) would be "True".
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _calculation_options(
    method: Any,
    solvent: Any,
    engine: str | None,
    preset: Any,
    properties: Iterable[Any],
    charge: int | None,
    multiplicity: int | None,
    keywords: Mapping[str, Any] | None,
) -> Any:
    options = _core.CalculationOptions()
    options.SetMethod(
        method if isinstance(method, _core.Method) else _core.Method(str(method))
    )
    options.SetSolvent(_solvent(solvent))
    if engine is not None:
        options.SetEngine(engine)
    options.SetPreset(_preset(preset))
    options.SetProperties(_properties(properties))
    if charge is not None:
        options.SetCharge(int(charge))
    if multiplicity is not None:
        options.SetMultiplicity(int(multiplicity))
    for name, value in (keywords or {}).items():
        options.SetKeyword(str(name), _keyword_text(value))
    return options


def energy(
    mol: Any,
    method: Any = "gfn2-xtb",
    *,
    solvent: Any = None,
    engine: str | None = None,
    preset: Any = "standard",
    properties: Iterable[Any] = (),
    charge: int | None = None,
    multiplicity: int | None = None,
    keywords: Mapping[str, Any] | None = None,
) -> Any:
    """Single-point energies for every conformer of ``mol``.

    The calculation releases the GIL while it runs, so do not pass the same molecule to
    calculations on more than one thread at a time.

    :param mol: An OpenEye molecule with 3D coordinates; SD tags are written to it.
    :param method: Method name or :class:`Method`, for example ``"mmff94"``.
    :param solvent: ``None`` for gas phase, a solvent name, or a :class:`Solvent`.
    :param engine: Engine name overriding the method's default routing.
    :param preset: ``"fast"``, ``"standard"``, ``"thorough"``, or a :class:`Preset`.
    :param properties: Property names or :class:`Property` values to compute.
    :param charge: Total charge override.
    :param multiplicity: Spin multiplicity override.
    :param keywords: Engine-specific keywords.
    :returns: A :class:`ResultSet` with one result per conformer.
    :raises OptionsError: For invalid or unsupported options.
    :raises EngineUnavailableError: When the engine or method cannot run.
    :raises InputError: When the molecule fails validation.
    """
    options = _calculation_options(
        method, solvent, engine, preset, properties, charge, multiplicity, keywords
    )
    return _core.Calculation(options).Energy(mol)


def gradient(
    mol: Any,
    method: Any = "gfn2-xtb",
    *,
    solvent: Any = None,
    engine: str | None = None,
    preset: Any = "standard",
    properties: Iterable[Any] = (),
    charge: int | None = None,
    multiplicity: int | None = None,
    keywords: Mapping[str, Any] | None = None,
) -> Any:
    """Energies and gradients for every conformer of ``mol``.

    Parameters match :func:`energy`. Gradients use the OpenEye atom layout in kcal/mol/Angstrom.
    The calculation releases the GIL while it runs, so do not pass the same molecule to
    calculations on more than one thread at a time.

    :returns: A :class:`ResultSet` with one result per conformer.
    """
    options = _calculation_options(
        method, solvent, engine, preset, properties, charge, multiplicity, keywords
    )
    return _core.Calculation(options).Gradient(mol)
