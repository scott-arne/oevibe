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
_INT_MIN = -(2**31)
_INT_MAX = 2**31 - 1
_UNSIGNED_MAX = 2**32 - 1


def _checked_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    # SWIG rejects integers outside the C parameter's range with OverflowError; the verbs
    # report every invalid option as OptionsError instead.
    number = int(value)
    if number < minimum:
        raise _core.OptionsError(f"{name} must be at least {minimum}, got {number}")
    if number > maximum:
        raise _core.OptionsError(f"{name} must be at most {maximum}, got {number}")
    return number


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
        options.SetCharge(_checked_int("charge", charge, _INT_MIN, _INT_MAX))
    if multiplicity is not None:
        options.SetMultiplicity(
            _checked_int("multiplicity", multiplicity, 1, _UNSIGNED_MAX)
        )
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

    Building the options and the calculation holds the GIL; only the energy evaluation releases
    it, so do not pass the same molecule to calculations on more than one thread at a time.

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
    Building the options and the calculation holds the GIL; only the gradient evaluation
    releases it, so do not pass the same molecule to calculations on more than one thread at a
    time.

    :returns: A :class:`ResultSet` with one result per conformer.
    """
    options = _calculation_options(
        method, solvent, engine, preset, properties, charge, multiplicity, keywords
    )
    return _core.Calculation(options).Gradient(mol)


_ALGORITHMS = {
    "auto": _core.Algorithm_AUTO,
    "anc_rfo": _core.Algorithm_ANC_RFO,
    "lbfgs": _core.Algorithm_LBFGS,
}


def _algorithm(value: Any) -> int:
    if isinstance(value, str):
        try:
            return _ALGORITHMS[value.lower()]
        except KeyError:
            raise _core.OptionsError(
                f"unknown algorithm {value!r}; expected auto, anc_rfo, or lbfgs"
            ) from None
    return int(value)


def optimize(
    mol: Any,
    method: Any = "gfn2-xtb",
    *,
    constraints: Any = None,
    algorithm: Any = "auto",
    max_iterations: int | None = None,
    solvent: Any = None,
    engine: str | None = None,
    preset: Any = "standard",
    properties: Iterable[Any] = (),
    charge: int | None = None,
    multiplicity: int | None = None,
    keywords: Mapping[str, Any] | None = None,
) -> Any:
    """Optimize every conformer of ``mol`` and write the final coordinates back.

    Shared parameters match :func:`energy`. Building the options and the calculation holds the
    GIL; only the optimization releases it, after copying the optimizer options, so do not pass
    the same molecule to calculations on more than one thread at a time.

    :param constraints: A :class:`ConstraintSet` of frozen atoms, constraints, and restraints.
    :param algorithm: ``"auto"``, ``"anc_rfo"``, ``"lbfgs"``, or an :class:`Algorithm`.
    :param max_iterations: Overrides the preset's iteration limit.
    :returns: A :class:`ResultSet` with one result per conformer.
    """
    options = _calculation_options(
        method, solvent, engine, preset, properties, charge, multiplicity, keywords
    )
    optimizer = _core.OptimizerOptions()
    optimizer.SetAlgorithm(_algorithm(algorithm))
    if max_iterations is not None:
        optimizer.SetMaxIterations(
            _checked_int("max_iterations", max_iterations, 1, _UNSIGNED_MAX)
        )
    if constraints is not None:
        optimizer.SetConstraints(constraints)
    return _core.Calculation(options).Optimize(mol, optimizer)
