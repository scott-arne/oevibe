"""One-call verbs over :class:`Calculation` (spec section 5.1)."""

from __future__ import annotations

import numbers
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


def _checked_float(name: str, value: Any) -> float:
    # SWIG raises TypeError for a value it cannot convert to a double; the verbs report every
    # invalid option as OptionsError, and the C++ setters check the range.
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise _core.OptionsError(f"{name} must be a number, got {value!r}") from None


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


def frequencies(
    mol: Any,
    method: Any = "gfn2-xtb",
    *,
    optimize: bool = True,
    temperature: float = 298.15,
    pressure: float = 1.0,
    solvent: Any = None,
    engine: str | None = None,
    preset: Any = "standard",
    properties: Iterable[Any] = (),
    charge: int | None = None,
    multiplicity: int | None = None,
    keywords: Mapping[str, Any] | None = None,
) -> Any:
    """Frequencies and thermochemistry for every conformer of ``mol``.

    Shared parameters match :func:`energy`. Each conformer is optimized first (unless
    ``optimize`` is false), then its Hessian is computed and analyzed; the optimized coordinates
    are written back. Building the options and the calculation holds the GIL; only the
    calculation releases it, after copying the thermo options, so do not pass the same molecule
    to calculations on more than one thread at a time.

    :param optimize: Optimize before the Hessian; with ``False`` the input geometry is analyzed
        and the molecule's coordinates are left unchanged.
    :param temperature: Temperature in K for the thermochemistry.
    :param pressure: Pressure in atm for the translational entropy.
    :returns: A :class:`ResultSet`; ``GetBoltzmannWeights`` then weights by free energy.
    :raises OptionsError: For invalid options, including a temperature or pressure that is not
        a finite number greater than 0.
    """
    options = _calculation_options(
        method, solvent, engine, preset, properties, charge, multiplicity, keywords
    )
    thermo = _core.ThermoOptions()
    thermo.SetOptimizeFirst(bool(optimize))
    thermo.SetTemperature(_checked_float("temperature", temperature))
    thermo.SetPressure(_checked_float("pressure", pressure))
    return _core.Calculation(options).Frequencies(mol, thermo)


def _dihedral_indices(dihedral: Any, mol: Any) -> list[int]:
    from openeye import oechem

    if isinstance(dihedral, str | bytes):
        raise _core.OptionsError(
            f"each dihedral must be four atoms or atom indices, got {dihedral!r}"
        )
    try:
        items = list(dihedral)
    except TypeError:
        raise _core.OptionsError(
            f"each dihedral must be four atoms or atom indices, got {dihedral!r}"
        ) from None
    if len(items) != 4:
        raise _core.OptionsError(
            f"each dihedral must have exactly four atoms, got {len(items)}"
        )
    indices = []
    for item in items:
        if isinstance(item, oechem.OEAtomBase):
            # A handle reduces to an index local to its molecule, so one from another molecule
            # would silently select whichever atom here has that index.
            own = mol.GetAtom(oechem.OEHasAtomIdx(item.GetIdx()))
            if own is None or own != item:
                raise _core.InputError(
                    f"dihedral atom {item.GetIdx()} does not belong to the scanned molecule"
                )
            indices.append(int(item.GetIdx()))
        elif isinstance(item, numbers.Integral) and not isinstance(item, bool):
            indices.append(_checked_int("dihedral atom index", item, 0, _UNSIGNED_MAX))
        else:
            raise _core.OptionsError(
                f"dihedral atoms must be OEAtomBase objects or atom indices, got {item!r}"
            )
    return indices


def _angle_range(value: Any) -> tuple[float, float]:
    try:
        start, end = value
    except (TypeError, ValueError):
        raise _core.OptionsError(
            f"angle_range must be a (start, end) pair in degrees, got {value!r}"
        ) from None
    return (
        _checked_float("angle_range start", start),
        _checked_float("angle_range end", end),
    )


def _optimization_cap(value: Any) -> int:
    # _checked_int would truncate 1.5 and True to 1 and let a string or a non-finite float raise
    # ValueError or OverflowError, so the cap must already be an integer here.
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise _core.OptionsError(f"max_optimizations must be an integer, got {value!r}")
    return _checked_int("max_optimizations", value, 1, _UNSIGNED_MAX)


def scan(
    mol: Any,
    dihedrals: Iterable[Any],
    method: Any = "gfn2-xtb",
    *,
    step: float = 15.0,
    angle_range: tuple[float, float] = (-180.0, 180.0),
    max_optimizations: int | None = None,
    solvent: Any = None,
    engine: str | None = None,
    preset: Any = "standard",
    properties: Iterable[Any] = (),
    charge: int | None = None,
    multiplicity: int | None = None,
    keywords: Mapping[str, Any] | None = None,
) -> Any:
    """Relaxed scan of one or two dihedrals by wavefront propagation.

    Every conformer of ``mol`` seeds the grid point nearest its dihedral values, and each
    grid-point optimization holds the scanned dihedrals at the grid values. The molecule's
    coordinates and SD data are not changed; implicit hydrogens are made explicit, as for every
    calculation. Shared parameters match :func:`energy`; a scan computes no properties, so a
    non-empty ``properties`` raises :class:`OptionsError`. Building the options and the
    calculation holds the GIL; only the scan releases it, after copying the scan options, so do
    not pass the same molecule to calculations on more than one thread at a time.

    :param dihedrals: One or two dihedrals, each four ``OEAtomBase`` objects or four OpenEye atom
        indices. Atom objects must belong to ``mol`` itself (or one of its conformers); atoms of
        a copy are another molecule's, so pass indices for those.
    :param step: Grid step in degrees; it must divide the range.
    :param angle_range: ``(start, end)`` in degrees, at most 360 apart. A range of exactly 360
        degrees gives a periodic grid from ``start + step`` to ``end``; any other includes both
        ends.
    :param max_optimizations: Caps the constrained optimizations; by default 20 per grid point.
    :returns: A :class:`ScanResult`.
    :raises OptionsError: For invalid dihedrals, scan options, or calculation options.
    :raises InputError: When the molecule fails validation, a dihedral atom index is not in it,
        or a dihedral atom object belongs to another molecule.
    """
    options = _calculation_options(
        method, solvent, engine, preset, properties, charge, multiplicity, keywords
    )
    if isinstance(dihedrals, str | bytes):
        raise _core.OptionsError(
            f"dihedrals must be a list of dihedrals, got {dihedrals!r}"
        )
    try:
        dihedral_list = list(dihedrals)
    except TypeError:
        raise _core.OptionsError(
            f"dihedrals must be a list of dihedrals, got {dihedrals!r}"
        ) from None
    scan_options = _core.ScanOptions()
    for dihedral in dihedral_list:
        scan_options.AddDihedral(*_dihedral_indices(dihedral, mol))
    scan_options.SetStep(_checked_float("step", step))
    scan_options.SetRange(*_angle_range(angle_range))
    if max_optimizations is not None:
        scan_options.SetMaxOptimizations(_optimization_cap(max_optimizations))
    return _core.Calculation(options).Scan(mol, scan_options)
