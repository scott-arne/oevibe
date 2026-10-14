"""Bridges between OEVibe and OpenEye's potential and optimizer interfaces (spec 8.8)."""

from __future__ import annotations

import importlib.util

from .oevibe import OptionsError, _MolFuncEngine


def _is_openeye_type(kind) -> bool:
    """:returns: Whether ``kind`` is defined in one of OpenEye's binding modules.

    OpenEye's SWIG proxies report a bare module name such as ``oeff``, which is importable as
    ``openeye.oeff``; a Python-defined subclass reports its own module, which is not.
    """
    try:
        return importlib.util.find_spec(f"openeye.{kind.__module__}") is not None
    except (ImportError, ValueError):
        return False


def _as_molfunc1(potential):
    """Upcast an OpenEye potential to its ``OEMolFunc1`` subobject.

    ``OEMolFunc1`` is a virtual base of every OpenEye potential, so its subobject is not at the
    address an OpenEye SWIG proxy stores (measured: 48 bytes earlier for ``OEMMFF``). Adding the
    potential to an ``OEGenericFF`` and reading it back makes OpenEye's own runtime perform the
    upcast; the component it returns does not own the potential.

    :param potential: An OpenEye-defined ``oeff.OEMolFunc1``.
    :returns: ``(component, holder)``: a proxy of exact type ``oeff.OEMolFunc1`` pointing at
        ``potential``, and the ``OEGenericFF`` it was read from, which the caller keeps alive
        while it uses the component.
    :raises TypeError: When ``potential`` is not an OpenEye ``OEMolFunc1``, or is a Python-defined
        subclass of one, which spec 8.8 does not support in v1.
    :raises OptionsError: When OpenEye refuses to hold the potential.
    """
    from openeye import oeff

    if not isinstance(potential, oeff.OEMolFunc1):
        raise TypeError(
            f"expected an openeye.oeff.OEMolFunc1, got {type(potential).__name__}"
        )
    if not _is_openeye_type(type(potential)):
        # Spec 8.8: Python-defined OEMolFunc1 subclasses are not supported in v1. Their C++ side
        # is an OpenEye director whose every call would re-enter Python under OEVibe's own
        # per-engine locking, with no lifetime contract between the two runtimes.
        raise TypeError(
            f"{type(potential).__name__} is a Python-defined OEMolFunc1 subclass, which "
            "MolFuncEngine does not support in v1; write an oevibe.PythonEngine instead"
        )
    holder = oeff.OEGenericFF()
    if not holder.Add(potential):
        raise OptionsError(
            "OpenEye rejected this potential; it cannot back a MolFuncEngine"
        )
    components = holder.GetComponents()
    if len(components) != 1:
        raise OptionsError(
            f"OpenEye reported {len(components)} components for one potential; expected 1"
        )
    return components[0], holder


def MolFuncEngine(name, version, method, potential, requires_partial_charges=False):
    """One OpenEye potential as a single-method OEVibe engine (spec 8.8).

    The potential is copied with ``CreateCopy()`` at construction, so the caller keeps ownership
    of the argument and may drop it afterwards. The engine is not registered automatically; pass
    it to :func:`oevibe.register_engine`.

    :param name: Registry name; must not be a built-in engine name.
    :param version: Reported engine version.
    :param method: The one ``oevibe.Method`` this engine declares.
    :param potential: Any ``openeye.oeff.OEMolFunc1``, for example ``oeff.OEMMFF(True)``.
    :param requires_partial_charges: Declared in the method's capabilities.
    :returns: An ``oevibe.Engine``.
    :raises TypeError: When ``potential`` is not an OpenEye ``OEMolFunc1``.
    :raises OptionsError: When the potential cannot be copied or OpenEye refuses to hold it.
    """
    upcast, holder = _as_molfunc1(potential)
    try:
        return _MolFuncEngine(name, version, method, upcast, requires_partial_charges)
    finally:
        # The holder only borrows the potential, but it must outlive the constructor's read.
        del holder
