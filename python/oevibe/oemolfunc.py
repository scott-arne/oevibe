"""Bridges between OEVibe and OpenEye's potential and optimizer interfaces (spec 8.8)."""

from __future__ import annotations

from .oevibe import OptionsError, _MolFuncEngine

_OPENEYE_POTENTIAL_CLASSES: frozenset[type] | None = None


def _openeye_potential_classes() -> frozenset[type]:
    """:returns: Every class ``openeye.oeff`` exports that derives from ``OEMolFunc1``.

    Captured once, on the first potential OEVibe is asked about, and reused from then on. The
    import stays inside the function because this module never imports ``openeye`` at ``oevibe``
    import time.

    Snapshotting ``oeff`` alone is complete: all 69 ``OEMolFunc1`` subclasses live there, and the
    other OpenEye modules that export an ``OEMolFunc1`` (``oeet``, ``oequacpac``, ``oeshape``)
    export a *distinct* class whose instances fail the ``isinstance`` gate in :func:`_as_molfunc1`
    before the snapshot is consulted. ``test_the_potential_snapshot_covers_every_oeff_potential``
    and ``test_no_other_openeye_module_exports_an_oeff_potential`` fail if either stops holding.
    """
    global _OPENEYE_POTENTIAL_CLASSES
    if _OPENEYE_POTENTIAL_CLASSES is None:
        from openeye import oeff

        # Attribute access only; constructing all 69 is far too expensive to do here, and the
        # enumeration does not need to know which of them are abstract.
        _OPENEYE_POTENTIAL_CLASSES = frozenset(
            candidate
            for candidate in (getattr(oeff, name) for name in dir(oeff))
            if isinstance(candidate, type) and issubclass(candidate, oeff.OEMolFunc1)
        )
    return _OPENEYE_POTENTIAL_CLASSES


def _is_openeye_type(kind) -> bool:
    """:returns: Whether ``kind`` is one of the potential classes OpenEye itself defines.

    Membership in the snapshot, not a live read of ``openeye.oeff``. A module is mutable, so
    asking it what it exports *now* answered yes to any class assigned into it:
    ``oeff.Reexported = Reexported`` made a Python-defined subclass pass, and ``MolFuncEngine``
    then ran native MMFF while the caller believed the subclass was in play. Comparing against a
    set captured before any of that happened closes it, and a frozenset lookup is also cheaper
    than the ``import_module`` plus ``getattr`` it replaces -- a nice property, not the reason.

    What this does NOT achieve: a caller who rebinds ``openeye.oeff``, or patches this module's
    own state, *before* OEVibe's first potential check still defeats it, and so does one who
    simply replaces this function. No predicate written in Python survives a caller willing to
    mutate module state. This is a guardrail against unsupported usage that would otherwise
    return a silently wrong answer, not a security boundary.
    """
    return kind in _openeye_potential_classes()


# Every class in ``openeye.oeff`` that derives from ``OEMolFunc1`` and refuses construction:
# the director interfaces, which raise "accessing abstract class or protected constructor", and
# the plain abstract bases, whose SWIG proxy raises "No constructor defined - class is abstract".
# Regenerate by calling each ``oeff`` class that is a subclass of ``OEMolFunc1``.
#
# No object's dynamic type can be one of these, so a proxy whose EXACT type is one of them is
# always a view of something derived. SWIG builds a proxy from the STATIC type of the pointer it
# is handed, so OpenEye's own accessors produce exactly that: ``OEGenericFF.GetComponents()``
# returns ``OEMolFunc1`` and ``OEGenericFF2.GetComponents()`` returns ``OEMolFunc2``, whatever
# was added to the holder. Nothing visible from Python recovers what is underneath.
_ABSTRACT_POTENTIAL_INTERFACES = (
    "OEFFPotential",
    "OEForceField",
    "OEInterMolFunc1",
    "OEInterMolFunc2",
    "OEInterNonBondPotBase",
    "OEMolAdaptor",
    "OEMolAdaptor2",
    "OEMolFunc1",
    "OEMolFunc2",
    "OENonBondPotBase",
)


def _is_abstract_potential_interface(kind) -> bool:
    """:returns: Whether ``kind`` is one of OpenEye's abstract potential interfaces.

    Identity again, and against a name list rather than a single name, because an upcast can
    land on any of them. ``getattr`` tolerates an OpenEye release that does not export one.
    """
    from openeye import oeff

    return any(
        kind is getattr(oeff, name, None) for name in _ABSTRACT_POTENTIAL_INTERFACES
    )


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
    :raises TypeError: When ``potential`` is not an OpenEye ``OEMolFunc1``; is a Python-defined
        subclass of one, which spec 8.8 does not support in v1; or is itself an upcast proxy,
        whose provenance OEVibe cannot establish.
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
    if _is_abstract_potential_interface(type(potential)):
        # Deliberately not phrased as "this is Python-defined": it may well not be. What OEVibe
        # knows is only that the proxy's exact type is an interface no object can be, so the
        # potential underneath is unidentifiable -- including as one spec 8.8 excludes.
        raise TypeError(
            f"this potential is an upcast proxy of exact type oeff.{type(potential).__name__}, "
            "an abstract interface; OEVibe cannot establish which potential is underneath an "
            "upcast and will not guess. Pass the original potential to MolFuncEngine instead."
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
    :raises TypeError: When ``potential`` is not an OpenEye ``OEMolFunc1``, is a Python-defined
        subclass of one, or is an upcast proxy such as an ``OEGenericFF`` component.
    :raises OptionsError: When the potential cannot be copied or OpenEye refuses to hold it.
    """
    # ``upcast`` points into the OEGenericFF that ``_holder`` owns, so that holder has to still
    # be alive when the constructor reads the potential. The local binding is what keeps it
    # alive, and it lasts until this function returns; nothing further is needed, and a
    # ``del`` at the end of the frame would do nothing at all.
    upcast, _holder = _as_molfunc1(potential)
    return _MolFuncEngine(name, version, method, upcast, requires_partial_charges)
