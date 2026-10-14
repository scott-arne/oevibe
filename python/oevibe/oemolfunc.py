"""Bridges between OEVibe and OpenEye's potential and optimizer interfaces (spec 8.8).

Spec 8.8 accepts OpenEye-defined potentials only, so the predicates below have to tell an
OpenEye-defined class from anything else. What they achieve, and what they do not:

- Every route that does not deliberately mutate module state is refused: a Python-defined
  subclass, one that spoofs ``__module__`` or ``__name__``, one re-exported from ``openeye.oeff``
  after OEVibe has looked at a potential, and an upcast proxy whose exact type is an abstract
  interface, behind which nothing visible from Python identifies the potential.
- A caller who rebinds an ``openeye`` module *before* OEVibe first reads it, forges ``__hash__``
  and ``__eq__`` on a class to fake membership in a snapshot, or patches these functions
  outright, still gets through. Nothing written in Python can prevent that.
- Which is accepted. This is a guardrail against unsupported usage that would otherwise return a
  silently wrong answer -- native MMFF numbers while the caller believes their own subclass is in
  play -- not a security boundary.

Every check is also import-order independent. OpenEye gives each of its binding modules its own
proxy class for ``OEMolPotential::OEMolFunc1``, and the last module imported wins the shared SWIG
type registry, so ``oeff.OEGenericFF.GetComponents()`` returns ``oequacpac.OEMolFunc1`` proxies in
a process that imported the partial-charge toolkit. A class is therefore judged by the module it
claims as its own, never against one module fixed in advance.

That claimed name is also import-*form* dependent -- ``oeff`` or ``openeye.oeff``, see
:func:`_openeye_leaf_name` -- so it is normalised before it is used for anything.
"""

from __future__ import annotations

import importlib

from .oevibe import OptionsError, _MolFuncEngine

_OPENEYE_POTENTIAL_CLASSES: dict[str, frozenset[type]] = {}


def _openeye_leaf_name(module_name: str) -> str:
    """:returns: ``module_name`` without the ``openeye.`` package prefix.

    Which of the two forms a class reports in ``__module__`` is the caller's import statement,
    not a property of the class: ``openeye/__init__.py`` replaces itself with a lazy module that
    loads each submodule under its bare name, so ``from openeye import oeff`` yields classes
    reporting ``oeff``, while ``import openeye.oeff`` goes through the ordinary submodule
    machinery and yields ``openeye.oeff``. Both name the one module the process has. The leaf is
    the normal form everything below uses -- for the import, for the snapshot cache key and for
    the messages -- so that the caller's import style cannot change an answer or a diagnostic.
    """
    return module_name.removeprefix("openeye.")


def _openeye_module(kind):
    """:returns: The ``openeye`` binding module ``kind`` belongs to, or ``None``.

    OpenEye's SWIG proxies report a module name such as ``oeff`` or ``oequacpac``, which is
    importable as ``openeye.<leaf>``; a Python-defined class reports its own module, which is
    not. Resolving the name the class carries is what keeps the checks here independent of which
    OpenEye modules the caller happened to import, and in which order.
    """
    try:
        return importlib.import_module(f"openeye.{_openeye_leaf_name(kind.__module__)}")
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


def _openeye_potential_classes(module_name: str = "oeff") -> frozenset[type]:
    """:returns: Every class ``openeye.<module_name>`` exports that derives from its
        ``OEMolFunc1``; empty when the module wraps no potentials.

    Captured once per module, on the first class OEVibe is asked about from it, and reused from
    then on. The default is ``oeff``, where all 69 potentials a caller would name live; the other
    modules that wrap ``OEMolFunc1`` are snapshotted only if a proxy of theirs turns up. The
    imports stay inside the function because this module never imports ``openeye`` at ``oevibe``
    import time.

    ``module_name`` is normalised to its leaf before anything else, so the two import forms of
    one module share a single cache entry rather than snapshotting it twice.

    :raises ImportError: When ``openeye.<module_name>`` does not exist. Nothing is cached then,
        so a module that appears later is still read.
    """
    module_name = _openeye_leaf_name(module_name)
    snapshot = _OPENEYE_POTENTIAL_CLASSES.get(module_name)
    if snapshot is None:
        module = importlib.import_module(f"openeye.{module_name}")
        base = getattr(module, "OEMolFunc1", None)
        # Attribute access only; constructing all 69 is far too expensive to do here, and the
        # enumeration does not need to know which of them are abstract.
        snapshot = (
            frozenset(
                candidate
                for candidate in (getattr(module, name) for name in dir(module))
                if isinstance(candidate, type) and issubclass(candidate, base)
            )
            if isinstance(base, type)
            else frozenset()
        )
        _OPENEYE_POTENTIAL_CLASSES[module_name] = snapshot
    return snapshot


def _is_openeye_type(kind) -> bool:
    """:returns: Whether ``kind`` is one of the potential classes OpenEye itself defines.

    Membership in a snapshot, not a live read of the module. A module is mutable, so asking it
    what it exports *now* answered yes to any class assigned into it:
    ``oeff.Reexported = Reexported`` made a Python-defined subclass pass. A set captured before
    any of that happened closes it, and a frozenset lookup is cheaper besides.

    A miss falls through to the snapshot of the module ``kind`` claims, taking it if this is the
    first class seen from there. Without that, a class from a module imported after the first
    snapshot -- ``oequacpac.OEMolFunc1``, handed out by ``oeff`` itself once the partial-charge
    toolkit is loaded -- was reported as Python-defined, which it is not. A module already
    snapshotted is never re-read, so the rebinding above stays closed for it; a module not yet
    seen is read as it stands, which is the cold-ordering hole the module docstring accepts.
    """
    if kind in _openeye_potential_classes():
        return True
    module_name = _openeye_leaf_name(getattr(kind, "__module__", ""))
    if module_name == "oeff":
        return False
    try:
        return kind in _openeye_potential_classes(module_name)
    except (ImportError, TypeError, ValueError):
        return False


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

    A name list rather than a single name, because an upcast can land on any of them, and an
    interface is just as abstract whichever OpenEye module's proxy class it is -- so the name is
    matched first and then confirmed by identity against what ``kind``'s own module exports under
    it. Identity is what a Python class claiming the name cannot have; ``getattr`` returning
    ``None`` tolerates an OpenEye release that does not export one of these.
    """
    if kind.__name__ not in _ABSTRACT_POTENTIAL_INTERFACES:
        return False
    module = _openeye_module(kind)
    return module is not None and getattr(module, kind.__name__, None) is kind


def _is_molfunc1_instance(potential) -> bool:
    """:returns: Whether ``potential`` is an ``OEMolFunc1`` of some OpenEye binding module.

    The modules' potential hierarchies are disjoint: an ``oequacpac.OEMolFunc1`` proxy is not an
    instance of ``oeff.OEMolFunc1``, so testing ``oeff`` alone turned away the very components
    ``oeff`` hands out once the partial-charge toolkit is loaded. ``oeff`` is still tested first
    and unconditionally, because that is what a Python-defined subclass derives from, and such a
    subclass has to reach the spec 8.8 refusal below rather than be mistaken for a non-potential.
    """
    from openeye import oeff

    if isinstance(potential, oeff.OEMolFunc1):
        return True
    module = _openeye_module(type(potential))
    base = getattr(module, "OEMolFunc1", None) if module is not None else None
    return isinstance(base, type) and isinstance(potential, base)


def _as_molfunc1(potential):
    """Upcast an OpenEye potential to its ``OEMolFunc1`` subobject.

    ``OEMolFunc1`` is a virtual base of every OpenEye potential, so its subobject is not at the
    address an OpenEye SWIG proxy stores (measured: 48 bytes earlier for ``OEMMFF``). Adding the
    potential to an ``OEGenericFF`` and reading it back makes OpenEye's own runtime perform the
    upcast; the component it returns does not own the potential.

    :param potential: An OpenEye-defined ``oeff.OEMolFunc1``.
    :returns: ``(component, holder)``: a proxy of exact type ``OEMolFunc1`` pointing at
        ``potential`` -- from whichever OpenEye module last registered that class with SWIG, not
        necessarily ``oeff`` -- and the ``OEGenericFF`` it was read from, which the caller keeps
        alive while it uses the component.
    :raises TypeError: When ``potential`` is not an OpenEye ``OEMolFunc1``; is a Python-defined
        subclass of one, which spec 8.8 does not support in v1; or is itself an upcast proxy,
        whose provenance OEVibe cannot establish.
    :raises OptionsError: When OpenEye refuses to hold the potential.
    """
    from openeye import oeff

    if not _is_molfunc1_instance(potential):
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
        kind = type(potential)
        # The leaf, so that one class is named one way whichever import form the caller used.
        module_name = _openeye_leaf_name(kind.__module__)
        raise TypeError(
            f"this potential is an upcast proxy of exact type {module_name}."
            f"{kind.__name__}, an abstract interface; OEVibe cannot establish which potential "
            "is underneath an upcast and will not guess. Pass the original potential to "
            "MolFuncEngine instead."
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
