"""Bridges between OEVibe and OpenEye's potential and optimizer interfaces (spec 8.8).

The traffic runs both ways. :func:`MolFuncEngine` wraps an OpenEye potential as an OEVibe engine,
so OEVibe's optimizer, scans and thermochemistry can drive it; :func:`as_oemolfunc` wraps OEVibe
options as an ``openeye.oeff.OEMolFunc1``, so OpenEye's optimizers can drive OEVibe. The two do
not compose: an ``as_oemolfunc`` adapter is a Python-defined subclass, which the rule below makes
:func:`MolFuncEngine` refuse. Neither half imports ``openeye`` at module scope, so importing
``oevibe`` does not pull the toolkit in.

The predicates in between belong to the inbound half, the only one that has to judge what it was
handed. The outbound half judges nothing, since it builds the class it hands out; what it has
instead are two conventions to know before reading it. One is an ownership split, ``CreateCopy``
being OpenEye's caller-owns hook and ``Copy`` the Python-owned sibling. The other is that its
``__call__`` never raises: an exception leaving a director becomes a
``Swig::DirectorMethodException`` that nothing in OpenEye's optimizers catches, which aborts the
process, so a failed evaluation is recorded, warned about and reported as an infinite energy
instead.

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

import contextlib
import ctypes
import importlib
import warnings

import numpy as np

from .oevibe import OptionsError, _EngineMolFunc, _MolFuncEngine

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


def _buffer_view(buffer, count):
    """:returns: A writable numpy view of ``count`` doubles, or None when ``buffer`` is a sequence.

    ``OEBFGSOpt`` upcalls a Python potential with bare ``SwigPyObject``s wrapping ``double*``; they
    expose no length and no indexing, only ``__int__``, which is the buffer address. A direct call
    from Python instead passes an ``oechem.OEDoubleArray``, which is a sequence.
    """
    if hasattr(buffer, "__len__"):
        return None
    pointer = ctypes.cast(int(buffer), ctypes.POINTER(ctypes.c_double))
    return np.ctypeslib.as_array(pointer, shape=(count,))


def _read_doubles(buffer, count):
    """:returns: ``count`` doubles from an OpenEye argument as a float64 array."""
    view = _buffer_view(buffer, count)
    if view is not None:
        return view
    return np.fromiter(
        (buffer[index] for index in range(count)), dtype=np.float64, count=count
    )


def _accumulate(buffer, values):
    """Adds ``values`` into an OpenEye output buffer, which OpenEye never expects to be cleared."""
    view = _buffer_view(buffer, len(values))
    if view is not None:
        view += values
        return
    for index, value in enumerate(values):
        buffer[index] += value


# What a failed evaluation reports to OpenEye in place of an energy, paired with no gradient
# contribution at all. An exception is not available: leaving a director it becomes a
# Swig::DirectorMethodException, OpenEye's optimizers install no handler for one, and the
# process aborts -- so the choice is between a value and a crash, and the only question is
# which value.
#
# Measured in OEBFGSOpt against an ill-conditioned quadratic that takes 43 evaluations to
# minimize unimpeded, failing at several points in the run: +inf, 1e30 and 1e12 all make the
# line search REJECT the failed step, so the geometry left in the output buffer is one that
# evaluated successfully, while nan, -inf and 0.0 are accepted and the caller is handed the
# failed point. Every candidate terminated within three evaluations of the failure, the zero
# gradient being what ends the run, so promptness did not separate them and step rejection did.
# +inf over a large finite value because a finite one is a plausible energy: a caller sanity-
# checking magnitudes cannot tell 1e30 kcal/mol from a real number out of a broken engine,
# where an infinity says exactly what happened. +inf over nan because nan compares false
# against everything, so ``energy < threshold`` silently fails closed and arithmetic downstream
# spreads it without a trace.
_FAILED_EVALUATION_ENERGY = float("inf")


def as_oemolfunc(options):
    """OEVibe as an ``openeye.oeff.OEMolFunc1``, for OpenEye's optimizers (spec 8.8).

    The returned object resolves and validates the engine at construction, builds its system and
    session in ``Setup(mol)``, and reports energies in kcal/mol and gradients in
    kcal/mol/Angstrom in the OpenEye layout of the molecule it was set up on.

    **Calling it never raises, so a caller driving an OpenEye optimizer must check
    ``GetEvaluationError()`` on the returned object when the run finishes.** An exception cannot
    cross OpenEye's director -- SWIG turns one into a ``Swig::DirectorMethodException``, nothing
    in OpenEye's optimizers catches it, and the process aborts -- so a failed evaluation warns,
    contributes nothing to the gradient and reports an infinite energy, which stops the
    optimizer. ``GetEvaluationError()`` returns the first exception an evaluation raised, or
    ``None`` if none did; a run that hit one has a meaningless result, whatever it returned.

    :param options: An ``oevibe.CalculationOptions``.
    :returns: An ``openeye.oeff.OEMolFunc1`` subclass instance. It is not thread-safe; its
        ``Copy()`` makes the independent, Python-owned copy a worker thread needs.
    :raises OptionsError: For invalid options or a method the engine does not provide.
    :raises EngineUnavailableError: When the engine or the method is unavailable.
    """
    from openeye import oeff

    from .engines import ensure_engines_loaded

    # Resolving the engine is a registry query, and this one does not go through Calculation,
    # so an entry-point engine would otherwise be invisible here (spec 8.7).
    ensure_engines_loaded()

    class _OEVibeMolFunc(oeff.OEMolFunc1):
        """One OEVibe calculation presented to OpenEye.

        Not thread-safe -- it caches the last geometry it was evaluated at -- so give each
        thread its own, made with :meth:`Copy`. :meth:`CreateCopy` is OpenEye's hook and gives
        its result away to C++; see there.

        An instance holds no reference to the caller's ``CalculationOptions``: the C++
        ``EngineMolFunc`` copied them at construction, and both copy methods go through C++, so
        a copy inherits that snapshot rather than re-reading a proxy the caller may have edited
        since.

        Calling it never raises; a failed evaluation is reported through
        :meth:`GetEvaluationError` instead.
        """

        def __init__(self, calculation_options=None, function=None):
            oeff.OEMolFunc1.__init__(self)
            self._function = (
                function
                if function is not None
                else _EngineMolFunc(calculation_options)
            )
            # Per adapter, not per copy: a copy is a different calculation and starts clean.
            self._evaluation_error: Exception | None = None

        def CreateCopy(self):
            """OpenEye's copy hook, for OpenEye's C++ to call.

            ``OEMolFunc1::CreateCopy() const`` is caller-owns, so the copy is disowned here and
            whoever asked for it deletes it. A Python caller cannot honour that contract -- it
            would be holding an object it no longer owns and nothing else ever will -- so Python
            should call :meth:`Copy` instead.

            :returns: A disowned copy with the same options, not yet set up on a molecule.
            """
            copy = _OEVibeMolFunc(function=self._function.CreateCopy())
            # OpenEye takes ownership of a copy, so Python must stop owning it.
            copy.__disown__()
            return copy

        def Copy(self):
            """The Python-owned sibling of :meth:`CreateCopy`.

            Same independent C++ function underneath, same options snapshot, but ownership stays
            with Python, so the copy is freed like any other object once the last reference to
            it goes. This is the one to use for the per-thread copy the class docstring asks
            for; ``CreateCopy`` would leak it.

            :returns: An independent copy with the same options, not yet set up on a molecule.
            """
            return _OEVibeMolFunc(function=self._function.CreateCopy())

        def Setup(self, mol):
            return self._function.Setup(mol)

        def NumVar(self):
            return self._function.NumVar()

        def GetFComponents(self, coords):
            return None

        def GetEvaluationError(self):
            """The first exception an evaluation raised, or ``None`` if none has.

            The one thing that has to be read after driving an OpenEye optimizer with this
            adapter: ``__call__`` cannot raise, so nothing else distinguishes a run that failed
            from one that succeeded.

            The FIRST failure, because an optimizer makes a step or two more after the one that
            failed and those fail for the same underlying reason, so the first is the one that
            explains the run. It is kept for the life of the adapter and never cleared; for an
            unrelated run, start from a fresh adapter or a :meth:`Copy`, which begins clean.

            :returns: The exception, or ``None``.
            """
            return self._evaluation_error

        def _report_failure(self, error):
            """Record ``error`` and warn about it, so ``__call__`` can return the sentinel."""
            if self._evaluation_error is None:
                self._evaluation_error = error
            # Under ``-W error`` warnings.warn raises, and a raise here is precisely the abort
            # this path exists to prevent. The failure is recorded before the warning is
            # attempted, so such a filter costs the notice but not the information.
            with contextlib.suppress(Exception):
                warnings.warn(
                    "an OEVibe evaluation failed and was reported to the caller as an "
                    f"infinite energy, so this result is meaningless: {error!r}. The first "
                    "failure of this function is available from GetEvaluationError().",
                    RuntimeWarning,
                    # Past _report_failure and __call__, which is the caller's own frame on a
                    # direct call; from OpenEye's director there is no Python frame to name.
                    stacklevel=3,
                )

        def __call__(self, *args):
            # Nothing may escape: this is called from an OpenEye director, where an exception
            # becomes a Swig::DirectorMethodException and aborts the process. The body is small
            # and the gradient write is last, so the whole of it is inside the guard rather
            # than only the engine call.
            try:
                count = self._function.NumVar()
                coords = _read_doubles(args[0], count)
                value = self._function.Evaluate(coords.tolist(), len(args) > 1)
                if len(args) > 1:
                    _accumulate(args[1], np.asarray(value.gradient, dtype=np.float64))
            except Exception as error:  # noqa: BLE001 - reported, not swallowed; see above
                # A KeyboardInterrupt is deliberately not caught: it is not an engine failure,
                # and turning one into a warning would finish the optimization the user asked
                # to stop.
                self._report_failure(error)
                return _FAILED_EVALUATION_ENERGY
            return value.energy

    return _OEVibeMolFunc(options)
