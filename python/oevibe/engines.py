"""Engines written in Python (spec 8.7).

A subclass of :class:`PythonEngine` describes itself to the registry and hands out
:class:`PythonEngineSession` objects; the session subclass implements ``evaluate`` against numpy
arrays in the units of spec 3.1. The C++ side sees ordinary ``Engine`` and ``EngineSession``
objects through SWIG directors.
"""

from __future__ import annotations

import dataclasses
import threading
from importlib import metadata

import numpy as np

from .oevibe import Engine as _Engine
from .oevibe import EngineRegistry
from .oevibe import EngineSession as _EngineSession
from .oevibe import Evaluation as _Evaluation


def _flat(values, name: str, width: int = 3):
    """Flatten ``values`` to a list of floats.

    :param values: An array-like of shape (M, ``width``) or (M * ``width``,).
    :param name: The quantity's name, for the error message.
    :param width: The column count of the two-dimensional form.
    :returns: ``values`` as a flat list of floats.
    :raises ValueError: When the array is neither one-dimensional nor ``width`` columns wide.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == width:
        return array.reshape(-1).tolist()
    if array.ndim == 1:
        return array.tolist()
    raise ValueError(
        f"{name} must be a flat array or an (N, {width}) array, got shape {array.shape}"
    )


class Evaluation(_Evaluation):
    """One geometry's results, built from numpy arrays in spec 3.1 units.

    Every argument is optional and absent values stay absent, so an engine reports exactly what
    was requested. ``gradient`` takes (N, 3) or (3N,); ``hessian`` takes (3N, 3N) or (9N**2,);
    ``charges`` takes (N,); ``dipole`` takes (3,).

    :param energy: kcal/mol.
    :param gradient: kcal/mol/Angstrom.
    :param hessian: kcal/mol/Angstrom**2, row-major in dense order.
    :param charges: Partial charges in e.
    :param dipole: Debye.
    :param orbital_energies: eV, ascending.
    :param homo_index: Index into ``orbital_energies``; -1 when there are no occupied orbitals.
    :param bond_orders: A sequence of ``oevibe.BondOrder``.
    :param messages: Engine messages attached to the result.
    :raises ValueError: For an array whose shape does not fit the quantity.
    """

    def __init__(
        self,
        *,
        energy=None,
        gradient=None,
        hessian=None,
        charges=None,
        dipole=None,
        orbital_energies=None,
        homo_index=None,
        bond_orders=None,
        messages=None,
    ):
        super().__init__()
        if energy is not None:
            self.energy = float(energy)
        if gradient is not None:
            self.gradient = _flat(gradient, "gradient")
        if hessian is not None:
            array = np.asarray(hessian, dtype=np.float64)
            if array.ndim > 2 or (array.ndim == 2 and array.shape[0] != array.shape[1]):
                raise ValueError(
                    f"hessian must be a flat array or a square matrix, got shape {array.shape}"
                )
            self.hessian = array.reshape(-1).tolist()
        if charges is not None:
            self.charges = np.asarray(charges, dtype=np.float64).reshape(-1).tolist()
        if dipole is not None:
            self.dipole = _flat(dipole, "dipole")
        if orbital_energies is not None:
            self.orbital_energies = (
                np.asarray(orbital_energies, dtype=np.float64).reshape(-1).tolist()
            )
        if homo_index is not None:
            self.homo_index = int(homo_index)
        if bond_orders is not None:
            self.bond_orders = list(bond_orders)
        if messages is not None:
            self.messages = [str(message) for message in messages]


class PythonEngineSession(_EngineSession):
    """Base for a session written in Python (spec 8.7).

    Subclasses implement :meth:`evaluate`. This class converts the C++ dense coordinate vector
    into a read-only ``(N, 3)`` float64 numpy array in Angstrom and passes the request through.
    """

    def evaluate(self, coords, request) -> Evaluation:
        """Evaluate one geometry.

        :param coords: Read-only (N, 3) float64 array in Angstrom, dense atom order.
        :param request: What to compute; ``request.gradient``, ``request.hessian`` and
            ``request.properties``.
        :returns: An :class:`Evaluation`.
        """
        raise NotImplementedError(
            "a PythonEngineSession subclass must implement evaluate()"
        )

    def Evaluate(self, coords, request):
        """The C++ ``EngineSession`` entry point; subclasses override :meth:`evaluate` instead."""
        xyz = np.fromiter(coords, dtype=np.float64, count=len(coords)).reshape(-1, 3)
        # A write here would not reach C++, so the array says so rather than losing the change.
        xyz.setflags(write=False)
        evaluation = self.evaluate(xyz, request)
        if not isinstance(evaluation, _Evaluation):
            raise TypeError(
                f"evaluate() must return an oevibe.Evaluation, got {type(evaluation).__name__}"
            )
        return evaluation


class PythonEngine(_Engine):
    """Base for an engine written in Python (spec 8.7).

    A subclass must implement :meth:`GetName`, :meth:`GetVersion`, :meth:`GetMethods` and
    :meth:`CreateSession`. It may override ``IsAvailable``, ``GetUnavailableReason``,
    ``GetKeywordSpecs``, ``ValidateSettings`` and ``ValidateSystem``; the defaults report an
    available engine with no keywords and the C++ system checks.
    """

    def GetName(self) -> str:
        raise NotImplementedError("a PythonEngine subclass must implement GetName()")

    def GetVersion(self) -> str:
        raise NotImplementedError("a PythonEngine subclass must implement GetVersion()")

    def GetMethods(self):
        raise NotImplementedError("a PythonEngine subclass must implement GetMethods()")

    def CreateSession(self, system, settings) -> PythonEngineSession:
        """Build the session that evaluates one conformer.

        :param system: The snapshot; ``system.molecule`` is a fresh ``oechem.OEMol`` copy.
        :param settings: The resolved method, solvent, preset, accuracy, properties and keywords.
        :returns: A new session; never shared between conformers (spec 6.4).
        :raises UnsupportedError: For a system this engine cannot handle; it becomes UNSUPPORTED.
        """
        raise NotImplementedError(
            "a PythonEngine subclass must implement CreateSession()"
        )

    def IsAvailable(self) -> bool:
        return True

    def GetUnavailableReason(self) -> str:
        return ""


def _register_retained(engine, replace: bool) -> None:
    """Register ``engine`` without triggering entry-point discovery.

    The public :func:`register_engine` starts discovery before it registers, and discovery
    registers the engines it finds through this function instead, so that loading an engine never
    goes back through the query path. The binding's ``Register`` wrapper keeps ``engine`` alive
    for the life of the process before the native call commits, so nothing further is retained
    here.

    :param engine: An ``oevibe.Engine`` instance.
    :param replace: Replace an engine already registered under the same name.
    :raises OptionsError: For an empty or built-in name, or a duplicate without ``replace``.
    """
    EngineRegistry.Instance().Register(engine, replace)


def register_engine(engine, *, replace: bool = False) -> None:
    """Register a Python engine in the process-wide registry (spec 8.7).

    The package keeps the engine alive for the life of the process, because the C++ side only
    borrows its Python object.

    Registering counts as the first registry query, so the ``oevibe.engines`` entry points load
    before ``engine`` is registered; an entry-point engine of the same name is therefore already
    present and is displaced only with ``replace=True``. The one exception is a registration from
    another thread while the first query is still loading; see :func:`ensure_engines_loaded`.

    :param engine: An :class:`PythonEngine` (strictly, any ``oevibe.Engine``) instance.
    :param replace: Replace an engine already registered under the same name.
    :raises TypeError: When ``engine`` is not an ``oevibe.Engine``.
    :raises OptionsError: For an empty or built-in name, or a duplicate name without ``replace``.
    """
    ensure_engines_loaded()
    if not isinstance(engine, _Engine):
        raise TypeError(
            f"register_engine needs an oevibe.Engine, got {type(engine).__name__}"
        )
    _register_retained(engine, replace)


_ENTRY_POINT_GROUP = "oevibe.engines"
# Guards only the check-and-claim of _LOADED and is never held while plugin code runs. A plugin
# may call the public registry API from its loading thread or from a worker thread it starts and
# waits for; holding any lock across the plugin's code would deadlock the second case, which no
# kind of re-entrant lock can fix. Claiming the load first means every later caller, on any
# thread, sees _LOADED true and returns at once.
_LOAD_LOCK = threading.Lock()
_LOADED = False
# Entry points that produced no usable engine, reported by available_engines() alongside the
# registry's own engines (spec 8.7).
_FAILED: list[EngineInfo] = []


@dataclasses.dataclass(frozen=True)
class MethodInfo:
    """One method of an engine, with its own availability (spec 8.2)."""

    name: str
    available: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class EngineInfo:
    """One engine as ``available_engines()`` reports it (spec 8.2)."""

    name: str
    version: str
    available: bool
    reason: str
    methods: tuple[MethodInfo, ...]


def _describe(engine) -> EngineInfo:
    """:returns: One engine's report; any exception it raises becomes its unavailable reason."""
    try:
        name = engine.GetName()
    except Exception as error:  # noqa: BLE001 - a broken engine must not break the listing
        return EngineInfo("<unnamed>", "", False, f"GetName() raised: {error}", ())
    try:
        version = engine.GetVersion()
    except Exception as error:  # noqa: BLE001
        return EngineInfo(name, "", False, f"GetVersion() raised: {error}", ())
    try:
        available = bool(engine.IsAvailable())
        reason = "" if available else engine.GetUnavailableReason()
    except Exception as error:  # noqa: BLE001
        return EngineInfo(name, version, False, f"IsAvailable() raised: {error}", ())
    if not available:
        return EngineInfo(name, version, False, reason, ())
    try:
        methods = tuple(
            MethodInfo(
                caps.method.GetName(), bool(caps.available), caps.unavailable_reason
            )
            for caps in engine.GetMethods()
        )
    except Exception as error:  # noqa: BLE001
        return EngineInfo(name, version, False, f"GetMethods() raised: {error}", ())
    return EngineInfo(name, version, True, "", methods)


def _load_entry_point(point) -> None:
    """Load one entry point and register its engine, recording any failure in ``_FAILED``.

    Runs after the load is claimed and outside ``_LOAD_LOCK``, and registers through the private
    :func:`_register_retained` rather than :func:`register_engine`, so loading an engine never
    goes back through the query path.
    """
    # Until the engine can name itself, the entry point's name is all there is to report under.
    name = point.name
    try:
        factory = point.load()
        engine = factory()
        if not isinstance(engine, PythonEngine):
            raise TypeError(
                f"{point.value!r} returned {type(engine).__name__}, not an oevibe.PythonEngine"
            )
        name = engine.GetName()
        engine.GetMethods()  # spec 8.7: a plugin that cannot describe itself is unavailable
    except Exception as error:  # noqa: BLE001 - one bad plugin must not hide the others
        _FAILED.append(
            EngineInfo(
                name, "", False, f"entry point {point.value!r} failed: {error}", ()
            )
        )
        return
    if EngineRegistry.Instance().Find(name) is not None:
        # First in importlib.metadata order wins (spec 8.7).
        _FAILED.append(
            EngineInfo(
                name,
                "",
                False,
                f"engine name {name!r} is already registered; "
                f"entry point {point.value!r} was skipped",
                (),
            )
        )
        return
    try:
        _register_retained(engine, False)
    except Exception as error:  # noqa: BLE001
        _FAILED.append(
            EngineInfo(
                name, "", False, f"entry point {point.value!r} failed: {error}", ()
            )
        )


def ensure_engines_loaded() -> None:
    """Load the ``oevibe.engines`` entry points once, on the first registry query (spec 8.7).

    The first caller claims the load and runs it on its own thread. A query that arrives from
    another thread while that load is still running returns at once instead of waiting, so it
    may see engines that are still being registered, or fail to find an entry-point engine that
    has not been reached yet. That race is deliberate: waiting would deadlock a plugin whose
    factory hands registry calls to a worker thread and joins it, and a plugin cannot be told
    apart from any other thread.
    """
    global _LOADED
    with _LOAD_LOCK:
        if _LOADED:
            return
        # Claimed before loading and never reset: a plugin or an enumeration that fails must not
        # make every later query retry it.
        _LOADED = True
    try:
        points = list(metadata.entry_points(group=_ENTRY_POINT_GROUP))
    except Exception as error:  # noqa: BLE001 - a malformed distribution must not break queries
        _FAILED.append(
            EngineInfo(
                f"<{_ENTRY_POINT_GROUP}>",
                "",
                False,
                f"could not enumerate the {_ENTRY_POINT_GROUP!r} entry points: {error}",
                (),
            )
        )
        return
    for point in points:
        _load_entry_point(point)


def available_engines() -> list[EngineInfo]:
    """Every known engine with its availability and, when unavailable, the reason (spec 8.2).

    :returns: The registry's engines in registration order, then the entry points that failed.
    """
    ensure_engines_loaded()
    infos = [_describe(engine) for engine in EngineRegistry.Instance().List()]
    infos.extend(_FAILED)
    return infos
