"""Entry-point discovery for the ``oevibe.engines`` group (spec 8.7)."""

import threading
import types

import pytest

import oevibe
from oevibe import engines as engines_module


class CountingEngine(oevibe.PythonEngine):
    """Minimal engine; ``methods_error`` makes GetMethods raise, as a broken plugin would."""

    def __init__(self, name, methods_error=None):
        super().__init__()
        self._name = name
        self._methods_error = methods_error

    def GetName(self):
        return self._name

    def GetVersion(self):
        return "0.9"

    def GetMethods(self):
        if self._methods_error is not None:
            raise self._methods_error
        capabilities = oevibe.MethodCapabilities()
        capabilities.method = oevibe.Method("spring")
        return [capabilities]

    def CreateSession(self, system, settings):
        raise oevibe.UnsupportedError("not needed for discovery tests")


def fake_entry_point(name, value, load):
    point = types.SimpleNamespace(name=name, value=value, group="oevibe.engines")
    point.load = load
    return point


@pytest.fixture
def fresh_discovery(monkeypatch):
    """Reset the load-once state so each test drives discovery itself."""
    monkeypatch.setattr(engines_module, "_LOADED", False)
    monkeypatch.setattr(engines_module, "_FAILED", [])
    return engines_module


def patch_entry_points(monkeypatch, points):
    def entry_points(*, group=None):
        return list(points) if group == "oevibe.engines" else []

    monkeypatch.setattr(engines_module.metadata, "entry_points", entry_points)


def by_name(infos, name):
    matches = [info for info in infos if info.name == name]
    assert matches, f"{name} not in {[info.name for info in infos]}"
    return matches[0]


def test_an_entry_point_engine_is_registered_and_listed(fresh_discovery, monkeypatch):
    patch_entry_points(
        monkeypatch,
        [
            fake_entry_point(
                "good", "pkg:make", lambda: lambda: CountingEngine("ep-good")
            )
        ],
    )
    info = by_name(oevibe.available_engines(), "ep-good")
    assert info.available
    assert info.version == "0.9"
    assert [method.name for method in info.methods] == ["spring"]


def test_discovery_does_not_deadlock_on_a_successful_load(fresh_discovery, monkeypatch):
    """Discovery registers the engines it finds while the first registry query is running.

    The bounded join is the assertion: a registration path that waited on the load it is part of
    would make this test time out instead of finishing.
    """
    patch_entry_points(
        monkeypatch,
        [
            fake_entry_point(
                "nodeadlock",
                "pkg:make",
                lambda: lambda: CountingEngine("ep-nodeadlock"),
            )
        ],
    )
    finished = threading.Event()

    def discover():
        oevibe.available_engines()
        finished.set()

    thread = threading.Thread(target=discover, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert finished.is_set(), "entry-point discovery did not finish; it is deadlocked"


def test_a_plugin_that_uses_the_public_api_does_not_deadlock(
    fresh_discovery, monkeypatch
):
    """A plugin may legitimately register a second engine, or ask what else is available.

    Both re-enter ``ensure_engines_loaded`` from the loading thread, which must find the load
    already claimed and return. The bounded join is the assertion.
    """

    def make():
        oevibe.register_engine(CountingEngine("ep-helper"), replace=True)
        oevibe.available_engines()
        return CountingEngine("ep-main")

    patch_entry_points(
        monkeypatch, [fake_entry_point("reentrant", "pkg:make", lambda: make)]
    )
    finished = threading.Event()

    def discover():
        oevibe.available_engines()
        finished.set()

    thread = threading.Thread(target=discover, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert finished.is_set(), (
        "a plugin using the public registry API deadlocked discovery"
    )


def test_a_plugin_worker_thread_using_the_public_api_does_not_deadlock(
    fresh_discovery, monkeypatch
):
    """A factory may hand registry calls to a worker thread and wait for it.

    Re-entry from the loading thread is not the only way in: the worker is a different thread,
    so a lock held across plugin code would block it while the factory blocks on the worker.
    The factory's own join is bounded too, and the outcome is recorded rather than asserted
    inside the factory, because the loader would swallow an assertion as a plugin failure.
    """
    worker_finished = []

    def make():
        done = threading.Event()

        def worker():
            oevibe.register_engine(CountingEngine("ep-worker-helper"), replace=True)
            oevibe.available_engines()
            done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=10)
        worker_finished.append(done.is_set())
        return CountingEngine("ep-worker-main")

    patch_entry_points(
        monkeypatch, [fake_entry_point("worker", "pkg:make", lambda: make)]
    )
    finished = threading.Event()

    def discover():
        oevibe.available_engines()
        finished.set()

    thread = threading.Thread(target=discover, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert finished.is_set(), "entry-point discovery did not finish; it is deadlocked"
    assert worker_finished == [True], (
        "a worker thread using the public registry API was blocked by discovery"
    )
    assert by_name(oevibe.available_engines(), "ep-worker-main").available


def test_a_factory_returning_something_else_is_listed_unavailable(
    fresh_discovery, monkeypatch
):
    """Spec 8.7: an entry point resolves to a callable returning a PythonEngine."""
    patch_entry_points(
        monkeypatch, [fake_entry_point("wrong", "pkg:make", lambda: lambda: object())]
    )
    info = by_name(oevibe.available_engines(), "wrong")
    assert not info.available
    assert "PythonEngine" in info.reason


def test_entry_points_load_exactly_once(fresh_discovery, monkeypatch):
    loads = []

    def load():
        loads.append(1)
        return lambda: CountingEngine("ep-once")

    patch_entry_points(monkeypatch, [fake_entry_point("once", "pkg:make", load)])
    oevibe.available_engines()
    oevibe.available_engines()
    assert len(loads) == 1


def test_a_failing_entry_point_is_listed_with_its_exception_text(
    fresh_discovery, monkeypatch
):
    def load():
        raise ImportError("no module named widget")

    patch_entry_points(monkeypatch, [fake_entry_point("broken", "widget:make", load)])
    info = by_name(oevibe.available_engines(), "broken")
    assert not info.available
    assert "no module named widget" in info.reason


def test_a_failing_enumeration_is_listed_and_not_retried(fresh_discovery, monkeypatch):
    """A malformed ``entry_points.txt`` in any installed distribution breaks enumeration.

    The failure is recorded like any other discovery failure rather than raised or lost, and
    it is not retried on every later query.
    """
    calls = []

    def entry_points(*, group=None):
        calls.append(group)
        raise ValueError("malformed entry_points.txt")

    monkeypatch.setattr(engines_module.metadata, "entry_points", entry_points)
    first = oevibe.available_engines()
    second = oevibe.available_engines()
    for infos in (first, second):
        failed = [info for info in infos if "malformed entry_points.txt" in info.reason]
        assert len(failed) == 1
        assert not failed[0].available
        assert by_name(infos, "oeff").available
    assert len(calls) == 1


def test_an_engine_whose_get_methods_raises_is_listed_unavailable(
    fresh_discovery, monkeypatch
):
    patch_entry_points(
        monkeypatch,
        [
            fake_entry_point(
                "angry",
                "pkg:make",
                lambda: (
                    lambda: CountingEngine(
                        "ep-angry", RuntimeError("parameter file missing")
                    )
                ),
            )
        ],
    )
    info = by_name(oevibe.available_engines(), "ep-angry")
    assert not info.available
    assert "parameter file missing" in info.reason


def test_a_name_collision_keeps_the_first_entry_point(fresh_discovery, monkeypatch):
    patch_entry_points(
        monkeypatch,
        [
            fake_entry_point(
                "first", "a:make", lambda: lambda: CountingEngine("ep-clash")
            ),
            fake_entry_point(
                "second", "b:make", lambda: lambda: CountingEngine("ep-clash")
            ),
        ],
    )
    infos = oevibe.available_engines()
    kept = [info for info in infos if info.name == "ep-clash" and info.available]
    rejected = [
        info for info in infos if info.name == "ep-clash" and not info.available
    ]
    assert len(kept) == 1
    assert len(rejected) == 1
    assert "already registered" in rejected[0].reason
    assert "b:make" in rejected[0].reason


def test_a_builtin_name_from_an_entry_point_is_rejected(fresh_discovery, monkeypatch):
    patch_entry_points(
        monkeypatch,
        [fake_entry_point("sneaky", "c:make", lambda: lambda: CountingEngine("oeff"))],
    )
    infos = oevibe.available_engines()
    assert any(info.name == "oeff" and not info.available for info in infos)
    assert any(info.name == "oeff" and info.available for info in infos)


def test_builtin_engines_are_listed_with_their_methods():
    oeff = by_name(oevibe.available_engines(), "oeff")
    assert oeff.available
    names = {method.name for method in oeff.methods}
    assert {"mmff94", "mmff94s"} <= names
    assert all(method.available for method in oeff.methods)


def test_a_runtime_loaded_engine_reports_why_it_is_unavailable():
    """Spec 12.2 asks for the missing-library reason; which branch runs depends on the
    environment, and both are real answers, so the test checks the pair rather than one of them."""
    tblite = by_name(oevibe.available_engines(), "tblite")
    if tblite.available:
        assert tblite.reason == ""
        assert {method.name for method in tblite.methods} >= {"gfn2-xtb", "gfn1-xtb"}
    else:
        assert "libtblite" in tblite.reason or "tblite" in tblite.reason
        assert tblite.methods == ()


def test_constructing_a_calculation_triggers_discovery(fresh_discovery, monkeypatch):
    patch_entry_points(
        monkeypatch,
        [
            fake_entry_point(
                "lazy", "pkg:make", lambda: lambda: CountingEngine("ep-lazy")
            )
        ],
    )
    options = oevibe.CalculationOptions()
    options.SetEngine("ep-lazy")
    options.SetMethod(oevibe.Method("spring"))
    oevibe.Calculation(options)  # resolves only if discovery ran inside the constructor


def test_bond_order_is_exported_and_round_trips_through_evaluation():
    """Spec 8.7 documents ``Evaluation(bond_orders=...)``, so a plugin needs ``oevibe.BondOrder``."""
    bond = oevibe.BondOrder()
    bond.i, bond.j, bond.order = 0, 1, 1.5
    evaluation = oevibe.Evaluation(energy=0.0, bond_orders=[bond])
    assert [(b.i, b.j, b.order) for b in evaluation.bond_orders] == [(0, 1, 1.5)]
