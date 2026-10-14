"""The GIL boundary: exactly five wrappers release it, and they really do (spec 8.1, 8.7)."""

import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from openeye import oechem, oeomega

import oevibe

INTERFACE = Path(__file__).resolve().parents[2] / "swig" / "oevibe.i"
RELEASING = {
    "Calculation::Energy",
    "Calculation::Gradient",
    "Calculation::Optimize",
    "Calculation::Frequencies",
    "Calculation::Scan",
}


def test_only_the_five_verbs_are_declared_to_release_the_gil():
    """A new %threadallow is a deliberate decision, not something a later task adds in passing."""
    text = INTERFACE.read_text()
    assert '%feature("nothreadallow", "1")' in text
    declared = set(
        re.findall(r"^%threadallow\s+OEVIBE::([\w:]+)\s*;", text, flags=re.MULTILINE)
    )
    assert declared == RELEASING


def test_another_python_thread_runs_between_two_director_callbacks():
    """A busy Python thread must make progress *between* two engine callbacks.

    This is the discriminating test. A director callback holds the GIL, so the worker thread can
    only run while the calculation is in native code between callbacks. If a verb held the GIL
    for its whole run, the counter the engine records on each conformer would never change.

    ``Gradient`` is used rather than ``Optimize`` because it makes exactly one callback per
    conformer whatever the potential looks like; an optimization of a spring anchored at its own
    starting geometry would converge after a single evaluation and observe nothing.

    The switch interval is raised for the duration. At the default 5 ms, a waiting thread makes
    the GIL holder yield at its next bytecode, so a verb that never released the GIL would still
    let the worker run during any engine callback once the verb had lasted that long, and the
    test would pass on exactly the regression it exists to catch. With a one-second interval a
    GIL-holding verb of this size cannot be preempted, and the worker yielding after every
    increment means a released verb never has to wait out the interval for it.
    """
    counter = [0]
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            counter[0] += 1
            time.sleep(0)

    class WatchingSession(oevibe.PythonEngineSession):
        def __init__(self, observations):
            super().__init__()
            self.observations = observations

        def evaluate(self, coords, request):
            self.observations.append(counter[0])
            if request.gradient:
                return oevibe.Evaluation(energy=0.0, gradient=np.zeros_like(coords))
            return oevibe.Evaluation(energy=0.0)

    class WatchingEngine(oevibe.PythonEngine):
        def __init__(self):
            super().__init__()
            self.observations: list[int] = []

        def GetName(self):
            return "gil-watch"

        def GetVersion(self):
            return "1.0"

        def GetMethods(self):
            capabilities = oevibe.MethodCapabilities()
            capabilities.method = oevibe.Method("spring")
            return [capabilities]

        def CreateSession(self, system, settings):
            return WatchingSession(self.observations)

    engine = WatchingEngine()
    oevibe.register_engine(engine, replace=True)

    mol = oechem.OEMol()
    assert oechem.OESmilesToMol(mol, "CCCCCCCCO")
    omega_options = oeomega.OEOmegaOptions()
    omega_options.SetMaxConfs(50)
    assert (
        oeomega.OEOmega(omega_options).Build(mol) == oeomega.OEOmegaReturnCode_Success
    )
    assert mol.NumConfs() >= 10, "too few conformers to observe the GIL being released"

    options = oevibe.CalculationOptions()
    options.SetEngine("gil-watch")
    options.SetMethod(oevibe.Method("spring"))

    previous_interval = sys.getswitchinterval()
    thread = threading.Thread(target=worker, daemon=True)
    try:
        # Set before the worker starts: a thread already inside a timed wait for the GIL keeps
        # the interval it started waiting with, and would still preempt a GIL-holding verb
        # after 5 ms.
        sys.setswitchinterval(1.0)
        thread.start()
        results = oevibe.Calculation(options).Gradient(mol)
    finally:
        stop.set()
        thread.join(timeout=10)
        sys.setswitchinterval(previous_interval)

    for result in results:
        assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
    assert len(engine.observations) == mol.NumConfs()
    assert engine.observations[-1] > engine.observations[0], (
        "the worker thread never ran between two engine callbacks, so the verb held the GIL"
    )
