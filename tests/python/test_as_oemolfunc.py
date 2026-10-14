"""OEVibe as an OpenEye potential, for OpenEye's own optimizers (spec 8.8)."""

import gc
import itertools
import os
import subprocess
import sys
import threading
import types
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest
from openeye import oechem, oeff

import oevibe


def mmff_options():
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94s"))
    return options


def native_mmff(mol):
    """OpenEye's own MMFF94s, for the reference numbers.

    ``mol`` has to outlive the returned potential: OpenEye keeps pointers into the molecule it
    was set up on, and a temporary argument leaves it reading freed memory -- which ``OEBFGSOpt``
    spins on indefinitely rather than failing. Every caller below therefore holds the molecule in
    a named local, and no copy is made here, because copying would only move the problem.
    """
    potential = oeff.OEMMFF(True)
    assert potential.PrepMol(mol, False, True)
    assert potential.Setup(mol)
    return potential


def coordinates(mol):
    values = oechem.OEDoubleArray(3 * mol.GetMaxAtomIdx())
    mol.GetCoords(values)
    return values


def test_energy_matches_native_mmff(butanol_conformers):
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    assert function.NumVar() == 3 * mol.GetMaxAtomIdx()
    xs = coordinates(mol)
    assert function(xs) == pytest.approx(native_mmff(mol)(xs), abs=1e-9)


def test_gradient_matches_native_mmff_and_accumulates(butanol_conformers):
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    xs = coordinates(mol)
    nvar = function.NumVar()

    mine = oechem.OEDoubleArray(nvar)
    energy = function(xs, mine)
    theirs = oechem.OEDoubleArray(nvar)
    native_energy = native_mmff(mol)(xs, theirs)
    assert energy == pytest.approx(native_energy, abs=1e-9)
    assert np.allclose(
        [mine[i] for i in range(nvar)], [theirs[i] for i in range(nvar)], atol=1e-9
    )

    prefilled = oechem.OEDoubleArray(nvar)
    for index in range(nvar):
        prefilled[index] = 5.0
    function(xs, prefilled)
    assert np.allclose(
        [prefilled[i] for i in range(nvar)],
        [5.0 + mine[i] for i in range(nvar)],
        atol=1e-9,
    )


def test_it_drives_oebfgsopt_to_the_native_minimum(butanol_conformers):
    """Spike S2's check, now against OEVibe rather than a delegate."""
    mol = oechem.OEMol(butanol_conformers)
    xs = coordinates(mol)

    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    mine_out = oechem.OEDoubleArray(function.NumVar())
    mine = oeff.OEBFGSOpt()(function, xs, mine_out)

    native_mol = oechem.OEMol(butanol_conformers)
    native = native_mmff(native_mol)
    native_out = oechem.OEDoubleArray(native.NumVar())
    theirs = oeff.OEBFGSOpt()(native, xs, native_out)

    assert mine == pytest.approx(theirs, abs=1e-4)
    assert np.allclose(
        [mine_out[i] for i in range(function.NumVar())],
        [native_out[i] for i in range(native.NumVar())],
        atol=1e-3,
    )


def test_an_entry_point_engine_is_reachable_through_the_adapter(
    butanol_conformers, monkeypatch
):
    """Spec 8.7 loads entry points on the first registry query; as_oemolfunc is one."""
    from mmff_engine import MMFFEngine

    from oevibe import engines as engines_module

    monkeypatch.setattr(engines_module, "_LOADED", False)
    monkeypatch.setattr(engines_module, "_FAILED", [])

    def entry_points(*, group=None):
        if group != "oevibe.engines":
            return []
        point = types.SimpleNamespace(name="mmff", value="mmff_engine:make_engine")
        point.load = lambda: lambda: MMFFEngine("mmff-entry-point")
        return [point]

    monkeypatch.setattr(engines_module.metadata, "entry_points", entry_points)

    options = oevibe.CalculationOptions()
    options.SetEngine("mmff-entry-point")
    options.SetMethod(oevibe.Method("mmff94s"))
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(options)
    assert function.Setup(mol)
    native_mol = oechem.OEMol(butanol_conformers)
    expected = native_mmff(native_mol)(coordinates(native_mol))
    assert function(coordinates(mol)) == pytest.approx(expected, abs=1e-9)


def test_copy_gives_an_independent_function(butanol_conformers):
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    copy = function.Copy()
    assert copy is not function
    assert copy.Setup(mol)
    xs = coordinates(mol)
    assert copy(xs) == pytest.approx(function(xs), abs=1e-12)


def test_a_copy_ignores_later_edits_to_the_callers_options(butanol_conformers):
    """The options are snapshotted in C++ at construction, so nothing the caller does to its
    own object afterwards can change what a copy computes."""
    mol = oechem.OEMol(butanol_conformers)
    options = mmff_options()
    function = oevibe.as_oemolfunc(options)
    assert function.Setup(mol)
    xs = coordinates(mol)
    before = function(xs)

    options.SetMethod(oevibe.Method("gfn2-xtb"))
    options.SetEngine("tblite")
    copy = function.Copy()
    assert copy.Setup(mol)
    assert copy(xs) == pytest.approx(before, abs=1e-12)


def test_a_python_copy_is_collected_when_its_last_reference_goes(butanol_conformers):
    """Why ``Copy`` exists at all: ``CreateCopy`` cannot be collectable and stay correct.

    ``OEMolFunc1::CreateCopy() const`` is caller-owns, so the adapter has to ``__disown__()``
    what it returns -- right when OpenEye's C++ is the caller, and a leak when Python is, since
    then nothing ever takes the ownership Python gave up. Collectability is the thing asserted,
    with weakrefs and a forced collection, because the defect is an unreachable object rather
    than a large one.
    """
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)

    copy = function.Copy()
    assert copy.Setup(mol)
    assert copy.this.own(), "a Copy stays owned by Python"
    adapter = weakref.ref(copy)
    inner = weakref.ref(copy._function)
    del copy
    gc.collect()
    assert adapter() is None
    assert inner() is None, "the C++ EngineMolFunc has to go with its adapter"

    disowned = function.CreateCopy()
    assert not disowned.this.own(), "CreateCopy still hands ownership to its C++ caller"


def test_create_copy_hands_an_owned_copy_to_an_openeye_holder(butanol_conformers):
    """The disown path, exercised the way OpenEye exercises it.

    ``OEGenericFF`` is a C++ owner only when it is asked to be. ``Add`` defaults to
    ``own=False``, which borrows: a disowned copy added that way is deleted by nobody, so the
    test would leak and the caller-owns contract it exists to cover would go untested.
    ``Add(copy, True)`` completes the handover ``CreateCopy`` starts, and the weakrefs are what
    tell the two apart -- under the borrowing form both objects outlive the holder.
    """
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    xs = coordinates(mol)
    expected = function(xs)

    copy = function.CreateCopy()
    assert copy is not function
    assert not copy.this.own()
    adapter = weakref.ref(copy)
    inner = weakref.ref(copy._function)

    holder = oeff.OEGenericFF()
    assert holder.Add(copy, True)
    assert holder.Setup(mol)
    assert holder(xs) == pytest.approx(expected, abs=1e-9)

    del copy
    del holder
    gc.collect()
    assert adapter() is None, "the holder's destructor has to delete the copy it owns"
    assert inner() is None, "the C++ EngineMolFunc has to go with its adapter"


def test_an_evaluation_value_is_read_only(butanol_conformers):
    """House rule: a value handed to Python is read-only, private name or not."""
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    value = function._function.Evaluate(list(coordinates(mol)), True)
    with pytest.raises(AttributeError):
        value.energy = 0.0
    with pytest.raises(AttributeError):
        value.gradient = ()


def test_get_fcomponents_is_none(butanol_conformers):
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    assert function.GetFComponents(coordinates(mol)) is None


def test_setup_returns_false_for_a_molecule_it_cannot_prepare():
    flat = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(flat, "CCO")  # implicit hydrogens, no coordinates
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(flat) is False


# An exception raised inside the adapter cannot reach the caller when OpenEye's optimizer is
# the one calling: SWIG converts it to a Swig::DirectorMethodException, nothing in OpenEye
# catches that, and the process aborts. The adapter therefore never raises. These cases pin the
# four parts of that contract -- record, warn, sentinel energy, retrievable error -- on the
# direct call path, which is safe to exercise in-process; the optimizer path, which is the one
# that aborts without the fix, needs a child process and lives at the bottom of this file.
#
# The registry keeps every engine for the life of the process, so each adapter needs a name no
# earlier one used.
_SCRIPTED_NAMES = itertools.count()


class _ScriptedSession(oevibe.PythonEngineSession):
    """Harmonic springs back to the origin, except where the test scripts a failure.

    ``outcome(call_number)`` returns the exception that evaluation must raise, or ``None`` for
    one that succeeds, so a test can fail every evaluation, only the first, or one partway into
    an optimization, and can make successive failures distinguishable.
    """

    def __init__(self, outcome):
        super().__init__()
        self._outcome = outcome
        self.calls = 0

    def evaluate(self, coords, request):
        self.calls += 1
        failure = self._outcome(self.calls)
        if failure is not None:
            raise failure
        energy = 0.5 * float(np.sum(coords * coords))
        if request.gradient:
            return oevibe.Evaluation(energy=energy, gradient=np.array(coords))
        return oevibe.Evaluation(energy=energy)


class _ScriptedEngine(oevibe.PythonEngine):
    def __init__(self, name, outcome):
        super().__init__()
        self._name = name
        self._outcome = outcome
        self.session = None

    def GetName(self):
        return self._name

    def GetVersion(self):
        return "1.0"

    def GetMethods(self):
        capabilities = oevibe.MethodCapabilities()
        capabilities.method = oevibe.Method("scripted")
        return [capabilities]

    def CreateSession(self, system, settings):
        self.session = _ScriptedSession(self._outcome)
        return self.session


def scripted_adapter(mol, outcome):
    """:returns: An adapter set up on ``mol``, over an engine whose failures ``outcome`` scripts."""
    engine = _ScriptedEngine(f"as-oemolfunc-scripted-{next(_SCRIPTED_NAMES)}", outcome)
    oevibe.register_engine(engine)
    options = oevibe.CalculationOptions()
    options.SetEngine(engine.GetName())
    options.SetMethod(oevibe.Method("scripted"))
    function = oevibe.as_oemolfunc(options)
    assert function.Setup(mol)
    return function


def test_a_direct_call_on_a_failing_engine_warns_instead_of_raising(butanol_conformers):
    """The never-raise contract, on the path where raising would in fact have been safe.

    The direct caller could have caught an exception, but the two paths are one ``__call__``
    and a contract that held only when the caller was Python would be the more dangerous of the
    two to document. So both report a failure the same way.
    """
    mol = oechem.OEMol(butanol_conformers)
    function = scripted_adapter(mol, lambda call: RuntimeError("scripted-failure"))
    with pytest.warns(RuntimeWarning, match="scripted-failure"):
        energy = function(coordinates(mol))
    assert energy == float("inf")
    error = function.GetEvaluationError()
    assert isinstance(error, Exception)
    assert "scripted-failure" in str(error)


def test_a_failed_evaluation_accumulates_nothing_into_the_gradient(butanol_conformers):
    """A zero gradient contribution is half of what stops the optimizer; the sentinel energy
    alone would leave it with a direction to keep walking in."""
    mol = oechem.OEMol(butanol_conformers)
    function = scripted_adapter(mol, lambda call: RuntimeError("scripted-failure"))
    nvar = function.NumVar()
    prefilled = oechem.OEDoubleArray(nvar)
    for index in range(nvar):
        prefilled[index] = 5.0
    with pytest.warns(RuntimeWarning):
        energy = function(coordinates(mol), prefilled)
    assert energy == float("inf")
    assert [prefilled[i] for i in range(nvar)] == [5.0] * nvar


def test_the_first_failure_is_the_one_kept(butanol_conformers):
    """An optimizer keeps calling for a step or two after the failure, and those later calls
    fail for the same underlying reason; the first is the one that explains the run."""
    mol = oechem.OEMol(butanol_conformers)
    function = scripted_adapter(mol, lambda call: RuntimeError(f"failure-{call}"))
    xs = coordinates(mol)
    for _ in range(3):
        with pytest.warns(RuntimeWarning):
            function(xs)
    assert "failure-1" in str(function.GetEvaluationError())


def test_a_failed_evaluation_is_not_cached(butanol_conformers):
    """``EngineMolFunc`` caches the last geometry it evaluated, and a failure must not enter
    that cache: a later call at the same geometry would then return the sentinel silently,
    without even a warning to show for it."""
    mol = oechem.OEMol(butanol_conformers)
    function = scripted_adapter(
        mol, lambda call: RuntimeError("scripted-failure") if call == 1 else None
    )
    xs = coordinates(mol)
    with pytest.warns(RuntimeWarning):
        assert function(xs) == float("inf")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        energy = function(xs)
    assert not caught
    assert np.isfinite(energy)


def test_a_warning_filter_that_raises_cannot_resurrect_the_abort(butanol_conformers):
    """Under ``-W error`` ``warnings.warn`` raises, and a raise here is the abort this whole
    path exists to prevent, so the warning is itself guarded. The record survives it."""
    mol = oechem.OEMol(butanol_conformers)
    function = scripted_adapter(mol, lambda call: RuntimeError("scripted-failure"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energy = function(coordinates(mol))
    assert energy == float("inf")
    assert "scripted-failure" in str(function.GetEvaluationError())


def test_a_successful_adapter_reports_no_evaluation_error(butanol_conformers):
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    assert function.GetEvaluationError() is None
    function(coordinates(mol))
    assert function.GetEvaluationError() is None


def test_two_threads_interleave_inside_the_engine(butanol_conformers):
    """Spike S2 ran single-threaded; the settled decision requires threaded director coverage.

    The rendezvous sits inside the *engine's* first evaluation, not before the optimizer calls,
    so both optimizer runs have to be in flight at the same time for either to get past it. A
    barrier placed before the two calls would be satisfied by running them back to back; this
    one times out if anything serializes the pair. Both directors are exercised at once: the
    OpenEye optimizer upcalls as_oemolfunc, which calls through to a Python engine.
    """
    barrier = threading.Barrier(2, timeout=60)

    class RendezvousSession(oevibe.PythonEngineSession):
        def __init__(self):
            super().__init__()
            self.reference = None
            self.waited = False

        def evaluate(self, coords, request):
            if not self.waited:
                self.waited = True
                barrier.wait()
            if self.reference is None:
                self.reference = coords.copy()
            delta = coords - self.reference
            energy = 0.5 * 10.0 * float(np.sum(delta * delta))
            if request.gradient:
                return oevibe.Evaluation(energy=energy, gradient=10.0 * delta)
            return oevibe.Evaluation(energy=energy)

    class RendezvousEngine(oevibe.PythonEngine):
        def GetName(self):
            return "as-oemolfunc-threads"

        def GetVersion(self):
            return "1.0"

        def GetMethods(self):
            capabilities = oevibe.MethodCapabilities()
            capabilities.method = oevibe.Method("spring")
            return [capabilities]

        def CreateSession(self, system, settings):
            return RendezvousSession()

    oevibe.register_engine(RendezvousEngine(), replace=True)
    options = oevibe.CalculationOptions()
    options.SetEngine("as-oemolfunc-threads")
    options.SetMethod(oevibe.Method("spring"))

    molecules = [oechem.OEMol(butanol_conformers) for _ in range(2)]
    functions = [oevibe.as_oemolfunc(options) for _ in range(2)]
    for function, mol in zip(functions, molecules, strict=True):
        assert function.Setup(mol)
    minima: list = [None, None]
    failures: list = []

    def run(slot):
        try:
            xs = coordinates(molecules[slot])
            out = oechem.OEDoubleArray(functions[slot].NumVar())
            minima[slot] = oeff.OEBFGSOpt()(functions[slot], xs, out)
        except Exception as error:  # noqa: BLE001 - reported below, not swallowed
            failures.append(error)

    threads = [threading.Thread(target=run, args=(slot,)) for slot in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not failures, failures
    assert not any(thread.is_alive() for thread in threads), "a thread never finished"
    assert all(minimum is not None for minimum in minima)
    assert minima[0] == pytest.approx(minima[1], abs=1e-6)


@pytest.mark.usefixtures("tblite_library")
def test_a_runtime_loaded_engine_also_works(butanol_conformers):
    """The point of the adapter: OpenEye's optimizer driving a semiempirical engine."""
    mol = oechem.OEMol(butanol_conformers)
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("gfn2-xtb"))
    function = oevibe.as_oemolfunc(options)
    assert function.Setup(mol)
    xs = coordinates(mol)
    start = function(xs)
    out = oechem.OEDoubleArray(function.NumVar())
    assert oeff.OEBFGSOpt()(function, xs, out) <= start + 1e-6


# A proxy class's __module__ follows how its OpenEye module was imported, and which form a
# process used first is global state the adapter has no say in: ``from openeye import oeff``
# yields classes reporting ``oeff``, ``import openeye.oeff`` yields ``openeye.oeff``. Every test
# above uses the first form only, which is exactly the blind spot that let two MolFuncEngine
# defects through. The form is a property of the interpreter, so each one needs its own process.
_IMPORT_FORMS = (
    pytest.param("from openeye import oeff", "oeff", id="from-openeye-import"),
    pytest.param(
        "import openeye.oeff as oeff", "openeye.oeff", id="import-openeye-dot"
    ),
)

_IMPORT_FORM_ADAPTER_CHILD = """
{import_form}
from openeye import oechem, oeomega

import oevibe

print("MODULE", oeff.OEMolFunc1.__module__)

mol = oechem.OEMol()
assert oechem.OESmilesToMol(mol, "CCCCO")
build_options = oeomega.OEOmegaOptions()
build_options.SetMaxConfs(1)
assert oeomega.OEOmega(build_options).Build(mol) == oeomega.OEOmegaReturnCode_Success

options = oevibe.CalculationOptions()
options.SetMethod(oevibe.Method("mmff94s"))
function = oevibe.as_oemolfunc(options)
assert function.Setup(mol)

native = oeff.OEMMFF(True)
assert native.PrepMol(mol, False, True)
assert native.Setup(mol)

xs = oechem.OEDoubleArray(3 * mol.GetMaxAtomIdx())
mol.GetCoords(xs)
print("ENERGIES", repr(function(xs)), repr(native(xs)))

mine_out = oechem.OEDoubleArray(function.NumVar())
native_out = oechem.OEDoubleArray(native.NumVar())
print(
    "MINIMA",
    repr(oeff.OEBFGSOpt()(function, xs, mine_out)),
    repr(oeff.OEBFGSOpt()(native, xs, native_out)),
)
"""


def _child_environment():
    """:returns: ``os.environ`` with this ``oevibe`` on ``PYTHONPATH`` for a child process."""
    environment = dict(os.environ)
    package_root = str(Path(oevibe.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_root, *filter(None, [environment.get("PYTHONPATH")])]
    )
    return environment


def _run_child(source, timeout=300):
    """:returns: The stdout of ``source`` run in a fresh interpreter; fails on a crash.

    ``timeout`` bounds the run, so a child that never terminates fails the test rather than
    hanging the suite.
    """
    child = subprocess.run(
        [sys.executable, "-c", source],
        env=_child_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    return child.stdout


@pytest.mark.parametrize(("import_form", "expected_module"), _IMPORT_FORMS)
def test_the_adapter_works_under_either_import_form(import_form, expected_module):
    """``import openeye.oeff`` is as ordinary as ``from openeye import oeff``.

    The adapter subclasses whichever ``OEMolFunc1`` the process already has, so OpenEye's
    optimizer must accept the subclass under both forms and reach the same minimum. The energy
    and the minimum are both matched against native MMFF inside the child, because a subclass
    OpenEye silently declined to call would not produce either.
    """
    lines = _run_child(_IMPORT_FORM_ADAPTER_CHILD.format(import_form=import_form))
    reported = next(line for line in lines.splitlines() if line.startswith("MODULE "))
    assert reported == f"MODULE {expected_module}", (
        "OpenEye no longer varies __module__ with the import form, so this case no longer "
        f"covers what it claims to: {reported}"
    )
    energies = next(line for line in lines.splitlines() if line.startswith("ENERGIES "))
    _, mine, theirs = energies.split()
    assert float(mine) == pytest.approx(float(theirs), abs=1e-9)
    minima = next(line for line in lines.splitlines() if line.startswith("MINIMA "))
    _, mine, theirs = minima.split()
    assert float(mine) == pytest.approx(float(theirs), abs=1e-4)


# The optimizer half of the never-raise contract. Before the adapter grew it, this scenario did
# not fail -- it killed the interpreter with "libc++abi: terminating due to uncaught exception
# of type Swig::DirectorMethodException", taking the rest of the suite with it. A child process
# is therefore the only way to assert on it: the parent sees an exit status instead of dying,
# and ``_run_child``'s timeout is the bound that makes "the optimizer must not hang" an
# assertion rather than a hope.
_FAILING_ENGINE_CHILD = """
import time
import warnings

from openeye import oechem, oeff, oeomega

import oevibe


class FailingSession(oevibe.PythonEngineSession):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def evaluate(self, coords, request):
        self.calls += 1
        if self.calls > {fail_after}:
            raise RuntimeError("sentinel-engine-failure")
        energy = 0.5 * float((coords * coords).sum())
        if request.gradient:
            return oevibe.Evaluation(energy=energy, gradient=coords.copy())
        return oevibe.Evaluation(energy=energy)


class FailingEngine(oevibe.PythonEngine):
    def __init__(self):
        super().__init__()
        self.session = None

    def GetName(self):
        return "abort-probe-engine"

    def GetVersion(self):
        return "1.0"

    def GetMethods(self):
        capabilities = oevibe.MethodCapabilities()
        capabilities.method = oevibe.Method("abort-probe")
        return [capabilities]

    def CreateSession(self, system, settings):
        self.session = FailingSession()
        return self.session


engine = FailingEngine()
oevibe.register_engine(engine)

mol = oechem.OEMol()
assert oechem.OESmilesToMol(mol, "CCCCO")
build = oeomega.OEOmegaOptions()
build.SetMaxConfs(1)
assert oeomega.OEOmega(build).Build(mol) == oeomega.OEOmegaReturnCode_Success

options = oevibe.CalculationOptions()
options.SetEngine("abort-probe-engine")
options.SetMethod(oevibe.Method("abort-probe"))
function = oevibe.as_oemolfunc(options)
assert function.Setup(mol)

nvar = function.NumVar()
xs = oechem.OEDoubleArray(nvar)
mol.GetCoords(xs)
out = oechem.OEDoubleArray(nvar)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    began = time.monotonic()
    result = oeff.OEBFGSOpt()(function, xs, out)
    seconds = time.monotonic() - began

print("CALLS", engine.session.calls)
print("SECONDS", round(seconds, 3))
print("RESULT", repr(result))
print("WARNINGS", len(caught))
print("FINITE_GEOMETRY", all(abs(out[i]) < 1e30 for i in range(nvar)))
print("ERROR", repr(str(function.GetEvaluationError())))
print("WARNING_TEXT", repr(str(caught[0].message)) if caught else "none")
print("ALIVE")
"""


@pytest.mark.parametrize("fail_after", [0, 5])
def test_a_failing_engine_under_oebfgsopt_does_not_kill_the_process(fail_after):
    """The abort this contract exists to prevent, at two points in a minimization.

    ``fail_after=0`` fails the very first evaluation, so the optimizer never gets a usable
    gradient; ``fail_after=5`` fails one partway in, when it has a BFGS history and a search
    direction it would otherwise keep following. Both must stop promptly -- the ``_run_child``
    timeout is what makes that an assertion -- warn, and leave the failure retrievable.
    """
    lines = _run_child(_FAILING_ENGINE_CHILD.format(fail_after=fail_after), timeout=180)
    reported = dict(
        line.split(" ", 1) for line in lines.splitlines() if line.count(" ") >= 1
    )
    assert "ALIVE" in lines.splitlines(), lines

    assert reported["RESULT"] == "inf", (
        "a failed evaluation reports an infinite energy; nan and -inf are accepted by the "
        f"line search instead of rejected: {reported['RESULT']}"
    )
    assert int(reported["WARNINGS"]) >= 1, "a failure is never silent"
    assert "sentinel-engine-failure" in reported["ERROR"]
    assert "sentinel-engine-failure" in reported["WARNING_TEXT"]
    assert reported["FINITE_GEOMETRY"] == "True"

    # A run that thrashed on the sentinel instead of stopping would be far past this. Measured:
    # 1 evaluation in 0.02 s for fail_after=0, and 8 in 0.002 s for fail_after=5, the three
    # beyond the fifth being the line search rejecting the infinite step and giving up.
    assert int(reported["CALLS"]) <= fail_after + 5, reported["CALLS"]
    assert float(reported["SECONDS"]) < 30.0, reported["SECONDS"]
