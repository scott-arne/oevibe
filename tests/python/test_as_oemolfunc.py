"""OEVibe as an OpenEye potential, for OpenEye's own optimizers (spec 8.8)."""

import os
import subprocess
import sys
import threading
import types
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


def test_create_copy_gives_an_independent_function(butanol_conformers):
    mol = oechem.OEMol(butanol_conformers)
    function = oevibe.as_oemolfunc(mmff_options())
    assert function.Setup(mol)
    copy = function.CreateCopy()
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
    copy = function.CreateCopy()
    assert copy.Setup(mol)
    assert copy(xs) == pytest.approx(before, abs=1e-12)


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


def _run_child(source):
    """:returns: The stdout of ``source`` run in a fresh interpreter; fails on a crash."""
    child = subprocess.run(
        [sys.executable, "-c", source],
        env=_child_environment(),
        capture_output=True,
        text=True,
        timeout=300,
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
