"""OpenEye potentials as OEVibe engines (spec 8.8)."""

import gc
import os
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from openeye import oechem, oeff

import oevibe
from oevibe import oevibe as _proxy
from oevibe.oemolfunc import _as_molfunc1
from oevibe.oevibe import _MolFuncEngine

METHOD = "mmff94s-molfunc"
GRADIENT_ONLY_METHOD = "harmonic-molfunc"

_OEMOLFUNC_SUBCLASS_SWEEP_CHILD = """
import importlib
import pkgutil

import openeye
from openeye import oeff

strays = []
for info in pkgutil.iter_modules(openeye.__path__):
    try:
        module = importlib.import_module("openeye." + info.name)
    except ImportError:
        continue
    if module is oeff:
        continue
    for name in dir(module):
        candidate = getattr(module, name)
        if isinstance(candidate, type) and issubclass(candidate, oeff.OEMolFunc1):
            strays.append("openeye." + info.name + "." + name)
print(("STRAYS " + " ".join(strays)) if strays else "NO STRAYS")
"""

# Importing a second OpenEye module that wraps OEMolPotential::OEMolFunc1 rebinds the proxy class
# SWIG hands back for it process-wide, so both children below have to run in their own process.
# openeye.oequacpac is the partial-charge toolkit: anyone assigning charges imports it.
_OEQUACPAC_ENERGY_CHILD = """
from openeye import oechem, oeff, oeomega, oequacpac  # noqa: F401

import oevibe

mol = oechem.OEMol()
assert oechem.OESmilesToMol(mol, "CCCCO")
options = oeomega.OEOmegaOptions()
options.SetMaxConfs(1)
assert oeomega.OEOmega(options).Build(mol) == oeomega.OEOmegaReturnCode_Success

engine = oevibe.MolFuncEngine(
    "oeff-molfunc-quacpac", "1.0", oevibe.Method("mmff94s-quacpac"), oeff.OEMMFF(True)
)
oevibe.register_engine(engine, replace=True)

through_molfunc = oevibe.CalculationOptions()
through_molfunc.SetMethod(oevibe.Method("mmff94s-quacpac"))
through_oeff = oevibe.CalculationOptions()
through_oeff.SetMethod(oevibe.Method("mmff94s"))
mine = oevibe.Calculation(through_molfunc).Energy(oechem.OEMol(mol)).Get(0)
theirs = oevibe.Calculation(through_oeff).Energy(oechem.OEMol(mol)).Get(0)
assert mine.GetStatus() == oevibe.ConformerStatus.OK, mine.GetMessage()
print("ENERGIES", repr(mine.GetEnergy()), repr(theirs.GetEnergy()))
"""

_OEQUACPAC_UPCAST_CHILD = """
from openeye import oeff, oequacpac  # noqa: F401

import oevibe

holder = oeff.OEGenericFF()
assert holder.Add(oeff.OEMMFF(True))
upcast = holder.GetComponents()[0]
print("EXACT_TYPE", type(upcast).__module__ + "." + type(upcast).__name__)
try:
    oevibe.MolFuncEngine("quacpac-upcast", "1.0", oevibe.Method("nope"), upcast)
except TypeError as error:
    print("REJECTED", " ".join(str(error).split()))
else:
    print("ACCEPTED")
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


@pytest.fixture
def mmff_engine():
    engine = oevibe.MolFuncEngine(
        "oeff-molfunc", "1.0", oevibe.Method(METHOD), oeff.OEMMFF(True)
    )
    oevibe.register_engine(engine, replace=True)
    return engine


@pytest.fixture
def harmonic_engine():
    """An OpenEye potential that is an ``OEMolFunc1`` but not an ``OEMolFunc2``."""
    engine = oevibe.MolFuncEngine(
        "oeff-molfunc-harmonic",
        "1.0",
        oevibe.Method(GRADIENT_ONLY_METHOD),
        oeff.OEHarmonicPotential(10.0, 0.0),
    )
    oevibe.register_engine(engine, replace=True)
    return engine


def molfunc_options():
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method(METHOD))
    return options


def harmonic_options():
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method(GRADIENT_ONLY_METHOD))
    return options


def oeff_options():
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94s"))
    return options


def test_the_engine_resolves_without_naming_it(butanol_conformers, mmff_engine):
    """Only one engine declares the method, so ResolveForMethod must find it (spec 5.3)."""
    calculation = oevibe.Calculation(molfunc_options())
    assert calculation.GetEngineName() == "oeff-molfunc"


def test_single_point_matches_the_builtin_oeff_engine(butanol_conformers, mmff_engine):
    through_molfunc = oevibe.Calculation(molfunc_options()).Energy(
        oechem.OEMol(butanol_conformers)
    )
    through_oeff = oevibe.Calculation(oeff_options()).Energy(
        oechem.OEMol(butanol_conformers)
    )
    for mine, theirs in zip(through_molfunc, through_oeff, strict=True):
        assert mine.GetStatus() == oevibe.ConformerStatus.OK, mine.GetMessage()
        assert mine.GetEnergy() == pytest.approx(theirs.GetEnergy(), abs=1e-9)


def test_optimization_matches_the_builtin_oeff_engine(butanol_conformers, mmff_engine):
    mine = oevibe.Calculation(molfunc_options()).Optimize(
        oechem.OEMol(butanol_conformers), oevibe.OptimizerOptions()
    )
    theirs = oevibe.Calculation(oeff_options()).Optimize(
        oechem.OEMol(butanol_conformers), oevibe.OptimizerOptions()
    )
    for from_molfunc, from_oeff in zip(mine, theirs, strict=True):
        assert from_molfunc.GetEnergy() == pytest.approx(
            from_oeff.GetEnergy(), abs=1e-6
        )


def test_the_engine_survives_the_caller_dropping_the_potential(butanol_conformers):
    """The C++ constructor takes a CreateCopy() of the prototype, so this must still work."""
    potential = oeff.OEMMFF(True)
    engine = oevibe.MolFuncEngine(
        "oeff-molfunc-dropped", "1.0", oevibe.Method("mmff94s-dropped"), potential
    )
    oevibe.register_engine(engine, replace=True)
    del potential
    gc.collect()
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94s-dropped"))
    results = oevibe.Calculation(options).Energy(oechem.OEMol(butanol_conformers))
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.OK


def test_the_declared_capabilities_follow_spec_8_8(mmff_engine):
    methods = list(mmff_engine.GetMethods())
    assert len(methods) == 1
    capabilities = methods[0]
    assert capabilities.method.GetName() == METHOD
    assert capabilities.analytic_gradient
    assert capabilities.analytic_hessian  # OEMMFF is an OEMolFunc2
    assert capabilities.charge_from_topology
    assert not capabilities.requires_partial_charges  # the factory's default
    assert list(capabilities.solvent_models) == []
    assert capabilities.properties == 0


def test_requires_partial_charges_reaches_the_capability():
    """The factory argument is the only way this capability can ever be true."""
    engine = oevibe.MolFuncEngine(
        "oeff-molfunc-charges",
        "1.0",
        oevibe.Method("mmff94s-charges"),
        oeff.OEMMFF(True),
        requires_partial_charges=True,
    )
    capabilities = next(iter(engine.GetMethods()))
    assert capabilities.requires_partial_charges


def test_a_gradient_only_potential_declares_no_analytic_hessian(harmonic_engine):
    """``OEHarmonicPotential`` is one of the 14 ``oeff`` potentials that are an ``OEMolFunc1``
    without being an ``OEMolFunc2``, so the ``dynamic_cast`` behind ``analytic_hessian`` has to
    report false for it."""
    capabilities = next(iter(harmonic_engine.GetMethods()))
    assert capabilities.analytic_gradient
    assert not capabilities.analytic_hessian


def test_frequencies_on_a_gradient_only_potential_use_finite_differences(
    butanol_conformers, harmonic_engine
):
    """Asking a potential that is not an ``OEMolFunc2`` for a Hessian raises inside the session,
    so this only returns at all if the declared capability sent ``Frequencies`` down the
    finite-difference path instead.

    The numbers are all zero, and that is OpenEye's doing rather than OEVibe's: the restraint
    evaluates flat once ``CreateCopy()`` has been through it, where the same object set up
    directly returns 2.5 kcal/mol for a 0.5 Angstrom displacement at ``kc=10``. What this test
    pins is the path, which the shape of the result still shows.
    """
    results = oevibe.Calculation(harmonic_options()).Frequencies(
        oechem.OEMol(butanol_conformers), oevibe.ThermoOptions()
    )
    result = results.Get(0)
    assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
    frequencies = np.array(list(result.GetThermochemistry().frequencies))
    assert len(frequencies) == 3 * butanol_conformers.NumAtoms() - 6


def test_a_non_potential_is_rejected():
    with pytest.raises(TypeError):
        oevibe.MolFuncEngine("bad", "1.0", oevibe.Method("nope"), object())


def test_a_python_defined_potential_is_rejected():
    """Spec 8.8 excludes Python-defined OEMolFunc1 subclasses from v1."""

    class MyPotential(oeff.OEMolFunc1):
        def __init__(self):
            oeff.OEMolFunc1.__init__(self)

        def CreateCopy(self):
            copy = MyPotential()
            copy.__disown__()
            return copy

        def Setup(self, mol):
            return True

        def NumVar(self):
            return 0

        def GetFComponents(self, coords):
            return None

        def __call__(self, *args):
            return 0.0

    with pytest.raises(TypeError, match="PythonEngine"):
        oevibe.MolFuncEngine("py", "1.0", oevibe.Method("nope"), MyPotential())


def test_a_python_defined_potential_cannot_spoof_its_module():
    """``type.__module__`` is writable, so the exclusion cannot be a module-name test.

    A subclass that claims ``oeff`` used to reach the native constructor and die there with
    ``SWIG director pure virtual method called`` instead of the ``TypeError`` spec 8.8 promises.
    """

    class Spoofed(oeff.OEMolFunc1):
        __module__ = "oeff"

        def __init__(self):
            oeff.OEMolFunc1.__init__(self)

        def CreateCopy(self):
            copy = Spoofed()
            copy.__disown__()
            return copy

        def Setup(self, mol):
            return True

        def NumVar(self):
            return 0

        def GetFComponents(self, coords):
            return None

        def __call__(self, *args):
            return 0.0

    with pytest.raises(TypeError, match="PythonEngine"):
        oevibe.MolFuncEngine("spoofed", "1.0", oevibe.Method("nope"), Spoofed())


def test_a_python_subclass_of_an_openeye_potential_cannot_spoof_its_module():
    """The same spoof over a concrete potential used to be accepted outright.

    ``MolFuncEngine`` then wrapped native MMFF behaviour while the caller believed the subclass
    was in play, so the overridden method below has to stay uncalled.
    """
    calls = []

    class Spoofed(oeff.OEMMFF):
        __module__ = "oeff"

        def CreateCopy(self):
            calls.append("CreateCopy")
            raise AssertionError("the subclass must never be used as a potential")

    with pytest.raises(TypeError, match="PythonEngine"):
        oevibe.MolFuncEngine(
            "spoofed-mmff", "1.0", oevibe.Method("nope"), Spoofed(True)
        )
    assert calls == []


def test_a_python_subclass_cannot_spoof_the_name_of_a_real_potential():
    """Spoofing ``__name__`` as well still fails: ``oeff.OEMMFF`` is not this class."""

    class Spoofed(oeff.OEMMFF):
        __module__ = "oeff"

    # Assigned after the class body: a class-body ``__name__`` lands in the class dict, where
    # ``type.__name__`` does not read it.
    Spoofed.__name__ = "OEMMFF"

    with pytest.raises(TypeError, match="PythonEngine"):
        oevibe.MolFuncEngine(
            "spoofed-named", "1.0", oevibe.Method("nope"), Spoofed(True)
        )


def test_a_subclass_rebound_into_the_openeye_module_is_rejected(monkeypatch):
    """Re-exporting the subclass from ``openeye.oeff`` used to satisfy the identity check.

    Reading ``getattr(oeff, kind.__name__)`` on every call asked a mutable module what it
    exported *now*; assigning ``oeff.Reexported = Reexported`` made the answer yes, and
    ``MolFuncEngine`` then returned native MMFF energies while the override below never ran.
    The snapshot is taken before the rebinding here, which is the ordering the guard relies on.
    """
    calls = []

    class Reexported(oeff.OEMMFF):
        __module__ = "oeff"

        def CreateCopy(self):
            calls.append("CreateCopy")
            raise AssertionError("the subclass must never be used as a potential")

    # Warm the snapshot first; a test that rebound oeff before OEVibe had ever looked at it
    # would prove nothing about the mechanism.
    oevibe.MolFuncEngine(
        "snapshot-warmup", "1.0", oevibe.Method("mmff94s-warmup"), oeff.OEMMFF(True)
    )
    monkeypatch.setattr(oeff, "Reexported", Reexported, raising=False)
    assert oeff.Reexported is Reexported, "the rebinding itself must have taken effect"

    with pytest.raises(TypeError, match="PythonEngine"):
        oevibe.MolFuncEngine("rebound", "1.0", oevibe.Method("nope"), Reexported(True))
    assert calls == []


def test_the_potential_snapshot_covers_every_oeff_potential():
    """The snapshot stands in for ``openeye.oeff`` itself, so it has to hold every potential.

    Re-derive the enumeration (69 classes in the pinned OpenEye) and fail on any difference: a
    class the snapshot missed would be a legitimate potential ``MolFuncEngine`` now refuses.
    """
    from oevibe.oemolfunc import _openeye_potential_classes

    enumerated = {
        candidate
        for candidate in (getattr(oeff, name) for name in dir(oeff))
        if isinstance(candidate, type) and issubclass(candidate, oeff.OEMolFunc1)
    }
    assert enumerated == set(_openeye_potential_classes())


def test_no_other_openeye_module_exports_an_oeff_potential():
    """Snapshotting ``oeff`` alone is only complete because nothing else subclasses its base.

    ``oeet``, ``oequacpac`` and ``oeshape`` each export an ``OEMolFunc1`` of their own, but a
    *distinct* class: an instance of one fails the ``isinstance`` gate in ``_as_molfunc1``
    before the snapshot is ever consulted.

    The sweep runs in a child process because importing an OpenEye module rebinds the proxy
    class SWIG hands back for ``OEMolPotential::OEMolFunc1*`` process-wide -- after
    ``import openeye.oequacpac``, ``oeff.OEGenericFF.GetComponents()`` returns
    ``oequacpac.OEMolFunc1`` proxies, which the rest of this file would then fail on.
    """
    child = subprocess.run(
        [sys.executable, "-c", _OEMOLFUNC_SUBCLASS_SWEEP_CHILD],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip().splitlines()[-1] == "NO STRAYS", child.stdout


def test_a_potential_still_works_when_oequacpac_was_imported_first():
    """Every OpenEye module defines its own ``OEMolFunc1`` proxy and the last import wins.

    After ``import openeye.oequacpac``, ``oeff.OEGenericFF.GetComponents()`` hands back
    ``oequacpac.OEMolFunc1`` proxies, which a guard that compares against ``oeff.OEMolFunc1``
    alone turns away -- so the documented ``MolFuncEngine(oeff.OEMMFF(True))`` failed for anyone
    who had also imported the partial-charge toolkit. The energy is matched against the built-in
    ``oeff`` engine inside the child, because a guard that admitted an unadjusted pointer would
    not produce one.
    """
    report = _run_child(_OEQUACPAC_ENERGY_CHILD).strip().splitlines()[-1]
    label, mine, theirs = report.split()
    assert label == "ENERGIES", report
    assert float(mine) == pytest.approx(float(theirs), abs=1e-9)


def test_an_upcast_is_still_rejected_when_oequacpac_was_imported_first():
    """The upcast guard has to recognise any OpenEye module's abstract interface, not oeff's.

    The component below is an ``oequacpac.OEMolFunc1``, which the ``oeff``-only identity tests
    did not know. It was still refused, but as something that is not a potential at all, and a
    caller cannot act on that: the fix has to say what is actually wrong with it.
    """
    lines = _run_child(_OEQUACPAC_UPCAST_CHILD).strip().splitlines()
    assert "EXACT_TYPE oequacpac.OEMolFunc1" in lines, lines
    rejection = lines[-1]
    assert rejection.startswith("REJECTED "), lines
    assert "upcast proxy of exact type oequacpac.OEMolFunc1" in rejection, rejection
    assert "Python-defined" not in rejection, rejection


def test_a_python_subclass_laundered_through_the_openeye_upcast_is_rejected():
    """OpenEye's own upcast erases the subclass with nothing spoofed at all.

    ``OEGenericFF.GetComponents()`` hands back a proxy whose exact type is ``oeff.OEMolFunc1``,
    a class ``openeye.oeff`` genuinely exports, so the identity check above passes it. Accepting
    it ran native MMFF while the caller believed the subclass was in play -- a silent wrong
    answer, which is why the overridden method below has to stay uncalled.
    """
    calls = []

    class Laundered(oeff.OEMMFF):
        def CreateCopy(self):
            calls.append("CreateCopy")
            raise AssertionError("the subclass must never be used as a potential")

    potential = Laundered(True)
    holder = oeff.OEGenericFF()
    assert holder.Add(potential)
    upcast = holder.GetComponents()[0]
    assert type(upcast) is oeff.OEMolFunc1

    with pytest.raises(TypeError, match="upcast"):
        oevibe.MolFuncEngine("laundered", "1.0", oevibe.Method("nope"), upcast)
    assert calls == []


def test_a_native_potential_laundered_the_same_way_is_rejected_just_as_hard():
    """OEVibe refuses the upcast shape, not the subclass.

    Nothing visible from Python separates this component from the laundered subclass above, so
    an ordinary ``oeff.OEMMFF`` behind the same upcast is refused too. The potential itself is
    still accepted -- that is what every fixture in this file passes.
    """
    potential = oeff.OEMMFF(True)
    holder = oeff.OEGenericFF()
    assert holder.Add(potential)
    upcast = holder.GetComponents()[0]
    assert type(upcast) is oeff.OEMolFunc1

    with pytest.raises(TypeError, match="upcast"):
        oevibe.MolFuncEngine("native-laundered", "1.0", oevibe.Method("nope"), upcast)


def test_an_upcast_to_oemolfunc2_is_rejected_as_well():
    """``OEGenericFF2`` returns its components as ``oeff.OEMolFunc2``, so rejecting the one
    interface name ``OEMolFunc1`` would have left the same hole one class higher."""
    potential = oeff.OEMMFF(True)
    holder = oeff.OEGenericFF2()
    assert holder.Add(potential)
    upcast = holder.GetComponents()[0]
    assert type(upcast) is oeff.OEMolFunc2

    with pytest.raises(TypeError, match="upcast"):
        oevibe.MolFuncEngine("molfunc2-laundered", "1.0", oevibe.Method("nope"), upcast)


def test_the_raw_constructor_rejects_an_unadjusted_proxy():
    """The pointer inside an OEMMFF proxy is 48 bytes away from its OEMolFunc1 subobject."""
    from oevibe.oevibe import _MolFuncEngine

    with pytest.raises(TypeError):
        _MolFuncEngine("raw", "1.0", oevibe.Method("nope"), oeff.OEMMFF(True))


def test_frequencies_use_the_analytic_hessian(butanol_conformers, mmff_engine):
    results = oevibe.Calculation(molfunc_options()).Frequencies(
        oechem.OEMol(butanol_conformers), oevibe.ThermoOptions()
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.OK, results.Get(
        0
    ).GetMessage()
    frequencies = np.array(list(results.Get(0).GetThermochemistry().frequencies))
    assert len(frequencies) == 3 * butanol_conformers.NumAtoms() - 6


# ---------------------------------------------------------------------------
# The Register keepalive (spec 8.7) and the one engine that is exempt from it
# ---------------------------------------------------------------------------
# Registering from Python retains the engine for the life of the process, because a Python
# engine's C++ director only borrows its Python object. A _MolFuncEngine borrows nothing: it is
# a native C++ engine holding a C++ copy of the potential, so retaining it would pin the engine
# and its copy of the OpenEye potential forever, once per registration.


def test_replacing_a_molfunc_engine_does_not_retain_it():
    """Registering over a name five times used to grow the keepalive list by five."""
    before = len(_proxy._REGISTERED_ENGINES)
    references = []
    for _ in range(5):
        engine = oevibe.MolFuncEngine(
            "oeff-molfunc-churn",
            "1.0",
            oevibe.Method("mmff94s-churn"),
            oeff.OEMMFF(True),
        )
        references.append(weakref.ref(engine))
        oevibe.register_engine(engine, replace=True)
        del engine
    assert len(_proxy._REGISTERED_ENGINES) == before
    gc.collect()
    assert [reference() for reference in references] == [None] * 5


def test_a_calculation_keeps_the_engine_it_resolved_through_a_replacement(
    butanol_conformers,
):
    """Dropping the keepalive must not drop the engine: the registry and every Calculation that
    resolved it hold C++ ``shared_ptr``s, which is the only thing standing between this change
    and a use-after-free."""
    name = "oeff-molfunc-replaced"
    method = "mmff94s-replaced"
    engine = oevibe.MolFuncEngine(name, "1.0", oevibe.Method(method), oeff.OEMMFF(True))
    oevibe.register_engine(engine, replace=True)
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method(method))
    calculation = oevibe.Calculation(options)
    reference = weakref.ref(engine)
    del engine
    gc.collect()
    assert reference() is None, "nothing should still pin the Python proxy"

    # The harmonic potential evaluates flat, so an energy of zero here would mean the
    # calculation had followed the registry to the replacement instead of holding its own.
    oevibe.register_engine(
        oevibe.MolFuncEngine(
            name, "1.0", oevibe.Method(method), oeff.OEHarmonicPotential(10.0, 0.0)
        ),
        replace=True,
    )
    gc.collect()
    result = calculation.Energy(oechem.OEMol(butanol_conformers)).Get(0)
    assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
    expected = (
        oevibe.Calculation(oeff_options())
        .Energy(oechem.OEMol(butanol_conformers))
        .Get(0)
    )
    assert result.GetEnergy() == pytest.approx(expected.GetEnergy(), abs=1e-9)


def test_a_python_engine_is_still_retained():
    """The exemption is for the native engine only; a director still borrows its Python self."""

    class Noop(oevibe.PythonEngine):
        def GetName(self):
            return "molfunc-keepalive-control"

        def GetVersion(self):
            return "1.0"

        def GetMethods(self):
            capabilities = oevibe.MethodCapabilities()
            capabilities.method = oevibe.Method("molfunc-keepalive-control")
            return [capabilities]

        def CreateSession(self, system, settings):
            raise NotImplementedError

    before = len(_proxy._REGISTERED_ENGINES)
    oevibe.register_engine(Noop(), replace=True)
    assert len(_proxy._REGISTERED_ENGINES) == before + 1


def test_a_python_subclass_of_the_native_engine_is_still_retained():
    """``isinstance`` would be the wrong test: a subclass carries Python state that the C++ side
    has no reference to, so retention stays the default for everything but the exact type."""

    class Subclassed(_MolFuncEngine):
        def __init__(self, name, method, potential):
            # _holder must stay referenced across the constructor; see oemolfunc.MolFuncEngine.
            upcast, _holder = _as_molfunc1(potential)
            super().__init__(name, "1.0", oevibe.Method(method), upcast)

    before = len(_proxy._REGISTERED_ENGINES)
    oevibe.register_engine(
        Subclassed("molfunc-subclass", "mmff94s-subclass", oeff.OEMMFF(True)),
        replace=True,
    )
    assert len(_proxy._REGISTERED_ENGINES) == before + 1


def test_the_abstract_interface_list_still_matches_the_installed_openeye():
    """``_ABSTRACT_POTENTIAL_INTERFACES`` is a literal tuple, so an OpenEye release that adds an
    abstract potential interface would silently reopen the upcast-laundering hole. Re-derive the
    set here and fail on drift.

    Constructing all 69 subclasses is far too expensive for the guard itself, which runs on every
    legitimate call, but it costs about 20 ms once in a test. The two abstract signals are
    distinct from the ``TypeError`` a concrete class raises about its arguments.
    """
    from oevibe.oemolfunc import _ABSTRACT_POTENTIAL_INTERFACES

    abstract = set()
    for name in dir(oeff):
        candidate = getattr(oeff, name)
        if not isinstance(candidate, type) or not issubclass(
            candidate, oeff.OEMolFunc1
        ):
            continue
        try:
            candidate()
        except AttributeError as error:
            if "class is abstract" in str(error):
                abstract.add(name)
        except RuntimeError as error:
            if "abstract class or protected constructor" in str(error):
                abstract.add(name)
        except TypeError:
            pass  # A concrete class complaining about its constructor arguments.

    assert abstract == set(_ABSTRACT_POTENTIAL_INTERFACES)
