"""SWIG directors: a Python subclass of the C++ Engine drives a real Calculation.

These tests use the raw wrapped classes. The ergonomic ``oevibe.PythonEngine`` base classes that
hide them are tested in ``test_python_engines.py``.
"""

import gc
import threading

import pytest
from openeye import oechem

import oevibe

LIVE_SESSIONS = []


class SpringSession(oevibe.EngineSession):
    """Harmonic springs pulling every coordinate back to this session's starting geometry.

    The reference is captured on the FIRST Evaluate, from that call's coordinates, which is the
    settled rule for every engine (spec 6.4 order independence). Capturing it at construction
    from conformer 0 would also make the per-conformer assertions below false, because
    ``Calculation`` gives each session its own conformer's coordinates.
    """

    def __init__(self):
        super().__init__()
        self.reference = None
        self.evaluations = 0
        LIVE_SESSIONS.append(id(self))

    def __del__(self):
        try:
            LIVE_SESSIONS.remove(id(self))
        except ValueError:
            pass

    def Evaluate(self, coords, request):
        self.evaluations += 1
        if self.reference is None:
            self.reference = [coords[index] for index in range(len(coords))]
        evaluation = oevibe.Evaluation()
        energy = 0.0
        gradient = []
        for index, reference in enumerate(self.reference):
            delta = coords[index] - reference
            energy += 0.5 * 10.0 * delta * delta
            gradient.append(10.0 * delta)
        evaluation.energy = energy
        if request.gradient:
            evaluation.gradient = gradient
        return evaluation


class SpringEngine(oevibe.Engine):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.sessions_created = 0

    def GetName(self):
        return self.name

    def GetVersion(self):
        return "1.0"

    def IsAvailable(self):
        return True

    def GetUnavailableReason(self):
        return ""

    def GetMethods(self):
        capabilities = oevibe.MethodCapabilities()
        capabilities.method = oevibe.Method("spring")
        return [capabilities]

    def CreateSession(self, system, settings):
        self.sessions_created += 1
        return SpringSession()


class RaisingEngine(SpringEngine):
    def __init__(self, name, error):
        super().__init__(name)
        self.error = error

    def CreateSession(self, system, settings):
        raise self.error


def register(engine):
    """Register ``engine`` and keep it alive for the process, as spec 8.7 requires."""
    oevibe.EngineRegistry.Instance().Register(engine, True)
    _REGISTERED.append(engine)
    return engine


_REGISTERED: list = []


def spring_options(name):
    options = oevibe.CalculationOptions()
    options.SetEngine(name)
    options.SetMethod(oevibe.Method("spring"))
    return options


def test_a_python_engine_answers_a_calculation(butanol_conformers):
    engine = register(SpringEngine("director-energy"))
    results = oevibe.Calculation(spring_options("director-energy")).Energy(
        butanol_conformers
    )
    assert len(results) == butanol_conformers.NumConfs()
    for result in results:
        assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
        assert result.GetEnergy() == pytest.approx(0.0, abs=1e-9)
    assert engine.sessions_created == butanol_conformers.NumConfs()


def test_a_session_outlives_every_python_reference(butanol_conformers):
    """``CreateSession``'s local disappears at once; C++ optimizes against it for many steps."""
    register(SpringEngine("director-optimize"))
    results = oevibe.Calculation(spring_options("director-optimize")).Optimize(
        butanol_conformers, oevibe.OptimizerOptions()
    )
    for result in results:
        assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
        assert result.GetOptimizationStats().gradient_evaluations >= 1


def test_sessions_are_released_when_the_calculation_ends(butanol_conformers):
    """A director that keeps its proxy alive and a proxy that keeps the director alive would
    make every session immortal; this is the test that catches that cycle."""
    register(SpringEngine("director-lifetime"))
    LIVE_SESSIONS.clear()
    oevibe.Calculation(spring_options("director-lifetime")).Optimize(
        butanol_conformers, oevibe.OptimizerOptions()
    )
    gc.collect()
    assert LIVE_SESSIONS == []


def test_unsupported_error_from_create_session_maps_to_unsupported(butanol_conformers):
    register(
        RaisingEngine("director-unsupported", oevibe.UnsupportedError("no open shells"))
    )
    results = oevibe.Calculation(spring_options("director-unsupported")).Energy(
        butanol_conformers
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.UNSUPPORTED
    assert "no open shells" in results.Get(0).GetMessage()


def test_any_other_exception_maps_to_engine_error(butanol_conformers):
    register(RaisingEngine("director-error", ValueError("engine went wrong")))
    results = oevibe.Calculation(spring_options("director-error")).Energy(
        butanol_conformers
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.ENGINE_ERROR
    assert "engine went wrong" in results.Get(0).GetMessage()


def test_optional_methods_fall_through_to_the_cpp_defaults():
    engine = SpringEngine("director-defaults")
    assert list(engine.GetKeywordSpecs()) == []


def test_capabilities_round_trip_a_list_of_solvent_models():
    """``solvent_models`` is a vector of a scoped enum, and SWIG emits the element traits such a
    vector needs only as a side effect of wrapping the container's ``front``, which
    ``OEVIBE_SAFE_VECTOR`` removes. Without the traits the elements cross as opaque pointers
    that no engine can declare and that leak."""
    capabilities = oevibe.MethodCapabilities()
    capabilities.solvent_models = [oevibe.SolventModel.ALPB, oevibe.SolventModel.GBSA]
    assert list(capabilities.solvent_models) == [
        oevibe.SolventModel.ALPB,
        oevibe.SolventModel.GBSA,
    ]


def test_a_registered_engine_survives_its_last_python_reference(butanol_conformers):
    """The registry holds a ``shared_ptr``, which keeps the C++ director alive, but a director
    only borrows its Python object. Without a keepalive on ``Register`` the first callback after
    the proxy is collected -- the ``GetName`` inside the next ``Calculation`` -- runs against
    freed memory."""
    oevibe.EngineRegistry.Instance().Register(SpringEngine("director-dropped"), True)
    gc.collect()
    results = oevibe.Calculation(spring_options("director-dropped")).Energy(
        butanol_conformers
    )
    for result in results:
        assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()


def test_a_system_kept_past_its_callback_still_reads(butanol_conformers):
    """A director's reference arguments point into per-call C++ state, so an engine that keeps
    the system it was handed read freed memory once the verb returned: ``NumAtoms`` came back as
    a number in the billions rather than raising."""
    kept = {}

    class Keeping(SpringEngine):
        def ValidateSystem(self, system, settings):
            kept["validated"] = system

        def CreateSession(self, system, settings):
            kept["created"] = system
            kept["settings"] = settings
            return SpringSession()

    register(Keeping("director-kept-system"))
    oevibe.Calculation(spring_options("director-kept-system")).Energy(
        butanol_conformers
    )
    gc.collect()
    # Fresh snapshots land in the memory the per-call state just freed.
    churn = [oevibe.MolecularSystem(butanol_conformers, 0, 1) for _ in range(8)]

    assert kept["created"].NumAtoms() == butanol_conformers.NumAtoms()
    assert kept["validated"].NumAtoms() == butanol_conformers.NumAtoms()
    assert kept["settings"].method.GetName() == "spring"
    assert len(churn) == 8


def test_class_valued_members_outlive_their_parent():
    """``%naturalvar`` gives a container member value semantics but leaves a member of a wrapped
    class type a bare pointer into its parent, so the temporary ``find_method`` returns could be
    freed before the name was ever read."""
    engine = SpringEngine("director-member-copy")
    method = oevibe.find_method(engine, oevibe.Method("spring")).method
    gc.collect()
    assert method.GetName() == "spring"

    settings = oevibe.EngineSettings()
    settings.method = oevibe.Method("gfn2-xtb")
    settings.solvent = oevibe.Solvent("water")
    kept_method = settings.method
    kept_solvent = settings.solvent
    del settings
    gc.collect()
    assert kept_method.GetName() == "gfn2-xtb"
    assert kept_solvent.GetName() == "water"


def test_find_method_reads_a_python_engine_s_capabilities():
    """``find_method`` returns a ``std::optional``, which SWIG has no support for: untyped it
    reaches Python as an opaque pointer that cannot be read and leaks on every call."""
    engine = SpringEngine("director-find-method")
    found = oevibe.find_method(engine, oevibe.Method("spring"))
    assert found is not None
    assert found.method.GetName() == "spring"
    assert oevibe.find_method(engine, oevibe.Method("mmff94")) is None


def test_a_malformed_evaluation_becomes_an_engine_error(butanol_conformers):
    class ShortGradient(SpringEngine):
        def CreateSession(self, system, settings):
            session = SpringSession()
            original = session.Evaluate

            def evaluate(coords, request):
                evaluation = original(coords, request)
                if request.gradient:
                    evaluation.gradient = list(evaluation.gradient)[:-1]
                return evaluation

            session.Evaluate = evaluate
            return session

    register(ShortGradient("director-malformed"))
    results = oevibe.Calculation(spring_options("director-malformed")).Gradient(
        butanol_conformers
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.ENGINE_ERROR


def test_four_threads_share_one_python_engine(butanol_conformers):
    """Four concurrent calculations against one director engine produce correct results.

    This proves the director callbacks and their GIL handling survive concurrency; it does not
    prove the GIL is released, which ``tests/python/test_gil.py`` covers.
    """
    register(SpringEngine("director-threads"))
    molecules = [oechem.OEMol(butanol_conformers) for _ in range(4)]
    energies: list = [None] * 4
    barrier = threading.Barrier(4)

    def run(slot):
        barrier.wait()
        results = oevibe.Calculation(spring_options("director-threads")).Energy(
            molecules[slot]
        )
        energies[slot] = results.Get(0).GetEnergy()

    threads = [threading.Thread(target=run, args=(slot,)) for slot in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(energy == pytest.approx(0.0, abs=1e-9) for energy in energies)


def test_each_session_references_its_own_conformer(butanol_conformers):
    """The settled rule: a session's native state comes from its FIRST Evaluate, not from
    conformer 0 at CreateSession. If it came from conformer 0, every conformer but the first
    would start with a non-zero spring energy."""
    register(SpringEngine("director-first-evaluate"))
    results = oevibe.Calculation(spring_options("director-first-evaluate")).Energy(
        butanol_conformers
    )
    assert len(results) >= 2
    for result in results:
        assert result.GetEnergy() == pytest.approx(0.0, abs=1e-9)


def test_the_system_hands_python_a_real_openeye_molecule(butanol_conformers):
    seen = {}

    class Inspecting(SpringEngine):
        def CreateSession(self, system, settings):
            molecule = system.molecule
            seen["type"] = type(molecule).__module__
            seen["atoms"] = molecule.NumAtoms()
            seen["title"] = oechem.OEMolToSmiles(molecule)
            return SpringSession()

    register(Inspecting("director-molecule"))
    oevibe.Calculation(spring_options("director-molecule")).Energy(butanol_conformers)
    assert seen["type"].startswith("openeye")
    assert seen["atoms"] == butanol_conformers.NumAtoms()
