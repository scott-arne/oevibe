"""The ergonomic Python engine base classes of spec 8.7."""

import gc
import itertools

import numpy as np
import pytest
from openeye import oechem

import oevibe


class SpringSession(oevibe.PythonEngineSession):
    """Harmonic springs back to this session's own starting geometry.

    The reference comes from the FIRST evaluate, not from conformer 0 at CreateSession: each
    session is evaluated with its own conformer's coordinates (spec 6.4 order independence), so
    anchoring on conformer 0 would give every later conformer a non-zero energy.
    """

    def __init__(self):
        super().__init__()
        self.reference = None
        self.seen_shapes = []

    def evaluate(self, coords, request):
        self.seen_shapes.append((coords.shape, coords.dtype, coords.flags.writeable))
        if self.reference is None:
            self.reference = coords.copy()
        delta = coords - self.reference
        energy = 0.5 * 10.0 * float(np.sum(delta * delta))
        if request.gradient:
            return oevibe.Evaluation(energy=energy, gradient=10.0 * delta)
        return oevibe.Evaluation(energy=energy)


class SpringEngine(oevibe.PythonEngine):
    def __init__(self, name="python-spring"):
        super().__init__()
        self._name = name
        self.last_session = None

    def GetName(self):
        return self._name

    def GetVersion(self):
        return "2.0"

    def GetMethods(self):
        capabilities = oevibe.MethodCapabilities()
        capabilities.method = oevibe.Method("spring")
        return [capabilities]

    def CreateSession(self, system, settings):
        self.last_session = SpringSession()
        return self.last_session


def options_for(name):
    options = oevibe.CalculationOptions()
    options.SetEngine(name)
    options.SetMethod(oevibe.Method("spring"))
    return options


# The registry is process-wide and keeps every engine for the life of the process, so each
# test needs a name no earlier test used. id(object()) does not work: CPython frees the
# temporary at once and hands the same address back every time.
_NAMES = itertools.count()


@pytest.fixture
def spring_engine():
    engine = SpringEngine(f"python-spring-{next(_NAMES)}")
    oevibe.register_engine(engine)
    return engine


def test_evaluate_receives_an_n_by_three_read_only_float64_array(
    butanol_conformers, spring_engine
):
    oevibe.Calculation(options_for(spring_engine.GetName())).Energy(butanol_conformers)
    shape, dtype, writeable = spring_engine.last_session.seen_shapes[0]
    assert shape == (butanol_conformers.NumAtoms(), 3)
    assert dtype == np.float64
    assert writeable is False


def test_gradient_and_energy_reach_cpp(butanol_conformers, spring_engine):
    results = oevibe.Calculation(options_for(spring_engine.GetName())).Gradient(
        butanol_conformers
    )
    for result in results:
        assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
        assert result.GetEnergy() == pytest.approx(0.0, abs=1e-9)
        assert len(result.GetGradient()) == 3 * butanol_conformers.NumAtoms()


def test_optimization_through_a_python_engine_converges(
    butanol_conformers, spring_engine
):
    results = oevibe.Calculation(options_for(spring_engine.GetName())).Optimize(
        butanol_conformers, oevibe.OptimizerOptions()
    )
    for result in results:
        assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()


@pytest.mark.parametrize("shape", [(6, 3), (18,)])
def test_evaluation_accepts_both_gradient_shapes(shape):
    values = np.arange(18, dtype=np.float64).reshape(shape)
    evaluation = oevibe.Evaluation(energy=1.5, gradient=values)
    assert evaluation.energy == 1.5
    assert list(evaluation.gradient) == list(np.arange(18, dtype=np.float64))


def test_evaluation_accepts_a_square_hessian():
    hessian = np.eye(9, dtype=np.float64)
    evaluation = oevibe.Evaluation(energy=0.0, hessian=hessian)
    assert len(evaluation.hessian) == 81


def test_evaluation_rejects_a_gradient_that_is_not_three_wide():
    with pytest.raises(ValueError, match="gradient"):
        oevibe.Evaluation(energy=0.0, gradient=np.zeros((4, 2)))


def test_registering_a_duplicate_name_raises(spring_engine):
    with pytest.raises(oevibe.OptionsError):
        oevibe.register_engine(SpringEngine(spring_engine.GetName()))
    oevibe.register_engine(SpringEngine(spring_engine.GetName()), replace=True)


def test_registering_a_builtin_name_raises():
    with pytest.raises(oevibe.OptionsError):
        oevibe.register_engine(SpringEngine("oeff"))


def test_the_package_keeps_a_registered_engine_alive(butanol_conformers):
    """Dropping every caller reference must not leave the C++ registry with a dead director."""
    name = "python-spring-retained"
    oevibe.register_engine(SpringEngine(name))
    gc.collect()
    results = oevibe.Calculation(options_for(name)).Energy(butanol_conformers)
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.OK


def test_system_molecule_is_a_usable_openeye_molecule(butanol_conformers):
    seen = {}

    class Inspecting(SpringEngine):
        def CreateSession(self, system, settings):
            molecule = system.molecule
            seen["smiles"] = oechem.OEMolToSmiles(molecule)
            seen["atoms"] = molecule.NumAtoms()
            return super().CreateSession(system, settings)

    engine = Inspecting("python-spring-molecule")
    oevibe.register_engine(engine)
    oevibe.Calculation(options_for(engine.GetName())).Energy(butanol_conformers)
    assert seen["atoms"] == butanol_conformers.NumAtoms()
    assert seen["smiles"] == oechem.OEMolToSmiles(butanol_conformers)


def test_evaluate_returning_something_else_becomes_an_engine_error(butanol_conformers):
    class WrongReturn(SpringSession):
        def evaluate(self, coords, request):
            return 1.5

    class Engine(SpringEngine):
        def CreateSession(self, system, settings):
            return WrongReturn()

    engine = Engine(f"python-spring-wrong-return-{next(_NAMES)}")
    oevibe.register_engine(engine)
    results = oevibe.Calculation(options_for(engine.GetName())).Energy(
        butanol_conformers
    )
    assert results.Get(0).GetStatus() == oevibe.ConformerStatus.ENGINE_ERROR
    assert "oevibe.Evaluation" in results.Get(0).GetMessage()


def test_register_engine_rejects_a_non_engine():
    with pytest.raises(TypeError, match="oevibe.Engine"):
        oevibe.register_engine("not an engine")


def test_evaluation_rejects_a_hessian_that_is_not_square():
    with pytest.raises(ValueError, match="hessian"):
        oevibe.Evaluation(energy=0.0, hessian=np.zeros((3, 6)))
    with pytest.raises(ValueError, match="hessian"):
        oevibe.Evaluation(energy=0.0, hessian=np.zeros((3, 3, 3)))
