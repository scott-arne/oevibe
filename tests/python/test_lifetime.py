"""Views returned by reference must keep the C++ object they point into alive.

Each test gives the surviving view its own owner. A shared owner would let one view's link
keep the memory valid for another view whose own link is missing.
"""

import gc

import oevibe


def _reuse_freed_memory():
    # Fresh default options land in the memory just freed, so a dangling view would read the
    # default method (gfn2-xtb) and the gas phase instead of the values set before the free.
    return [oevibe.CalculationOptions() for _ in range(64)]


def test_method_view_outlives_its_options():
    opts = oevibe.CalculationOptions()
    opts.SetMethod(oevibe.Method("mmff94"))
    method_view = opts.GetMethod()
    del opts
    gc.collect()
    reused = _reuse_freed_memory()

    assert method_view.GetName() == "mmff94"
    assert len(reused) == 64


def test_solvent_view_outlives_its_options():
    opts = oevibe.CalculationOptions()
    opts.SetSolvent(oevibe.Solvent("water"))
    solvent_view = opts.GetSolvent()
    del opts
    gc.collect()
    reused = _reuse_freed_memory()

    assert solvent_view.GetName() == "water"
    assert len(reused) == 64


def test_calculation_options_copy_outlives_the_calculation():
    opts = oevibe.CalculationOptions()
    opts.SetMethod(oevibe.Method("mmff94"))
    opts.SetSolvent(oevibe.Solvent("water"))
    calc = oevibe.Calculation(opts)
    options_copy = calc.GetOptions()
    del calc, opts
    gc.collect()
    reused = _reuse_freed_memory()

    assert options_copy.GetMethod().GetName() == "mmff94"
    assert options_copy.GetSolvent().GetName() == "water"
    assert len(reused) == 64


# The frequency types are not exported by the package until the frequencies verb arrives, so these
# tests reach them through the generated module.
def test_optimizer_options_view_outlives_its_thermo_options():
    optimizer = oevibe.OptimizerOptions()
    optimizer.SetMaxIterations(50)
    thermo = oevibe.oevibe.ThermoOptions()
    thermo.SetOptimizerOptions(optimizer)
    optimizer_view = thermo.GetOptimizerOptions()
    del thermo, optimizer
    gc.collect()
    # Fresh default options land in the memory just freed, where a dangling view would find the
    # optimizer's iteration limit unset.
    reused = [oevibe.oevibe.ThermoOptions() for _ in range(64)]

    assert optimizer_view.GetMaxIterations() == 50
    assert len(reused) == 64


def test_thermochemistry_view_keeps_its_result_alive(monkeypatch):
    # Python cannot build a result that carries thermochemistry before the frequencies verb, and
    # an absent value raises before the getter's wrapper code runs. A stand-in for the C++ call
    # lets the wrapper run, so this checks that it links the view to the result it points into.
    stand_in = oevibe.oevibe.Thermochemistry()
    monkeypatch.setattr(
        oevibe.oevibe._oevibe,
        "ConformerResult_GetThermochemistry",
        lambda result: stand_in,
    )
    result = oevibe.ConformerResult()

    assert result.GetThermochemistry()._parent is result


def test_thermochemistry_vectors_are_copies():
    # A vector member returned as a view would be mutable from Python and would dangle once the
    # struct is freed.
    thermo = oevibe.oevibe.Thermochemistry()
    frequencies = thermo.frequencies
    principal_moments = thermo.principal_moments
    del thermo
    gc.collect()

    assert isinstance(frequencies, tuple)
    assert isinstance(principal_moments, tuple)
    assert frequencies == ()
    assert principal_moments == ()
