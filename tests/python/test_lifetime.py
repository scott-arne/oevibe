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
