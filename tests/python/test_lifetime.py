"""Views returned by reference must keep the C++ object they point into alive."""

import gc

import oevibe


def test_option_views_outlive_their_owners():
    opts = oevibe.CalculationOptions()
    opts.SetMethod(oevibe.Method("mmff94"))
    opts.SetSolvent(oevibe.Solvent("water"))
    calc = oevibe.Calculation(opts)
    options_view = calc.GetOptions()
    method_view = opts.GetMethod()
    solvent_view = opts.GetSolvent()
    del calc, opts
    gc.collect()

    # Fresh default options reuse the freed memory, so a dangling view would now read the
    # default method and the gas phase instead of mmff94 and water.
    reused = [oevibe.CalculationOptions() for _ in range(64)]

    assert options_view.GetMethod().GetName() == "mmff94"
    assert options_view.GetSolvent().GetName() == "water"
    assert method_view.GetName() == "mmff94"
    assert solvent_view.GetName() == "water"
    assert len(reused) == 64
