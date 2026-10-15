"""Views returned by reference must keep the C++ object they point into alive.

Each test gives the surviving view its own owner. A shared owner would let one view's link
keep the memory valid for another view whose own link is missing.

``CalculationOptions.GetMethod`` and ``GetSolvent`` are the two exceptions, and the first two
tests pin them as such: they hand back an owned copy, which needs no link at all.
"""

import gc

import oevibe


def _reuse_freed_memory():
    # Fresh default options land in the memory just freed, so a dangling view would read the
    # default method (gfn2-xtb) and the gas phase instead of the values set before the free.
    return [oevibe.CalculationOptions() for _ in range(64)]


def test_the_method_getter_returns_an_independent_copy():
    # Method is an immutable value type with value equality and a hash, so a borrowed view
    # into CalculationOptions' storage would change value underneath a key already in a set.
    # It crosses as a copy instead, which is why this getter carries no _parent link: the
    # absence is asserted so the project-wide keepalive rule is not reapplied to it by reflex.
    opts = oevibe.CalculationOptions()
    opts.SetMethod(oevibe.Method("mmff94"))
    method = opts.GetMethod()
    assert not hasattr(method, "_parent")

    opts.SetMethod(oevibe.Method("sage"))
    assert method.GetName() == "mmff94"

    del opts
    gc.collect()
    reused = _reuse_freed_memory()

    assert method.GetName() == "mmff94"
    assert len(reused) == 64


def test_the_solvent_getter_returns_an_independent_copy():
    opts = oevibe.CalculationOptions()
    opts.SetSolvent(oevibe.Solvent("water"))
    solvent = opts.GetSolvent()
    assert not hasattr(solvent, "_parent")

    opts.SetSolvent(oevibe.Solvent("octanol"))
    assert solvent.GetName() == "water"

    del opts
    gc.collect()
    reused = _reuse_freed_memory()

    assert solvent.GetName() == "water"
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


def test_optimizer_options_view_outlives_its_thermo_options():
    optimizer = oevibe.OptimizerOptions()
    optimizer.SetMaxIterations(50)
    thermo = oevibe.ThermoOptions()
    thermo.SetOptimizerOptions(optimizer)
    optimizer_view = thermo.GetOptimizerOptions()
    del thermo, optimizer
    gc.collect()
    # Fresh default options land in the memory just freed, where a dangling view would find the
    # optimizer's iteration limit unset.
    reused = [oevibe.ThermoOptions() for _ in range(64)]

    assert optimizer_view.GetMaxIterations() == 50
    assert len(reused) == 64


def test_optimizer_options_view_outlives_its_scan_options():
    optimizer = oevibe.OptimizerOptions()
    optimizer.SetMaxIterations(50)
    scan = oevibe.oevibe.ScanOptions()
    scan.SetOptimizerOptions(optimizer)
    optimizer_view = scan.GetOptimizerOptions()
    linked = getattr(optimizer_view, "_parent", None) is scan
    del scan, optimizer
    gc.collect()
    # Fresh default options land in the memory just freed, where a dangling view would find the
    # optimizer's iteration limit unset.
    reused = [oevibe.oevibe.ScanOptions() for _ in range(64)]

    assert optimizer_view.GetMaxIterations() == 50
    assert linked
    assert len(reused) == 64


def test_thermochemistry_view_keeps_its_result_alive(monkeypatch):
    # A real frequencies result cannot show a missing link: freed memory is not reliably
    # overwritten, so reading a view after its result set is deleted passes either way. A stand-in
    # for the C++ call lets the wrapper run, so this checks deterministically that it links the
    # view to the result it points into.
    stand_in = oevibe.Thermochemistry()
    monkeypatch.setattr(
        oevibe.oevibe._oevibe,
        "ConformerResult_GetThermochemistry",
        lambda result: stand_in,
    )
    result = oevibe.ConformerResult()

    assert result.GetThermochemistry()._parent is result


def test_scan_point_view_keeps_its_result_alive(monkeypatch):
    # Reading a view after its result is deleted passes with or without the link, because freed
    # memory is not reliably overwritten, and a result's points cannot be built from Python. A
    # stand-in for the C++ call lets the wrapper run, so this checks deterministically that it
    # links the view to the result it points into.
    stand_in = oevibe.ScanPoint()
    monkeypatch.setattr(
        oevibe.oevibe._oevibe,
        "ScanResult_GetPoint",
        lambda result, index: stand_in,
    )
    scan = oevibe.ScanResult()

    assert scan.GetPoint(0)._parent is scan


def test_scan_provenance_view_keeps_its_result_alive():
    scan = oevibe.ScanResult()

    assert scan.GetProvenance()._parent is scan


def test_thermochemistry_vectors_are_copies():
    # A vector member returned as a view would be mutable from Python and would dangle once the
    # struct is freed.
    thermo = oevibe.Thermochemistry()
    frequencies = thermo.frequencies
    principal_moments = thermo.principal_moments
    del thermo
    gc.collect()

    assert isinstance(frequencies, tuple)
    assert isinstance(principal_moments, tuple)
    assert frequencies == ()
    assert principal_moments == ()
