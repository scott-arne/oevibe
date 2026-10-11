"""Python only reads results and option views; it cannot rewrite what C++ computed."""

import gc

import pytest
from openeye import oechem

import oevibe
from oevibe.oevibe import BondOrder, BondOrderVector

# Indexing a BondOrderVector returns a pointer into its storage, which a reference to the vector
# keeps allocated but not in place, so any of these would leave elements taken earlier dangling.
# The slice names cover SWIG versions that still generate them.
_STORAGE_MUTATORS = (
    "append",
    "push_back",
    "extend",
    "insert",
    "assign",
    "resize",
    "reserve",
    "pop",
    "pop_back",
    "erase",
    "clear",
    "swap",
    "__setitem__",
    "__delitem__",
    "__setslice__",
    "__delslice__",
)


def _mutators(cls):
    return [name for name in dir(cls) if name.startswith(("Set", "Add", "Clear"))]


def test_calculation_options_are_an_independent_copy(butanol_conformers):
    opts = oevibe.CalculationOptions()
    opts.SetMethod(oevibe.Method("mmff94"))
    calc = oevibe.Calculation(opts)

    # The engine and its settings were resolved at construction, so editing a live alias of
    # the stored options would desynchronize the calculation from the SD tags it writes.
    calc.GetOptions().SetMethod(oevibe.Method("sage"))

    assert calc.GetOptions().GetMethod().GetName() == "mmff94"
    results = calc.Energy(butanol_conformers)
    assert all(r.GetStatus() == oevibe.ConformerStatus.OK for r in results)
    conf = next(iter(butanol_conformers.GetConfs()))
    assert oechem.OEHasSDData(conf, "OEVibe.mmff94.gas.Energy")
    assert not oechem.OEHasSDData(conf, "OEVibe.sage.gas.Energy")


def test_results_have_no_mutators(butanol_conformers):
    results = oevibe.energy(butanol_conformers, method="mmff94")
    assert not hasattr(results[0], "SetEnergy")
    # Listing the names also covers mutators added to the C++ classes later.
    assert _mutators(oevibe.ConformerResult) == []
    assert _mutators(oevibe.ResultSet) == []
    assert not hasattr(results, "AddWarningToAll")


def test_result_structs_are_read_only(butanol_conformers):
    results = oevibe.energy(butanol_conformers, method="mmff94")
    provenance = results[0].GetProvenance()
    with pytest.raises(AttributeError):
        provenance.method = "x"
    assert provenance.method == "mmff94"
    with pytest.raises(AttributeError):
        results[0].GetTimings().total = 0.0

    with pytest.raises(AttributeError):
        oevibe.OptimizationStats().iterations = 3


def test_bond_order_vector_cannot_move_its_storage():
    assert [name for name in _STORAGE_MUTATORS if hasattr(BondOrderVector, name)] == []


def test_bond_order_vector_reads_and_keeps_its_elements_alive():
    vector = BondOrderVector(3)
    assert len(vector) == 3
    assert len(vector[1:]) == 2
    assert [bond.order for bond in vector] == [0.0, 0.0, 0.0]

    element = vector[1]
    del vector
    gc.collect()
    # Filled vectors land in the memory just freed, so a dangling element would read 7, 8, 9.5.
    filler = BondOrder()
    filler.i, filler.j, filler.order = 7, 8, 9.5
    reused = [BondOrderVector(3, filler) for _ in range(64)]

    assert (element.i, element.j, element.order) == (0, 0, 0.0)
    assert len(reused) == 64
