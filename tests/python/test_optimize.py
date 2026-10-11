"""The optimize verb with the OpenEye force-field engine."""

import math

import pytest
from openeye import oechem

import oevibe


def _jitter(mol, amplitude=0.08):
    for conf in mol.GetConfs():
        coords = conf.GetCoords()
        for index, (x, y, z) in coords.items():
            shift = amplitude * math.sin(1.7 * index + 0.3)
            conf.SetCoords(
                conf.GetAtom(oechem.OEHasAtomIdx(index)), (x + shift, y - shift, z)
            )


class IsOxygen(oechem.OEUnaryAtomPred):
    def __call__(self, atom):
        return atom.GetAtomicNum() == 8

    def CreateCopy(self):
        return IsOxygen().__disown__()


def test_optimize_lowers_energies_and_writes_coordinates(butanol_conformers):
    _jitter(butanol_conformers)
    reference = oechem.OEMol(butanol_conformers)
    singles = oevibe.energy(reference, method="mmff94")
    results = oevibe.optimize(butanol_conformers, method="mmff94")
    for single, optimized in zip(singles, results, strict=True):
        assert optimized.GetStatus() == oevibe.ConformerStatus.OK
        assert optimized.GetEnergy() <= single.GetEnergy() + 1e-6
        assert optimized.GetOptimizationStats().converged
    first = next(iter(butanol_conformers.GetConfs()))
    written = results[0].GetCoordinates()
    # OpenEye stores coordinates in single precision.
    for index, xyz in first.GetCoords().items():
        assert written[3 * index : 3 * index + 3] == pytest.approx(xyz, abs=1e-4)


def test_freeze_atoms_with_builtin_and_python_predicates(butanol_conformers):
    _jitter(butanol_conformers)
    constraints = oevibe.ConstraintSet()
    constraints.FreezeAtoms(
        butanol_conformers, oechem.OEHasAtomicNum(oechem.OEElemNo_C)
    )
    constraints.FreezeAtoms(butanol_conformers, IsOxygen())
    before = {conf.GetIdx(): conf.GetCoords() for conf in butanol_conformers.GetConfs()}
    results = oevibe.optimize(
        butanol_conformers, method="mmff94", constraints=constraints
    )
    # A failed conformer writes nothing back, which would leave the frozen atoms trivially put.
    assert all(r.GetStatus() == oevibe.ConformerStatus.OK for r in results)
    for conf in butanol_conformers.GetConfs():
        for atom in conf.GetAtoms(oechem.OEIsHeavy()):
            assert conf.GetCoords(atom) == pytest.approx(
                before[conf.GetIdx()][atom.GetIdx()]
            )


def test_optimize_options(butanol_conformers):
    _jitter(butanol_conformers)
    results = oevibe.optimize(butanol_conformers, method="mmff94", max_iterations=1)
    assert all(r.GetStatus() == oevibe.ConformerStatus.NOT_CONVERGED for r in results)
    constraints = oevibe.ConstraintSet()
    constraints.AddDistance(0, 1)
    with pytest.raises(oevibe.OptionsError):
        oevibe.optimize(
            butanol_conformers,
            method="mmff94",
            constraints=constraints,
            algorithm="lbfgs",
        )
    with pytest.raises(oevibe.OptionsError):
        oevibe.optimize(butanol_conformers, method="mmff94", algorithm="newton")


def test_out_of_range_integers_are_option_errors(butanol_conformers):
    # SWIG's unsigned conversion would raise OverflowError for these.
    with pytest.raises(oevibe.OptionsError, match="max_iterations"):
        oevibe.optimize(butanol_conformers, method="mmff94", max_iterations=-1)
    with pytest.raises(oevibe.OptionsError, match="max_iterations"):
        oevibe.optimize(butanol_conformers, method="mmff94", max_iterations=2**32)
    with pytest.raises(oevibe.OptionsError, match="multiplicity"):
        oevibe.optimize(butanol_conformers, method="mmff94", multiplicity=-1)
