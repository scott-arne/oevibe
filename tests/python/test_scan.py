"""The scan verb with the OpenEye force-field engine."""

import gc
import math

import pytest
from openeye import oechem, oeomega

import oevibe
from oevibe.oevibe import scan_point_status_name

_PREFIX = "OEVibe.mmff94.gas."


def _chain(mol):
    """Heavy atoms along the chain from its lowest-index end; OMEGA renumbers atoms."""
    heavy = list(mol.GetAtoms(oechem.OEIsHeavy()))
    chain = [min((a for a in heavy if a.GetHvyDegree() == 1), key=lambda a: a.GetIdx())]
    while True:
        seen = {atom.GetIdx() for atom in chain}
        following = [
            nbr
            for nbr in chain[-1].GetAtoms(oechem.OEIsHeavy())
            if nbr.GetIdx() not in seen
        ]
        if not following:
            return chain
        chain.append(following[0])


def _indices(mol):
    return [atom.GetIdx() for atom in _chain(mol)]


def _torsion(conf, indices):
    atoms = [conf.GetAtom(oechem.OEHasAtomIdx(index)) for index in indices]
    return math.degrees(oechem.OEGetTorsion(conf, *atoms))


def _relative_by_angle(scan):
    return {
        round(scan.GetPoint(i).GetAngles()[0]): scan.GetPoint(i).GetRelativeEnergy()
        for i in range(scan.GetNumPoints())
        if scan.GetPoint(i).HasRelativeEnergy()
    }


@pytest.fixture
def anti_butane():
    """One OMEGA conformer of butane with the backbone dihedral set to 180 degrees."""
    mol = oechem.OEMol()
    assert oechem.OESmilesToMol(mol, "CCCC")
    options = oeomega.OEOmegaOptions()
    options.SetMaxConfs(1)
    assert oeomega.OEOmega(options).Build(mol) == oeomega.OEOmegaReturnCode_Success
    assert oechem.OESetTorsion(mol, *_chain(mol), math.pi)
    return mol


def test_butane_profile_has_the_expected_shape(anti_butane):
    scan = oevibe.scan(anti_butane, [_chain(anti_butane)], method="mmff94", step=30.0)
    assert isinstance(scan, oevibe.ScanResult)
    assert scan.GetNumPoints() == 12
    assert not scan.IsTruncated()
    for i in range(scan.GetNumPoints()):
        point = scan.GetPoint(i)
        assert point.GetStatus() == oevibe.ScanPointStatus.OK, point.GetMessage()
    energy = _relative_by_angle(scan)
    assert energy[180] == 0.0
    for sign in (1, -1):
        assert energy[30 * sign] > energy[60 * sign] < energy[90 * sign]
        assert energy[90 * sign] < energy[120 * sign] > energy[150 * sign]
        assert energy[60 * sign] > 0.3
    assert max(energy, key=energy.get) == 0


def test_atom_objects_and_indices_give_the_same_scan(anti_butane):
    copy = oechem.OEMol(anti_butane)
    by_atoms = oevibe.scan(
        anti_butane,
        [_chain(anti_butane)],
        method="mmff94",
        step=60.0,
        angle_range=(60.0, 180.0),
    )
    by_indices = oevibe.scan(
        copy,
        [tuple(_indices(copy))],
        method="mmff94",
        step=60.0,
        angle_range=(60.0, 180.0),
    )
    assert list(by_atoms.GetDihedral(0)) == _indices(anti_butane)
    assert list(by_indices.GetDihedral(0)) == _indices(anti_butane)
    assert by_atoms.GetNumPoints() == by_indices.GetNumPoints() == 3
    for i in range(3):
        first, second = by_atoms.GetPoint(i), by_indices.GetPoint(i)
        assert list(first.GetAngles()) == list(second.GetAngles()) == [60.0 * (i + 1)]
        assert first.GetEnergy() == second.GetEnergy()


def test_get_molecule_has_one_tagged_conformer_per_point(anti_butane):
    before = anti_butane.GetCoords()
    indices = _indices(anti_butane)
    scan = oevibe.scan(anti_butane, [indices], method="mmff94", step=60.0)
    profile = scan.GetMolecule()
    assert isinstance(profile, oechem.OEMol)
    assert profile.NumAtoms() == anti_butane.NumAtoms()
    assert profile.NumConfs() == scan.GetNumPoints() == 6
    for i, conf in enumerate(profile.GetConfs()):
        point = scan.GetPoint(i)
        angle = point.GetAngles()[0]
        assert oechem.OEGetSDData(conf, _PREFIX + "ScanAngles") == f"{angle:.6f}"
        assert float(oechem.OEGetSDData(conf, _PREFIX + "Energy")) == pytest.approx(
            point.GetEnergy(), abs=1e-6
        )
        assert float(
            oechem.OEGetSDData(conf, _PREFIX + "RelativeEnergy")
        ) == pytest.approx(point.GetRelativeEnergy(), abs=1e-6)
        assert oechem.OEGetSDData(conf, _PREFIX + "Status") == "OK"
        # OpenEye stores coordinates in single precision.
        deviation = (_torsion(conf, indices) - angle + 180.0) % 360.0 - 180.0
        assert abs(deviation) < 0.01
    # The scan writes nothing back to its input.
    assert anti_butane.GetCoords() == before
    assert not oechem.OEHasSDData(anti_butane.GetActive(), _PREFIX + "Status")


def test_max_optimizations_truncates_the_scan(anti_butane):
    scan = oevibe.scan(
        anti_butane,
        [_chain(anti_butane)],
        method="mmff94",
        step=30.0,
        max_optimizations=3,
    )
    assert scan.IsTruncated()
    assert any("cap of 3" in warning for warning in scan.GetWarnings())
    statuses = [scan.GetPoint(i).GetStatus() for i in range(scan.GetNumPoints())]
    assert statuses.count(oevibe.ScanPointStatus.OK) == 3
    assert statuses.count(oevibe.ScanPointStatus.NOT_ATTEMPTED) == 9
    assert scan.GetMolecule().NumConfs() == 3


def test_scan_options_work_through_calculation(anti_butane):
    options = oevibe.CalculationOptions()
    options.SetMethod(oevibe.Method("mmff94"))
    scan_options = oevibe.ScanOptions()
    scan_options.AddDihedral(*_indices(anti_butane))
    scan_options.SetStep(60.0)
    scan_options.SetRange(120.0, 180.0)
    scan = oevibe.Calculation(options).Scan(anti_butane, scan_options)
    assert [list(scan.GetPoint(i).GetAngles()) for i in range(2)] == [[120.0], [180.0]]
    assert scan.GetProvenance().GetResolvedOptions()["task"] == "scan"


# Each is rejected while the options are built or validated, before any atom is looked up.
@pytest.mark.parametrize(
    ("dihedrals", "kwargs"),
    [
        ([], {}),
        ([(0, 1, 2)], {}),
        ([(0, 1, 1, 2)], {}),
        ([(0, 1, 2, "3")], {}),
        ([(0, 1, 2, -1)], {}),
        ([(0, 1, 2, 3.0)], {}),
        ([(0, 1, 2, True)], {}),
        ((0, 1, 2, 3), {}),
        ("0123", {}),
        ([(0, 1, 2, 3), (1, 2, 3, 4), (4, 1, 2, 3)], {}),
        ([(0, 1, 2, 3)], {"step": 0.0}),
        ([(0, 1, 2, 3)], {"step": 7.0}),
        ([(0, 1, 2, 3)], {"angle_range": (90.0, 0.0)}),
        ([(0, 1, 2, 3)], {"angle_range": (-180.0, 200.0)}),
        ([(0, 1, 2, 3)], {"angle_range": (0.0,)}),
        ([(0, 1, 2, 3)], {"max_optimizations": 0}),
        (5, {}),
        ([(0, 1, 2, 3)], {"step": "fifteen"}),
        ([(0, 1, 2, 3)], {"step": None}),
        ([(0, 1, 2, 3)], {"angle_range": ("start", 180.0)}),
        ([(0, 1, 2, 3)], {"angle_range": (-180.0, None)}),
    ],
)
def test_invalid_options_raise_options_error(anti_butane, dihedrals, kwargs):
    with pytest.raises(oevibe.OptionsError):
        oevibe.scan(anti_butane, dihedrals, method="mmff94", **kwargs)


def test_missing_atoms_raise_input_error(anti_butane):
    with pytest.raises(oevibe.InputError):
        oevibe.scan(anti_butane, [(0, 1, 2, 99)], method="mmff94")


def test_scan_results_are_read_only(anti_butane):
    scan = oevibe.scan(anti_butane, [_chain(anti_butane)], method="mmff94", step=120.0)
    mutators = [
        f"{cls.__name__}.{name}"
        for cls in (oevibe.ScanResult, oevibe.ScanPoint)
        for name in dir(cls)
        if name.startswith(("Set", "Add", "Clear"))
    ]
    assert mutators == []
    assert not hasattr(scan, "Finalize")
    assert int(oevibe.ScanPointStatus.NOT_ATTEMPTED) == 4
    assert (
        scan_point_status_name(oevibe.ScanPointStatus.NOT_ATTEMPTED) == "NOT_ATTEMPTED"
    )


def test_point_view_outlives_its_result(anti_butane):
    scan = oevibe.scan(anti_butane, [_chain(anti_butane)], method="mmff94", step=120.0)
    point = scan.GetPoint(2)
    energy = point.GetEnergy()
    angles = list(point.GetAngles())
    del scan
    gc.collect()
    # New results land in the memory just freed, so a dangling view would misread it.
    reused = [oevibe.ScanResult() for _ in range(64)]
    assert point.GetEnergy() == energy
    assert list(point.GetAngles()) == angles == [180.0]
    assert len(reused) == 64
