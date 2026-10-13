"""Energy and gradient verbs with the OpenEye force-field engine."""

import pytest
from openeye import oechem

import oevibe


def test_energy_runs_every_conformer_and_writes_tags(butanol_conformers):
    results = oevibe.energy(butanol_conformers, method="mmff94")
    assert len(results) == butanol_conformers.NumConfs()
    assert all(r.GetStatus() == oevibe.ConformerStatus.OK for r in results)
    assert results.GetLowest().GetRelativeEnergy() == pytest.approx(0.0)
    for conf in butanol_conformers.GetConfs():
        assert oechem.OEHasSDData(conf, "OEVibe.mmff94.gas.Energy")
        assert oechem.OEGetSDData(conf, "OEVibe.mmff94.gas.Status") == "OK"


def test_relative_energies_and_provenance(butanol_conformers):
    results = oevibe.energy(butanol_conformers, method="mmff94")
    relative = dict(results.GetRelativeEnergies())
    assert min(relative.values()) == pytest.approx(0.0)
    resolved = dict(results[0].GetProvenance().GetResolvedOptions())
    assert resolved["method"] == "mmff94"
    assert resolved["engine"] == "oeff"


def test_gradient_uses_the_openeye_layout(butanol_conformers):
    results = oevibe.gradient(butanol_conformers, method="mmff94")
    assert len(results[0].GetGradient()) == 3 * butanol_conformers.GetMaxAtomIdx()


def test_sheffield_water_changes_the_energy_and_the_tag(butanol_conformers):
    gas = oevibe.energy(butanol_conformers, method="mmff94")[0].GetEnergy()
    water = oevibe.energy(butanol_conformers, method="mmff94", solvent="water")[
        0
    ].GetEnergy()
    assert abs(water - gas) > 0.1
    conf = next(iter(butanol_conformers.GetConfs()))
    assert oechem.OEHasSDData(conf, "OEVibe.mmff94.sheffield-water.Energy")


def test_option_errors(butanol_conformers):
    oevibe.energy(butanol_conformers, method="mmff94", preset="thorough")
    with pytest.raises(oevibe.OptionsError):
        oevibe.energy(butanol_conformers, method="mmff94", preset="slow")
    with pytest.raises(oevibe.OptionsError):
        oevibe.energy(butanol_conformers, method="mmff94", properties=["charges"])
    with pytest.raises(oevibe.OptionsError):
        oevibe.energy(butanol_conformers, method="mmff94", solvent="octanol")


def test_out_of_range_integers_are_option_errors(butanol_conformers):
    # SWIG's integer conversion would raise OverflowError for these.
    with pytest.raises(oevibe.OptionsError, match="multiplicity"):
        oevibe.energy(butanol_conformers, method="mmff94", multiplicity=-1)
    with pytest.raises(oevibe.OptionsError, match="multiplicity"):
        oevibe.gradient(butanol_conformers, method="mmff94", multiplicity=2**32)
    with pytest.raises(oevibe.OptionsError, match="charge"):
        oevibe.energy(butanol_conformers, method="mmff94", charge=2**31)
    with pytest.raises(oevibe.OptionsError, match="charge"):
        oevibe.energy(butanol_conformers, method="mmff94", charge=-(2**31) - 1)


def test_input_errors(flat_ethanol):
    with pytest.raises(oevibe.InputError):
        oevibe.energy(flat_ethanol, method="mmff94")
