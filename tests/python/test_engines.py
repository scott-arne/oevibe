"""The built-in tblite and xtb engines through the verbs, and their library discovery.

Engine tests skip unless OEVIBE_TBLITE_LIBRARY or OEVIBE_XTB_LIBRARY names the library (the
``tblite_library`` and ``xtb_library`` fixtures).
"""

import math
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from openeye import oechem

import oevibe

ALL_PROPERTIES = ("charges", "dipole", "bond_orders", "orbital_energies")

# Runs in a child process whose engines can only search empty directories and the system loader.
_MISSING_LIBRARY_CHILD = """
import sys

import oevibe

oevibe.set_python_library_directory(sys.argv[1])
options = oevibe.CalculationOptions()
options.SetMethod(oevibe.Method(sys.argv[2]))
try:
    oevibe.Calculation(options)
except oevibe.EngineUnavailableError as error:
    # The reason puts each candidate on its own line; one line keeps the report last.
    print("UNAVAILABLE " + " ".join(str(error).splitlines()))
else:
    print("AVAILABLE")
"""


@pytest.fixture
def ethanol():
    """Ethanol with one OMEGA conformer and explicit hydrogens."""
    from openeye import oeomega

    mol = oechem.OEMol()
    assert oechem.OESmilesToMol(mol, "CCO")
    options = oeomega.OEOmegaOptions()
    options.SetMaxConfs(1)
    omega = oeomega.OEOmega(options)
    assert omega.Build(mol) == oeomega.OEOmegaReturnCode_Success
    return mol


def _ok(results):
    result = results[0]
    assert result.GetStatus() == oevibe.ConformerStatus.OK, result.GetMessage()
    return result


def test_import_passes_the_interpreter_library_directory():
    libdir = sysconfig.get_config_var("LIBDIR")
    expected = (
        libdir
        if isinstance(libdir, str) and libdir
        else os.path.join(sys.prefix, "lib")
    )
    assert oevibe.get_python_library_directory() == expected


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("solvent", [None, "water"])
def test_gfn2_energy_gradient_and_optimize_through_tblite(ethanol, solvent):
    single = _ok(
        oevibe.energy(oechem.OEMol(ethanol), method="gfn2-xtb", solvent=solvent)
    )
    gradient = _ok(
        oevibe.gradient(oechem.OEMol(ethanol), method="gfn2-xtb", solvent=solvent)
    )
    assert gradient.GetEnergy() == pytest.approx(single.GetEnergy(), abs=1e-6)
    assert max(abs(g) for g in gradient.GetGradient()) > 1e-3
    optimized = _ok(
        oevibe.optimize(oechem.OEMol(ethanol), method="gfn2-xtb", solvent=solvent)
    )
    assert optimized.GetOptimizationStats().converged
    assert optimized.GetEnergy() <= single.GetEnergy() + 1e-6
    provenance = optimized.GetProvenance()
    assert provenance.engine == "tblite"
    assert re.fullmatch(r"0\.[67]\.\d+", provenance.engine_version)
    assert provenance.solvent_model == ("gas" if solvent is None else "alpb")


@pytest.mark.usefixtures("tblite_library")
def test_auto_water_is_alpb_and_is_recorded(ethanol):
    gas = _ok(oevibe.energy(oechem.OEMol(ethanol), method="gfn2-xtb"))
    tagged = oechem.OEMol(ethanol)
    water = _ok(oevibe.energy(tagged, method="gfn2-xtb", solvent="water"))
    explicit = _ok(
        oevibe.energy(
            oechem.OEMol(ethanol),
            method="gfn2-xtb",
            solvent=oevibe.Solvent("water", oevibe.SolventModel.ALPB),
        )
    )
    # ALPB water lowers the GFN2 energy of the spike S4 ethanol by 4.65 kcal/mol (probe).
    assert water.GetEnergy() < gas.GetEnergy() - 1.0
    assert explicit.GetEnergy() == pytest.approx(water.GetEnergy(), abs=1e-6)
    assert any("resolved to 'alpb'" in warning for warning in water.GetWarnings())
    assert not any("resolved to" in warning for warning in explicit.GetWarnings())
    assert not any("resolved to" in warning for warning in gas.GetWarnings())
    conf = next(iter(tagged.GetConfs()))
    assert oechem.OEHasSDData(conf, "OEVibe.gfn2-xtb.alpb-water.Energy")


@pytest.mark.usefixtures("tblite_library")
def test_gfn1_energy_through_tblite(ethanol):
    gfn1 = _ok(oevibe.energy(oechem.OEMol(ethanol), method="gfn1-xtb"))
    gfn2 = _ok(oevibe.energy(oechem.OEMol(ethanol), method="gfn2-xtb"))
    assert gfn1.GetProvenance().engine == "tblite"
    assert gfn1.GetProvenance().method == "gfn1-xtb"
    # The parametrizations' totals differ by 483 kcal/mol for the spike S4 ethanol (probe).
    assert abs(gfn1.GetEnergy() - gfn2.GetEnergy()) > 100.0


@pytest.mark.parametrize(
    ("method", "engine", "library"),
    [("gfn2-xtb", "tblite", "tblite_library"), ("gfn0-xtb", "xtb", "xtb_library")],
)
def test_properties_through_the_verbs(request, ethanol, method, engine, library):
    request.getfixturevalue(library)
    result = _ok(oevibe.energy(ethanol, method=method, properties=ALL_PROPERTIES))
    assert result.GetProvenance().engine == engine
    assert sum(result.GetCharges()) == pytest.approx(0.0, abs=1e-6)
    # For the spike S4 ethanol, GFN2 and GFN0 give the oxygen -0.45 and -0.55 e, dipoles of
    # 1.78 and 1.81 D, and bond orders from 0.91 to 1.03 (probe); the bounds leave room for the
    # OMEGA conformer.
    oxygen = next(iter(ethanol.GetAtoms(oechem.OEIsOxygen())))
    assert result.GetCharge(oxygen) < -0.3
    assert math.hypot(*result.GetDipole()) > 0.5
    assert result.GetHOMO() < result.GetLUMO()
    for bond in ethanol.GetBonds():
        assert 0.8 < result.GetBondOrder(bond) < 1.2


@pytest.mark.usefixtures("xtb_library")
def test_gfn0_reports_the_xtb_c_api(ethanol):
    result = _ok(oevibe.energy(ethanol, method="gfn0-xtb"))
    assert result.GetProvenance().engine_version == "C API 1.0.0"


@pytest.mark.usefixtures("xtb_library")
def test_gfn_ff_is_unavailable_with_its_reason(ethanol):
    with pytest.raises(oevibe.EngineUnavailableError, match="gfnff_topo"):
        oevibe.energy(ethanol, method="gfn-ff")


@pytest.mark.usefixtures("xtb_library")
def test_gfn0_is_gas_phase_only(ethanol):
    with pytest.raises(oevibe.OptionsError, match="gas phase only"):
        oevibe.energy(ethanol, method="gfn0-xtb", solvent="water")


@pytest.mark.parametrize(
    ("method", "engine"), [("gfn2-xtb", "tblite"), ("gfn0-xtb", "xtb")]
)
def test_a_missing_library_makes_the_engine_unavailable(tmp_path, method, engine):
    # The registry and its built-in engines are process-wide, and each engine loads its library
    # once and keeps the outcome, as does the library directory. A child process gets its own
    # registry, so the failed load and the changed directory cannot leak into the other tests,
    # which need the libraries named by OEVIBE_TBLITE_LIBRARY and OEVIBE_XTB_LIBRARY.
    conda_prefix = tmp_path / "conda"
    python_lib = tmp_path / "python-lib"
    environment = dict(os.environ)
    environment.pop("OEVIBE_TBLITE_LIBRARY", None)
    environment.pop("OEVIBE_XTB_LIBRARY", None)
    environment["CONDA_PREFIX"] = str(conda_prefix)
    package_root = str(Path(oevibe.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_root, *filter(None, [environment.get("PYTHONPATH")])]
    )
    child = subprocess.run(
        [sys.executable, "-c", _MISSING_LIBRARY_CHILD, str(python_lib), method],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    report = child.stdout.strip().splitlines()[-1]
    if report == "AVAILABLE":
        # Only the last candidate, the bare file name, can still succeed: the system loader
        # (DYLD/LD library paths, or the extension's own rpaths) reaches the library here.
        pytest.skip(f"the system loader finds lib{engine} on this machine")
    assert report.startswith("UNAVAILABLE "), report
    assert f"engine '{engine}' is unavailable" in report
    # The reason lists the candidates: $CONDA_PREFIX/lib and the Python library directory.
    assert str(conda_prefix / "lib") in report
    assert str(python_lib) in report
    assert f"lib{engine}" in report
