"""Shared fixtures and configuration for oevibe Python tests."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")

from oevibe import _build_info

# The module must share the openeye-toolkits wheel's OpenEye runtime; a static copy carries a
# second runtime whose generic-data tag registry relabels SD data on every molecule passed in.
if _build_info.OPENEYE_LIBRARY_TYPE != "SHARED":
    pytest.exit(
        "python/oevibe/_oevibe.so is not linked to the openeye-toolkits shared libraries; "
        "build it with: cmake --preset python && cmake --build --preset python",
        returncode=pytest.ExitCode.USAGE_ERROR,
    )


@pytest.fixture
def butanol_conformers():
    """1-Butanol with up to three OMEGA conformers and explicit hydrogens."""
    from openeye import oechem, oeomega

    mol = oechem.OEMol()
    assert oechem.OESmilesToMol(mol, "CCCCO")
    options = oeomega.OEOmegaOptions()
    options.SetMaxConfs(3)
    omega = oeomega.OEOmega(options)
    assert omega.Build(mol) == oeomega.OEOmegaReturnCode_Success
    return mol


@pytest.fixture
def flat_ethanol():
    """Ethanol without coordinates, which every calculation must reject."""
    from openeye import oechem

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, "CCO")
    return mol


def _engine_library(variable: str) -> str:
    """:returns: The library path in ``variable``; skips the test when it names no file."""
    path = os.environ.get(variable, "")
    if not path or not os.path.isfile(path):
        pytest.skip(f"{variable} is unset or does not name an existing file")
    return path


@pytest.fixture
def tblite_library() -> str:
    """Path to libtblite from ``OEVIBE_TBLITE_LIBRARY``; skips the test when it is unusable."""
    return _engine_library("OEVIBE_TBLITE_LIBRARY")


@pytest.fixture
def xtb_library() -> str:
    """Path to libxtb from ``OEVIBE_XTB_LIBRARY``; skips the test when it is unusable."""
    return _engine_library("OEVIBE_XTB_LIBRARY")


# The committed xtb reference set of spec 12.3, shared by every acceptance module.
ACCEPTANCE_DIR = Path(__file__).resolve().parents[1] / "data" / "acceptance"


@pytest.fixture(scope="session")
def acceptance_manifest():
    """The committed xtb reference manifest.

    This FAILS rather than skips when the file is absent: the manifest is committed, so its
    absence is a repository defect, and a skip here would silently disable the integrity checks
    that exist to catch exactly that. Only the engine-library fixtures skip.
    """
    manifest = ACCEPTANCE_DIR / "manifest.json"
    assert manifest.is_file(), (
        f"{manifest} is missing; it is committed data, regenerate it with "
        "scripts/acceptance/generate_references.py"
    )
    return json.loads(manifest.read_text())


@pytest.fixture(scope="session")
def acceptance_dir():
    """The directory the committed acceptance references live in."""
    return ACCEPTANCE_DIR


def read_xyz_molecule(path):
    """Read one acceptance reference geometry.

    The ``XYZ_Connect`` flavor is deliberate and load-bearing. OEChem's default xyz flavor also
    perceives bond orders, formal charges and implicit hydrogens, and on a tertiary aliphatic
    amine it invents an N-H: lidocaine and nicotine come back as cations carrying one implicit
    hydrogen each. ``Calculation`` then adds that hydrogen for real (``SetAddHydrogens`` is on
    by default), so the engine sees C14H23N2O rather than C14H22N2O and tblite lands 0.36 Eh --
    226 kcal/mol -- away from the xtb reference for the same file. Connectivity alone is all the
    comparisons need: the automorphism-aware RMSD matches on the graph, and the engines take
    elements, coordinates and the charge the manifest records.

    :param path: An xyz file under ``ACCEPTANCE_DIR``.
    :returns: An ``oechem.OEMol`` with one conformer and no implicit hydrogens.
    """
    from openeye import oechem

    mol = oechem.OEMol()
    stream = oechem.oemolistream()
    stream.SetFormat(oechem.OEFormat_XYZ)
    stream.SetFlavor(oechem.OEFormat_XYZ, oechem.OEIFlavor_XYZ_Connect)
    assert stream.open(str(path)), f"cannot open {path}"
    assert oechem.OEReadMolecule(stream, mol), f"cannot read {path}"
    stream.close()
    assert not oechem.OEHasImplicitHydrogens(mol), (
        f"{path} was read with implicit hydrogens; every atom in an xyz file is explicit, so "
        "one invented here becomes a real atom the reference calculation never had"
    )
    return mol
