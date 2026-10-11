"""Shared fixtures and configuration for oevibe Python tests."""

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
