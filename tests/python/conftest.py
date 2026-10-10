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
def aspirin_mol():
    """Create an aspirin molecule (C9H8O4) for testing."""
    from openeye import oechem

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O")
    return mol


@pytest.fixture
def ethanol_mol():
    """Create an ethanol molecule (C2H6O) for testing."""
    from openeye import oechem

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, "CCO")
    return mol
