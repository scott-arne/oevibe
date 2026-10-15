"""The acceptance molecule set and the reference cases built from it (spec 12.3)."""

from __future__ import annotations

import dataclasses
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "acceptance"

# xtb's optimization levels, which are where the preset thresholds came from (src/Preset.cpp).
PRESET_LEVELS = {
    "fast": ("loose", "2.0"),
    "standard": ("normal", "1.0"),
    "thorough": ("tight", "0.2"),
}

# The five molecules that also get FAST and THOROUGH references (spec 12.3).
SUBSET = ("imidazole", "glycine", "anisole", "paracetamol", "metformin")


@dataclasses.dataclass(frozen=True)
class AcceptanceMolecule:
    """One input structure and what is generated from it."""

    title: str
    smiles: str
    charge: int = 0
    solvated: bool = False  # also generate an --alpb water optimization at STANDARD
    frequencies: bool = False  # also generate an xtb --hess reference
    single_point: bool = False  # also generate a GFN0-xTB --sp reference
    constraints: tuple[dict, ...] = ()

    @property
    def presets(self) -> tuple[str, ...]:
        return (
            ("fast", "standard", "thorough") if self.title in SUBSET else ("standard",)
        )


MOLECULES: tuple[AcceptanceMolecule, ...] = (
    AcceptanceMolecule("imidazole", "c1c[nH]cn1", frequencies=True, single_point=True),
    AcceptanceMolecule(
        "glycine", "NCC(=O)O", frequencies=True, single_point=True, solvated=True
    ),
    AcceptanceMolecule("anisole", "COc1ccccc1", frequencies=True, single_point=True),
    AcceptanceMolecule("urea", "NC(=O)N", frequencies=True),
    AcceptanceMolecule("salicylamide", "NC(=O)c1ccccc1O", frequencies=True),
    AcceptanceMolecule("phenol", "Oc1ccccc1", single_point=True),
    AcceptanceMolecule("aniline", "Nc1ccccc1"),
    AcceptanceMolecule("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
    AcceptanceMolecule("paracetamol", "CC(=O)Nc1ccc(O)cc1", solvated=True),
    AcceptanceMolecule("phenacetin", "CCOc1ccc(NC(C)=O)cc1"),
    AcceptanceMolecule("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    AcceptanceMolecule("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    AcceptanceMolecule("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    AcceptanceMolecule("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    AcceptanceMolecule("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    AcceptanceMolecule("nicotine", "CN1CCCC1c1cccnc1"),
    AcceptanceMolecule("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    AcceptanceMolecule("metformin", "CN(C)C(=N)NC(=N)N", solvated=True),
    AcceptanceMolecule("pyridoxine", "Cc1ncc(CO)c(CO)c1O", solvated=True),
    AcceptanceMolecule("sulfanilamide", "Nc1ccc(cc1)S(N)(=O)=O"),
    AcceptanceMolecule("acetazolamide", "CC(=O)Nc1nnc(s1)S(N)(=O)=O"),
    AcceptanceMolecule(
        "aspirin-constrained",
        "CC(=O)Oc1ccccc1C(=O)O",
        constraints=(
            {"kind": "distance", "atoms": (0, 1)},
            {"kind": "frozen", "atoms": (0,)},
        ),
    ),
    AcceptanceMolecule(
        "lidocaine-constrained",
        "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        constraints=({"kind": "dihedral", "atoms": (4, 5, 6, 7)},),
    ),
)
