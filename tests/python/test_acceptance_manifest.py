"""The committed acceptance references must agree with their manifest (spec 12.3).

The acceptance suite never runs xtb; it trusts ``tests/data/acceptance``. This module is the
check that the trust is warranted, so it compares the manifest against the committed files
themselves and deliberately does not import ``scripts/acceptance``: a test that read the
generator's own tables would agree with the generator however the generator changed.

For the same reason nothing here is derived from the manifest it is checking. The required
molecules, their charges, their solvents and the case each one must carry are written out below
from spec 12.3, so a manifest that drops a reference or relabels the settings behind one
disagrees with this module rather than with itself.

Limits worth knowing before relying on this module. A number with no committed file behind it --
a single-point energy, an optimizer cycle count, an individual wavenumber -- can only be
re-derived by running xtb again, which the acceptance suite does not do. Those are bounded for
plausibility here, not pinned. Everything that does have a file behind it is pinned exactly.

Nothing here runs xtb, but the module carries the ``acceptance`` marker: spec 12.3's suite is run
with ``pytest -m acceptance``, and that command must not skip the check that the data the suite
benchmarks against is the data the manifest describes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import re

import pytest

pytestmark = pytest.mark.acceptance

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "acceptance"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Spec 12.3 pins the references to one xtb release. The whole banner string is pinned, not the
# release number inside it: a substring test accepts 6.7.10, and a regeneration under such a
# release stamps its own version into every geometry comment, so the stamps would agree with it
# too and nothing committed would contradict the claim. Regenerating under another build means
# changing this constant in the same commit as the data.
PINNED_XTB_STAMP = "6.7.1 (edcfbbe)"
PINNED_XTB_VERSION = f"xtb version {PINNED_XTB_STAMP}"

# xtb's optimization levels, which are where the preset thresholds came from (src/Preset.cpp:21).
PRESET_ARGUMENTS = {
    "fast": ("loose", "2.0"),
    "standard": ("normal", "1.0"),
    "thorough": ("tight", "0.2"),
}

# xtb writes "energy: <Eh> gnorm: <...> xtb: <version> (<commit>)" as the xyz comment line.
XYZ_ENERGY = re.compile(r"energy:\s*(-?\d+\.\d+)")
XYZ_XTB = re.compile(r"xtb:\s*(.+?)\s*$")

# Anything that would tie the committed data to the machine that generated it.
MACHINE_TEXT = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\|\S+@\S+|\.local\b)")

# Spec 12.3's tolerances under names that carry the qualifier the spec attaches to them: the RMSD
# the spec bounds is over heavy atoms, and a bare "rmsd" invites an all-atom comparison instead.
# frequency_soft_crossover_cm1 is the wavenumber below which spec 12.3 widens the flat 2 cm^-1 by
# crossover/wavenumber; above it the tolerance is the flat value.
SPEC_TOLERANCES = {
    "heavy_atom_rmsd_angstrom": 0.05,
    "energy_kcal_per_mol": 0.01,
    "step_ratio": 1.3,
    "frequency_cm1": 2.0,
    "frequency_floor_cm1": 50.0,
    "frequency_soft_crossover_cm1": 175.0,
    "constraint_angstrom": 1e-4,
    "constraint_degrees": 1e-3,
    "single_point_hartree": 1e-6,
}

# The settings every reference was generated under. The OpenMP thread count moves an optimized
# geometry in the fifth decimal, so it is pinned and recorded rather than inherited from whoever
# ran the generator; XTBPATH is written relative to the binary's own prefix.
XTB_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "XTBHOME": None,
    "XTBPATH": "${xtb prefix}/share/xtb",
}

# How a frequency case lays its wavenumbers out. This is xtb's convention, not OEVibe's:
# ``Result::GetImaginaryModeCount()`` counts every negative frequency among the 3N-6 values
# ``src/core/Frequencies.h`` returns with the rigid-body modes projected out, so neither the
# length nor the count is comparable to a manifest entry without this layout in hand.
WAVENUMBER_LAYOUT = (
    "3N values in xtb vibspectrum order: six near-zero translations and rotations first, "
    "then the 3N-6 vibrations ascending"
)


@dataclasses.dataclass(frozen=True)
class RequiredMolecule:
    """What spec 12.3 requires of one acceptance molecule.

    Charge and solvent are fields of this table rather than of the manifest on purpose. They are
    what the xtb command line was built from, so reading them back out of the manifest would let
    a reference be relabelled -- a neutral molecule described as a cation, a water reference
    described as a methanol one -- with its own command line changed to agree and nothing left to
    contradict it.
    """

    # Every molecule in the acceptance set is neutral and closed-shell; a charged or open-shell
    # addition has to be declared here as well as in the generator.
    charge: int = 0
    multiplicity: int = 1
    presets: tuple[str, ...] = ("standard",)
    solvents: tuple[str, ...] = ()
    frequencies: bool = False
    single_point: bool = False
    constraints: tuple[tuple[str, tuple[int, ...]], ...] = ()


# Spec 12.3: about 20 drug-like molecules plus constrained cases; STANDARD for all of them and a
# five-molecule subset at FAST and THOROUGH; --alpb water for the solvated cases; --hess
# frequencies; GFN0-xTB single points. The identities are the plan's roster for this repo, and
# they are the published interface Tasks 12 and 13 are written against, so a case may be added
# here but one may not quietly go missing.
ALL_PRESETS = ("fast", "standard", "thorough")

REQUIRED_MOLECULES: dict[str, RequiredMolecule] = {
    "imidazole": RequiredMolecule(
        presets=ALL_PRESETS, frequencies=True, single_point=True
    ),
    "glycine": RequiredMolecule(
        presets=ALL_PRESETS,
        solvents=("water",),
        frequencies=True,
        single_point=True,
    ),
    "anisole": RequiredMolecule(
        presets=ALL_PRESETS, frequencies=True, single_point=True
    ),
    "urea": RequiredMolecule(frequencies=True),
    "salicylamide": RequiredMolecule(frequencies=True),
    "phenol": RequiredMolecule(single_point=True),
    "aniline": RequiredMolecule(),
    "benzocaine": RequiredMolecule(),
    "paracetamol": RequiredMolecule(presets=ALL_PRESETS, solvents=("water",)),
    "phenacetin": RequiredMolecule(),
    "aspirin": RequiredMolecule(),
    "ibuprofen": RequiredMolecule(),
    "naproxen": RequiredMolecule(),
    "caffeine": RequiredMolecule(),
    "theophylline": RequiredMolecule(),
    "nicotine": RequiredMolecule(),
    "lidocaine": RequiredMolecule(),
    "metformin": RequiredMolecule(presets=ALL_PRESETS, solvents=("water",)),
    "pyridoxine": RequiredMolecule(solvents=("water",)),
    "sulfanilamide": RequiredMolecule(),
    "acetazolamide": RequiredMolecule(),
    "aspirin-constrained": RequiredMolecule(
        constraints=(("distance", (0, 1)), ("frozen", (0,)))
    ),
    "lidocaine-constrained": RequiredMolecule(
        constraints=(("dihedral", (4, 5, 6, 7)),)
    ),
}


@dataclasses.dataclass(frozen=True)
class RequiredCase:
    """One reference spec 12.3 requires, and the settings it must have been produced under."""

    title: str
    kind: str
    charge: int
    preset: str | None = None
    solvent: str | None = None
    method: str | None = None

    @property
    def geometry(self) -> str:
        """:returns: The committed file this case's kind takes its structure from."""
        if self.kind == "optimization":
            return f"{self.title}/{_identifier(self)}.xyz"
        if self.kind == "frequencies":
            # The Hessian is taken at xtb's own STANDARD optimized geometry, not at the input.
            return f"{self.title}/{self.title}-standard.xyz"
        return f"{self.title}/input.xyz"

    @property
    def command(self) -> list[str]:
        """:returns: The xtb command line spec 12.3's settings for this case call for."""
        common = ["--chrg", str(self.charge), "--uhf", "0"]
        if self.kind == "optimization":
            assert self.preset is not None
            level, accuracy = PRESET_ARGUMENTS[self.preset]
            command = [
                "xtb",
                "input.xyz",
                "--opt",
                level,
                "--gfn",
                "2",
                "--acc",
                accuracy,
            ]
            command += common
            if self.solvent is not None:
                command += ["--alpb", self.solvent]
            return command
        if self.kind == "frequencies":
            return ["xtb", "input.xyz", "--hess", "--gfn", "2", "--acc", "1.0", *common]
        return ["xtb", "input.xyz", "--sp", "--gfn", "0", "--acc", "1.0", *common]


def _identifier(case: RequiredCase) -> str:
    if case.kind == "optimization":
        return f"{case.title}-{case.preset}" + (
            f"-{case.solvent}" if case.solvent else ""
        )
    if case.kind == "frequencies":
        return f"{case.title}-hessian"
    return f"{case.title}-gfn0-sp"


def _required_cases() -> dict[str, RequiredCase]:
    """:returns: Every reference the spec requires, keyed by the identifier it must carry."""
    required: dict[str, RequiredCase] = {}
    for title, molecule in REQUIRED_MOLECULES.items():
        cases = [
            RequiredCase(title, "optimization", molecule.charge, preset=preset)
            for preset in molecule.presets
        ]
        # Spec 12.3 solvates at STANDARD only; the FAST and THOROUGH references are gas phase.
        cases += [
            RequiredCase(
                title,
                "optimization",
                molecule.charge,
                preset="standard",
                solvent=solvent,
            )
            for solvent in molecule.solvents
        ]
        if molecule.frequencies:
            cases.append(RequiredCase(title, "frequencies", molecule.charge))
        if molecule.single_point:
            cases.append(
                RequiredCase(title, "single_point", molecule.charge, method="gfn0-xtb")
            )
        for case in cases:
            required[_identifier(case)] = case
    return required


REQUIRED_CASES = _required_cases()

CONSTRAINED_MOLECULES = frozenset(
    title for title, molecule in REQUIRED_MOLECULES.items() if molecule.constraints
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    """The committed manifest; fails rather than skips, because the data is a deliverable."""
    assert MANIFEST_PATH.is_file(), (
        f"{MANIFEST_PATH} is missing. Regenerate it with "
        "scripts/acceptance/generate_references.py --xtb <path to xtb 6.7.1>"
    )
    return json.loads(MANIFEST_PATH.read_text())


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_xyz(path: pathlib.Path) -> list[tuple[str, float, float, float]]:
    """:returns: ``(element, x, y, z)`` for every atom in an xyz file, in file order."""
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
    assert len(atoms) == count, f"{path} promises {count} atoms and lists {len(atoms)}"
    return atoms


def _dihedral_degrees(points: list[tuple[float, float, float]]) -> float:
    """:returns: The i-j-k-l torsion in degrees, in (-180, 180].

    Written out longhand rather than shared with the generator so that the measurement in the
    manifest and the measurement here are two independent calculations of the same quantity.
    """

    def subtract(a, b):
        return tuple(p - q for p, q in zip(a, b, strict=True))

    def dot(a, b):
        return sum(p * q for p, q in zip(a, b, strict=True))

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    b0 = subtract(points[0], points[1])
    b1 = subtract(points[2], points[1])
    b2 = subtract(points[3], points[2])
    length = math.sqrt(dot(b1, b1))
    b1 = tuple(component / length for component in b1)
    v = subtract(b0, tuple(dot(b0, b1) * component for component in b1))
    w = subtract(b2, tuple(dot(b2, b1) * component for component in b1))
    return math.degrees(math.atan2(dot(cross(b1, v), w), dot(v, w)))


def _required(case: dict) -> RequiredCase:
    """:returns: The spec's entry for a case, failing the test when the case is not one."""
    assert case["id"] in REQUIRED_CASES, (
        f"{case['id']} is not a reference spec 12.3 asks for"
    )
    return REQUIRED_CASES[case["id"]]


def test_the_manifest_records_the_pinned_xtb_and_a_tblite_version(manifest):
    """Exact, not a substring: ``6.7.1 in "6.7.10"`` is true and the releases are not the same."""
    assert manifest["xtb_version"] == PINNED_XTB_VERSION
    tblite = manifest["tblite_version"]
    assert re.match(r"\d+\.\d+\.\d+ build \S+", tblite), tblite


def test_the_manifest_records_the_tolerances_spec_12_3_sets(manifest):
    """The names are part of the contract: ``heavy_atom_rmsd_angstrom`` says which atoms count."""
    assert manifest["tolerances"] == SPEC_TOLERANCES


def test_the_manifest_records_the_environment_its_references_came_from(manifest):
    """A setting that moves the numbers and is not recorded makes the data unreproducible.

    ``OMP_NUM_THREADS`` is the one that bites: naproxen at STANDARD lands 1.9e-5 Angstrom away at
    one thread from where it lands at fourteen. That is three orders below every tolerance here,
    but spec 12.3 asks for the settings, so the generator pins the thread count and records it.
    """
    assert manifest["xtb_environment"] == XTB_ENVIRONMENT


def test_the_manifest_covers_exactly_the_molecules_spec_12_3_requires(manifest):
    """Pinned, because the committed tree cannot notice a molecule that is simply gone."""
    titles = [molecule["title"] for molecule in manifest["molecules"]]
    assert titles == list(REQUIRED_MOLECULES)


def test_every_molecule_records_the_charge_and_multiplicity_it_was_run_at(manifest):
    """The charge the command lines are checked against comes from here, not from the manifest."""
    for molecule in manifest["molecules"]:
        required = REQUIRED_MOLECULES[molecule["title"]]
        assert molecule["charge"] == required.charge, molecule["title"]
        assert molecule["multiplicity"] == required.multiplicity, molecule["title"]
        assert molecule["input"] == f"{molecule['title']}/input.xyz", molecule["title"]


def test_the_manifest_covers_exactly_the_references_spec_12_3_requires(manifest):
    """The whole required set, not just the ones the manifest happens to list.

    A frequency or single-point case reuses a geometry that an optimization case already names,
    so deleting one leaves every file still accounted for and every surviving entry still
    self-consistent. Only an expectation written down outside the data notices it.
    """
    present = {case["id"] for case in manifest["cases"]}
    missing = sorted(set(REQUIRED_CASES) - present)
    extra = sorted(present - set(REQUIRED_CASES))
    assert not missing, f"references spec 12.3 requires are missing: {missing}"
    assert not extra, f"references the spec does not describe: {extra}"
    assert len(manifest["cases"]) == len(REQUIRED_CASES), (
        "a case identifier is duplicated"
    )


def test_the_references_of_each_kind_are_the_ones_spec_12_3_names(manifest):
    """Counts and identities per kind, so a whole kind cannot thin out unnoticed."""
    for kind in ("optimization", "frequencies", "single_point"):
        present = {case["id"] for case in manifest["cases"] if case["kind"] == kind}
        expected = {
            identifier
            for identifier, case in REQUIRED_CASES.items()
            if case.kind == kind
        }
        assert present == expected, kind
    counts = {kind: 0 for kind in ("optimization", "frequencies", "single_point")}
    for case in manifest["cases"]:
        counts[case["kind"]] += 1
    assert counts == {"optimization": 37, "frequencies": 5, "single_point": 4}


def test_every_case_records_the_settings_it_was_generated_under(manifest):
    """Kind, preset, solvent and method must be the spec's, not whatever the manifest says.

    Taking the solvent from the manifest and treating anything truthy as water is what let a
    water reference be relabelled a methanol one with its command line changed to agree.
    """
    for case in manifest["cases"]:
        required = _required(case)
        assert case["title"] == required.title, case["id"]
        assert case["kind"] == required.kind, case["id"]
        if required.kind == "optimization":
            assert case["preset"] == required.preset, case["id"]
            assert case["solvent"] == required.solvent, case["id"]
        if required.kind == "single_point":
            assert case["method"] == required.method, case["id"]


def test_the_manifest_names_every_committed_file_and_no_others(manifest):
    committed = {
        str(path.relative_to(DATA_DIR))
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    named = {molecule["input"] for molecule in manifest["molecules"]}
    named |= {case["geometry"] for case in manifest["cases"]}
    assert committed == named, (
        f"only in the tree: {sorted(committed - named)}; only in the manifest: "
        f"{sorted(named - committed)}"
    )


def test_every_input_geometry_matches_its_recorded_hash(manifest):
    for molecule in manifest["molecules"]:
        path = DATA_DIR / molecule["input"]
        assert _sha256(path) == molecule["input_sha256"], (
            f"{path} is not the file that was used"
        )


def test_every_case_geometry_matches_its_recorded_hash(manifest):
    """Every kind, not just the optimizations: the hash is what makes a repointed case show up."""
    for case in manifest["cases"]:
        path = DATA_DIR / case["geometry"]
        assert _sha256(path) == case["geometry_sha256"], (
            f"{case['id']}: {path} is not the file the run used"
        )


def test_every_case_names_the_geometry_file_its_kind_and_identifier_call_for(manifest):
    for case in manifest["cases"]:
        assert case["geometry"] == _required(case).geometry, case["id"]


def test_every_command_line_is_the_one_spec_12_3_s_settings_call_for(manifest):
    """Rebuilt from the pinned settings, so the manifest cannot describe a run it did not make."""
    for case in manifest["cases"]:
        assert case["command"] == _required(case).command, case["id"]


def test_no_command_line_names_a_path_on_the_generating_machine(manifest):
    for case in manifest["cases"]:
        assert case["command"][0] == "xtb", case["command"]
        for argument in case["command"]:
            assert not argument.startswith("/"), case["command"]


def test_every_optimized_energy_is_the_energy_in_its_geometry_file(manifest):
    """xtb writes the energy into the xyz comment, so the pair must agree digit for digit.

    Both numbers come from the same decimal text through ``float``, and json round-trips a double
    exactly, so this is an equality rather than a tolerance.
    """
    for case in manifest["cases"]:
        if case["kind"] != "optimization":
            continue
        comment = (DATA_DIR / case["geometry"]).read_text().splitlines()[1]
        written = XYZ_ENERGY.search(comment)
        assert written is not None, (
            f"{case['geometry']} has no energy in its comment line"
        )
        assert float(written.group(1)) == case["energy_hartree"], case["id"]


def test_every_optimized_geometry_came_from_the_pinned_xtb(manifest):
    """Against the pinned release, not against whatever the manifest claims to have run.

    A regeneration under an unsupported release stamps itself into every comment line, so the
    stamps would agree with a manifest that named it. They are checked against the constant.
    """
    for case in manifest["cases"]:
        if case["kind"] != "optimization":
            continue
        comment = (DATA_DIR / case["geometry"]).read_text().splitlines()[1]
        written = XYZ_XTB.search(comment)
        assert written is not None, (
            f"{case['geometry']} has no xtb stamp in its comment line"
        )
        assert written.group(1) == PINNED_XTB_STAMP, (
            f"{case['geometry']} was written by xtb {written.group(1)}, "
            f"but the references are pinned to {PINNED_XTB_STAMP}"
        )


def test_every_case_geometry_keeps_the_elements_of_its_input(manifest):
    """A geometry filed under the wrong molecule, or an atom reordered, breaks every index."""
    inputs = {
        molecule["title"]: [atom[0] for atom in _read_xyz(DATA_DIR / molecule["input"])]
        for molecule in manifest["molecules"]
    }
    for case in manifest["cases"]:
        elements = [atom[0] for atom in _read_xyz(DATA_DIR / case["geometry"])]
        assert elements == inputs[case["title"]], case["id"]


def test_no_optimization_reference_was_run_under_a_constraint(manifest):
    """Nothing in the committed data was optimized under a constraint, and every case says so.

    The two ``*-constrained`` molecules duplicate their twins byte for byte; only their names
    suggest otherwise. Spec 12.3's constrained clause is self-referential -- a constrained
    optimization holds its constraint -- so the reference it needs is the value measured on the
    hash-pinned input, which is ``molecules[].constraints``, not an xtb geometry. The label and
    the pointer are here so a consumer cannot RMSD a constrained result against an unconstrained
    reference and believe it checked something.
    """
    for case in manifest["cases"]:
        if case["kind"] != "optimization":
            continue
        assert case["constrained"] is False, case["id"]
        expected = (
            f"molecules[{case['title']}].constraints"
            if case["title"] in CONSTRAINED_MOLECULES
            else None
        )
        assert case["constraint_reference"] == expected, case["id"]


def test_the_constrained_molecules_declare_the_constraints_spec_12_3_requires(manifest):
    """Pinned here, because the constraint references are the whole of the constrained coverage.

    Discovering the constrained molecules from the manifest and asking only that one survives
    lets a constraint be deleted with its case pointer nulled to match, which is the only way
    spec 12.3's constrained clause is covered at all.
    """
    for molecule in manifest["molecules"]:
        required = REQUIRED_MOLECULES[molecule["title"]].constraints
        declared = tuple(
            (constraint["kind"], tuple(constraint["atoms"]))
            for constraint in molecule["constraints"]
        )
        assert declared == required, molecule["title"]
    assert CONSTRAINED_MOLECULES, "the spec requires constrained cases"


def test_every_constraint_value_measures_the_committed_input(manifest):
    for molecule in manifest["molecules"]:
        title = molecule["title"]
        atoms = _read_xyz(DATA_DIR / molecule["input"])
        for constraint in molecule["constraints"]:
            positions = [atoms[index][1:] for index in constraint["atoms"]]
            if constraint["kind"] == "distance":
                measured = math.dist(positions[0], positions[1])
                assert measured == pytest.approx(constraint["value"], abs=1e-9), title
            elif constraint["kind"] == "dihedral":
                measured = _dihedral_degrees(positions)
                assert measured == pytest.approx(constraint["value"], abs=1e-6), title
            else:
                # A frozen atom's reference is its position in the hash-pinned input file.
                assert "value" not in constraint, constraint


def test_every_frequency_case_has_one_wavenumber_per_degree_of_freedom(manifest):
    """3N wavenumbers: the six near-zero translations and rotations, then the rest ascending.

    xtb emits the translations and rotations first whatever their value and the vibrations in
    ascending order after them, so the ordering is part of what a consumer indexes by, and each
    case states it rather than leaving it to be discovered.
    """
    for case in manifest["cases"]:
        if case["kind"] != "frequencies":
            continue
        assert case["wavenumber_layout"] == WAVENUMBER_LAYOUT, case["id"]
        atoms = _read_xyz(DATA_DIR / case["geometry"])
        wavenumbers = case["wavenumbers_cm1"]
        assert len(wavenumbers) == 3 * len(atoms), case["id"]
        floor = manifest["tolerances"]["frequency_floor_cm1"]
        assert all(abs(value) < floor for value in wavenumbers[:6]), case["id"]
        vibrations = wavenumbers[6:]
        assert vibrations == sorted(vibrations), case["id"]


def test_every_frequency_case_counts_its_imaginary_modes_below_the_floor(manifest):
    """Not every reference geometry is a minimum, so the count is recorded rather than assumed.

    The urea reference is the planar saddle point xtb's optimizer reaches from the OMEGA input;
    its two imaginary modes are a property of the committed data, and a consumer that silently
    assumed real frequencies would be reading it wrong.

    The field counts the vibrations below ``-frequency_floor_cm1``, which is NOT what
    ``Result::GetImaginaryModeCount()`` counts: that is every negative frequency, over the 3N-6
    values that remain once the rigid-body modes are projected out. The name says which of the
    two this is, so they cannot be compared by accident.
    """
    floor = manifest["tolerances"]["frequency_floor_cm1"]
    for case in manifest["cases"]:
        if case["kind"] != "frequencies":
            continue
        imaginary = [value for value in case["wavenumbers_cm1"][6:] if value < -floor]
        assert case["imaginary_modes_below_floor"] == len(imaginary), case["id"]


def test_every_single_point_energy_is_a_plausible_gfn0_energy(manifest):
    """Weak by construction: nothing committed holds this number, only xtb could re-derive it."""
    for case in manifest["cases"]:
        if case["kind"] != "single_point":
            continue
        atoms = _read_xyz(DATA_DIR / case["geometry"])
        assert -10.0 * len(atoms) < case["energy_hartree"] < 0.0, case["id"]


def test_the_subset_molecules_carry_all_three_presets_and_the_rest_carry_one(manifest):
    """Spec 12.3's preset rule, stated on its own rather than left implicit in the case set."""
    presets: dict[str, set[str]] = {title: set() for title in REQUIRED_MOLECULES}
    for case in manifest["cases"]:
        if case["kind"] == "optimization" and case["solvent"] is None:
            presets[case["title"]].add(case["preset"])
    subset = [
        title for title, found in presets.items() if found == set(PRESET_ARGUMENTS)
    ]
    assert len(subset) == 5, subset
    for title, found in presets.items():
        assert found == set(REQUIRED_MOLECULES[title].presets), title


def test_the_manifest_holds_no_text_from_the_generating_machine(manifest):
    """The manifest records versions and settings, never where they lived."""
    found = MACHINE_TEXT.search(MANIFEST_PATH.read_text())
    assert found is None, f"machine-specific text in the manifest: {found.group(0)!r}"
