"""The committed acceptance references must agree with their manifest (spec 12.3).

The acceptance suite never runs xtb; it trusts ``tests/data/acceptance``. This module is the
check that the trust is warranted, so it compares the manifest against the committed files
themselves and deliberately does not import ``scripts/acceptance``: a test that read the
generator's own tables would agree with the generator however the generator changed.

Limits worth knowing before relying on this module. A number with no committed file behind it --
a single-point energy, an optimizer cycle count, an individual wavenumber -- can only be
re-derived by running xtb again, which the acceptance suite does not do. Those are bounded for
plausibility here, not pinned. Everything that does have a file behind it is pinned exactly.

Nothing here runs xtb, but the module carries the ``acceptance`` marker: spec 12.3's suite is run
with ``pytest -m acceptance``, and that command must not skip the check that the data the suite
benchmarks against is the data the manifest describes.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re

import pytest

pytestmark = pytest.mark.acceptance

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "acceptance"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Spec 12.3 pins the references to one xtb release; data from another release is not comparable.
PINNED_XTB_VERSION = "6.7.1"

# xtb's optimization levels, which are where the preset thresholds came from (src/Preset.cpp:21).
PRESET_ARGUMENTS = {
    "fast": ("loose", "2.0"),
    "standard": ("normal", "1.0"),
    "thorough": ("tight", "0.2"),
}

# The five molecules that also get FAST and THOROUGH references (spec 12.3).
SUBSET = ("imidazole", "glycine", "anisole", "paracetamol", "metformin")

# xtb writes "energy: <Eh> gnorm: <...> xtb: <version> (<commit>)" as the xyz comment line.
XYZ_ENERGY = re.compile(r"energy:\s*(-?\d+\.\d+)")
XYZ_XTB = re.compile(r"xtb:\s*(.+?)\s*$")

# Anything that would tie the committed data to the machine that generated it.
MACHINE_TEXT = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\|\S+@\S+|\.local\b)")

# Spec 12.3's tolerances under names that carry the qualifier the spec attaches to them: the RMSD
# the spec bounds is over heavy atoms, and a bare "rmsd" invites an all-atom comparison instead.
SPEC_TOLERANCES = {
    "heavy_atom_rmsd_angstrom": 0.05,
    "energy_kcal_per_mol": 0.01,
    "step_ratio": 1.3,
    "frequency_cm1": 2.0,
    "frequency_floor_cm1": 50.0,
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


def _expected_command(case: dict, charge: int) -> list[str]:
    """:returns: The xtb command line the case's own recorded settings call for.

    Derived from the settings rather than copied from the manifest, so a reference regenerated at
    a different level -- or a command line edited after the fact -- stops agreeing with the entry
    that describes it.
    """
    common = ["--chrg", str(charge), "--uhf", "0"]
    if case["kind"] == "optimization":
        level, accuracy = PRESET_ARGUMENTS[case["preset"]]
        command = ["xtb", "input.xyz", "--opt", level, "--gfn", "2", "--acc", accuracy]
        command += common
        if case["solvent"] is not None:
            command += ["--alpb", case["solvent"]]
        return command
    if case["kind"] == "frequencies":
        return ["xtb", "input.xyz", "--hess", "--gfn", "2", "--acc", "1.0", *common]
    if case["kind"] == "single_point":
        assert case["method"] == "gfn0-xtb", case
        return ["xtb", "input.xyz", "--sp", "--gfn", "0", "--acc", "1.0", *common]
    raise AssertionError(f"unknown case kind {case['kind']!r}")


def _expected_identifier(case: dict) -> str:
    """:returns: The identifier the case's own recorded settings call for.

    Anchoring the identifier is what keeps ``solvent`` honest: the command line is rebuilt from
    ``solvent``, so flipping it to ``null`` and dropping ``--alpb water`` from the command agrees
    with itself, and only the ``-water`` the identifier carries contradicts it.
    """
    title = case["title"]
    if case["kind"] == "optimization":
        return f"{title}-{case['preset']}" + ("-water" if case["solvent"] else "")
    if case["kind"] == "frequencies":
        return f"{title}-hessian"
    if case["kind"] == "single_point":
        return f"{title}-gfn0-sp"
    raise AssertionError(f"unknown case kind {case['kind']!r}")


def _expected_geometry(case: dict) -> str:
    """:returns: The reference file the case's kind and identifier call for."""
    title = case["title"]
    if case["kind"] == "optimization":
        return f"{title}/{case['id']}.xyz"
    if case["kind"] == "frequencies":
        # The Hessian is taken at xtb's own STANDARD optimized geometry, not at the input.
        return f"{title}/{title}-standard.xyz"
    if case["kind"] == "single_point":
        return f"{title}/input.xyz"
    raise AssertionError(f"unknown case kind {case['kind']!r}")


def _charges(manifest: dict) -> dict[str, int]:
    return {molecule["title"]: molecule["charge"] for molecule in manifest["molecules"]}


def test_the_manifest_records_the_pinned_xtb_and_a_tblite_version(manifest):
    assert PINNED_XTB_VERSION in manifest["xtb_version"], manifest["xtb_version"]
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


def test_every_case_identifier_follows_the_settings_recorded_beside_it(manifest):
    for case in manifest["cases"]:
        assert case["id"] == _expected_identifier(case), case


def test_every_case_names_the_geometry_file_its_kind_and_identifier_call_for(manifest):
    for case in manifest["cases"]:
        assert case["geometry"] == _expected_geometry(case), case["id"]


def test_every_command_line_follows_the_settings_recorded_beside_it(manifest):
    charges = _charges(manifest)
    for case in manifest["cases"]:
        expected = _expected_command(case, charges[case["title"]])
        assert case["command"] == expected, case["id"]


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


def test_every_optimized_geometry_came_from_the_xtb_the_manifest_names(manifest):
    for case in manifest["cases"]:
        if case["kind"] != "optimization":
            continue
        comment = (DATA_DIR / case["geometry"]).read_text().splitlines()[1]
        written = XYZ_XTB.search(comment)
        assert written is not None, (
            f"{case['geometry']} has no xtb stamp in its comment line"
        )
        assert written.group(1) in manifest["xtb_version"], (
            f"{case['geometry']} was written by xtb {written.group(1)}, "
            f"but the manifest records {manifest['xtb_version']!r}"
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
    constrained = {
        molecule["title"]
        for molecule in manifest["molecules"]
        if molecule["constraints"]
    }
    assert constrained, "no molecule declares a constraint"
    for case in manifest["cases"]:
        if case["kind"] != "optimization":
            continue
        assert case["constrained"] is False, case["id"]
        expected = (
            f"molecules[{case['title']}].constraints"
            if case["title"] in constrained
            else None
        )
        assert case["constraint_reference"] == expected, case["id"]


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
    presets: dict[str, set[str]] = {
        molecule["title"]: set() for molecule in manifest["molecules"]
    }
    for case in manifest["cases"]:
        if case["kind"] == "optimization" and case["solvent"] is None:
            presets[case["title"]].add(case["preset"])
    for title, found in presets.items():
        expected = set(PRESET_ARGUMENTS) if title in SUBSET else {"standard"}
        assert found == expected, title


def test_every_case_identifier_is_unique(manifest):
    identifiers = [case["id"] for case in manifest["cases"]]
    assert len(identifiers) == len(set(identifiers)), sorted(identifiers)


def test_the_manifest_holds_no_text_from_the_generating_machine(manifest):
    """The manifest records versions and settings, never where they lived."""
    found = MACHINE_TEXT.search(MANIFEST_PATH.read_text())
    assert found is None, f"machine-specific text in the manifest: {found.group(0)!r}"
