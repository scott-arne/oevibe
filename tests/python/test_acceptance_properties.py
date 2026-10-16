"""Frequencies, constraints and GFN0 single points against xtb (spec 12.3).

The three criteria spec 12.3 states after the optimization one, which
``test_acceptance_optimization.py`` covers:

- "Finite-difference frequencies match ``xtb --hess`` within the tolerance set from spike S5's
  measured tblite/xtb agreement (target 2 cm^-1 for modes above 50 cm^-1), recorded in the
  manifest."
- "Constrained optimizations hold every constraint within 1e-4 Angstrom or 1e-3 degrees."
- "GFN0-xTB single points ... through ``XtbEngine`` match ``xtb --sp`` within 1e-6 Eh, since
  they share the same library."

Every tolerance is read from ``manifest["tolerances"]``, which ``test_acceptance_manifest.py``
pins to spec 12.3's literal numbers, so neither can drift without the other. Each criterion is
its own test, so a failure names the criterion that moved and a case that misses one is still
held to the others. Run with ``pytest -m acceptance``; nothing here runs xtb, it only consumes
what xtb 6.7.1 produced.

Spec 12.3's remaining clause, "and GFN-FF single points when S4 left ``gfn-ff`` available", is
not exercised: S4 left it unavailable because libxtb writes ``gfnff_topo`` into the process
working directory (``src/engines/XtbLibrary.cpp``), and the manifest holds no GFN-FF case.

The two sides lay their frequencies out differently. The manifest records 3N wavenumbers in
xtb's vibspectrum order, six near-zero rigid-body modes first; ``GetFrequencies()`` returns the
3N-6 vibrations with those projected out (``src/core/Frequencies.h``). The comparison pairs
``GetFrequencies()`` elementwise with ``wavenumbers_cm1[6:]``, and the mode-count test asserts
the lengths and the ascending order that pairing needs rather than leaving them assumed.

What each assertion can catch, measured on this reference set:

- The constraint criterion is the one that needs help to discriminate at all. Every target was
  measured on the input geometry, so an optimizer that returned its input unchanged would hold
  every constraint perfectly. Each constrained molecule is therefore also optimized WITHOUT
  constraints, and ``test_an_unconstrained_optimization_moves_every_constraint`` requires that
  run to move the same quantity off the target by more than the tolerance. Measured:
  0.0090 Angstrom for aspirin's distance (90x the tolerance), 0.130 Angstrom for its frozen
  atom (1300x), 2.12 degrees for lidocaine's dihedral (2100x).
- The constrained runs are real optimizations, not no-ops: 27 and 21 accepted iterations,
  moving some atom 0.89 and 0.70 Angstrom while the constrained quantities hold.
- The frequency tolerance has teeth at 2 cm^-1 against modes that run to 3500 cm^-1, and the
  mode count alone separates all five references (21, 24, 42, 18 and 45 vibrations), so a case
  compared against the wrong reference fails on length before it reaches the tolerance. What
  no frequency number can show is which geometry the Hessian was taken at, so
  ``test_the_hessian_is_taken_at_the_committed_xtb_geometry`` checks that separately.
- The GFN0 agreement is 2e-11 to 4e-9 Eh against a 1e-6 Eh tolerance, which is tight enough to
  resolve the geometry it is pinned at: the same molecule's xtb-optimized STANDARD geometry is
  6.6e-4 to 4.4e-3 Eh away, and ``test_the_gfn0_tolerance_resolves_the_geometry_it_runs_at``
  asserts that margin per case before the criterion is asserted.
- The imaginary-mode check would be vacuous on an all-minima reference set, so
  ``test_some_frequency_reference_has_an_imaginary_mode_below_the_floor`` requires the set to
  keep a case that exercises it. Today that is urea, the planar saddle point xtb reaches.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math

import numpy as np
import pytest
from conftest import ACCEPTANCE_DIR, read_xyz_molecule
from openeye import oechem

import oevibe

pytestmark = pytest.mark.acceptance

MANIFEST_PATH = ACCEPTANCE_DIR / "manifest.json"

# CODATA 2018, the value in include/oevibe/Units.h. Spec 12.3 states the single-point tolerance
# in Hartree and every public OEVibe energy is kcal/mol, so one of the two has to be converted.
HARTREE_TO_KCAL = 627.5094740631

# The preset this module runs, and the xtb --acc that the references were generated at
# (scripts/acceptance/molecules.py maps STANDARD to normal/1.0). A reference regenerated at
# another accuracy would no longer be comparable with these runs, so the pairing is asserted.
PRESET = "standard"
PRESET_ACCURACY = "1.0"

# Measured disagreements with spec 12.3, recorded as strict xfails rather than absorbed into a
# tolerance: spec 12.3 sets the tolerances and names the molecule set, and both are the user's
# to change. The numbers are from the run that first measured them.
KNOWN_DIVERGENCES: dict[tuple[str, str], str] = {
    ("glycine-hessian", "above the floor"): (
        "62.77 cm^-1, the lowest reference mode above the 50 cm^-1 floor anywhere in the set, "
        "comes back at 64.90 cm^-1: 2.132 cm^-1 against the 2.0 cm^-1 tolerance. The next "
        "worst mode in the whole set is 0.833 cm^-1 out, so this is the floor sitting low "
        "enough to admit a soft mode, not a broken comparison."
    ),
}


def _manifest() -> dict:
    """:returns: The committed manifest, or an empty mapping when it is not there.

    The mapping is read at import so each case is its own test id. An empty one would let this
    module pass by collecting nothing, so every group below turns one into a sentinel parameter
    that fails loudly; the ``acceptance_manifest`` fixture reports the missing file itself.
    """
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


_MANIFEST = _manifest()
_CASES = _MANIFEST.get("cases", [])
MOLECULES = {entry["title"]: entry for entry in _MANIFEST.get("molecules", [])}
FREQUENCY_CASES = {case["id"]: case for case in _CASES if case["kind"] == "frequencies"}
SINGLE_POINT_CASES = {
    case["id"]: case for case in _CASES if case["kind"] == "single_point"
}
CONSTRAINED = {
    title: entry for title, entry in MOLECULES.items() if entry["constraints"]
}
# Each molecule's gas-phase STANDARD optimization, which is xtb's own minimum for it. The GFN0
# references are single points at the UNoptimized input, so this is a geometry the tolerance
# has to be able to tell apart from the one it is pinned at.
MINIMUM_GEOMETRIES = {
    case["title"]: case["geometry"]
    for case in _CASES
    if case["kind"] == "optimization"
    and case["preset"] == PRESET
    and case["solvent"] is None
}

MISSING_FREQUENCY = "the-committed-manifest-lists-no-frequency-reference"
MISSING_SINGLE_POINT = "the-committed-manifest-lists-no-gfn0-single-point-reference"
MISSING_CONSTRAINED = "the-committed-manifest-lists-no-constrained-molecule"


def _parameters(cases: dict, missing: str, criterion: str = "") -> list:
    """:returns: One parameter per case, xfailed where this criterion is known to diverge."""
    if not cases:
        return [pytest.param(missing, id=missing)]
    parameters = []
    for identifier in cases:
        reason = KNOWN_DIVERGENCES.get((identifier, criterion))
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        parameters.append(pytest.param(identifier, marks=marks, id=identifier))
    return parameters


def _present(identifier: str, cases: dict, what: str) -> dict:
    """:returns: The entry, failing the test when the committed manifest did not supply one."""
    assert identifier in cases, (
        f"{MANIFEST_PATH} lists no {what}; it is committed data, regenerate it with "
        "scripts/acceptance/generate_references.py"
    )
    return cases[identifier]


def _coordinates_of(mol) -> np.ndarray:
    """:returns: An (N, 3) array of the molecule's coordinates in the OpenEye index layout.

    :raises AssertionError: When the molecule's OpenEye indices are not 0..N-1, which is what
        lets the manifest's atom indices address rows of this array directly.
    """
    assert mol.GetMaxAtomIdx() == mol.NumAtoms(), (
        "this molecule's OpenEye atom indices are not contiguous, so the manifest's atom "
        "indices do not address rows of the coordinate array"
    )
    values = oechem.OEDoubleArray(3 * mol.GetMaxAtomIdx())
    mol.GetCoords(values)
    return np.fromiter(values, dtype=float, count=len(values)).reshape(-1, 3)


def _result_coordinates(result) -> np.ndarray:
    """:returns: An (N, 3) array of a result's coordinates, in the same OpenEye layout."""
    coordinates = list(result.GetCoordinates())
    return np.asarray(coordinates, dtype=float).reshape(-1, 3)


def _dihedral_degrees(points) -> float:
    """:returns: The i-j-k-l torsion in degrees, in (-180, 180].

    Written out here rather than taken from ``oevibe``: ``Constraints.cpp`` normalizes a
    dihedral target with ``normalize_dihedral_degrees``, so measuring the result with the same
    function would let one error cancel the other.
    """
    p = [np.asarray(point, dtype=float) for point in points]
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _measure(coordinates: np.ndarray, constraint: dict) -> float:
    """:returns: The constrained quantity: Angstrom for a distance, degrees for a dihedral."""
    atoms = constraint["atoms"]
    if constraint["kind"] == "distance":
        return float(math.dist(coordinates[atoms[0]], coordinates[atoms[1]]))
    if constraint["kind"] == "dihedral":
        return _dihedral_degrees([coordinates[atom] for atom in atoms])
    raise AssertionError(f"no measurement for a {constraint['kind']} constraint")


@dataclasses.dataclass(frozen=True)
class Deviation:
    """How far one constraint sits from the value it is supposed to hold."""

    label: str
    amount: float
    tolerance: float


def _deviations(
    coordinates: np.ndarray,
    molecule: dict,
    start: np.ndarray,
    tolerances: dict,
) -> list[Deviation]:
    """Every constraint of one molecule, measured on one set of coordinates.

    A frozen atom's reference is its own position in the input, which is why ``start`` is
    needed; a distance's and a dihedral's is the value the manifest measured there.

    :param coordinates: The (N, 3) coordinates to measure.
    :param molecule: The manifest's molecule entry, which carries the constraints.
    :param start: The (N, 3) input coordinates the constraints were measured on.
    :param tolerances: The manifest's tolerance block.
    :returns: One ``Deviation`` per constraint, in manifest order.
    """
    deviations = []
    for constraint in molecule["constraints"]:
        atoms = constraint["atoms"]
        if constraint["kind"] == "frozen":
            for atom in atoms:
                deviations.append(
                    Deviation(
                        f"frozen atom {atom}",
                        float(np.linalg.norm(coordinates[atom] - start[atom])),
                        tolerances["constraint_angstrom"],
                    )
                )
            continue
        measured = _measure(coordinates, constraint)
        if constraint["kind"] == "dihedral":
            deviations.append(
                Deviation(
                    f"dihedral {atoms}",
                    # IEEE remainder, so the comparison is the shorter way round the circle
                    # without reaching for the library's own normalization.
                    abs(math.remainder(measured - constraint["value"], 360.0)),
                    tolerances["constraint_degrees"],
                )
            )
        else:
            deviations.append(
                Deviation(
                    f"distance {atoms}",
                    abs(measured - constraint["value"]),
                    tolerances["constraint_angstrom"],
                )
            )
    return deviations


@dataclasses.dataclass(frozen=True)
class Spectrum:
    """One frequency reference and the run measured against it."""

    status: int
    message: str
    atoms: int
    mine: tuple[float, ...]  # cm^-1, 3N-6 ascending, rigid-body modes projected out
    theirs: tuple[float, ...]  # cm^-1, the manifest's 3N values past the first six
    moved: float  # Angstrom, the furthest any atom travelled during the run


@functools.cache
def _spectrum(identifier: str) -> Spectrum:
    """Run OEVibe's Hessian at the geometry xtb took its own Hessian at.

    Memoized because each test below asserts on one part of the same run and the run costs 6N
    GFN2 gradients. ``optimize=False`` is load-bearing: the reference is ``xtb --hess`` at this
    exact geometry, so relaxing it first would compare two different stationary points.

    :returns: The measured and reference wavenumbers, paired index for index.
    """
    case = FREQUENCY_CASES[identifier]
    molecule = MOLECULES[case["title"]]
    mol = read_xyz_molecule(ACCEPTANCE_DIR / case["geometry"])
    atoms = mol.NumAtoms()
    before = _coordinates_of(mol)
    result = oevibe.frequencies(
        mol,
        method="gfn2-xtb",
        optimize=False,
        preset=PRESET,
        charge=molecule["charge"],
    ).Get(0)
    # Measured on the molecule rather than on the options, because what matters is where the
    # Hessian was taken, and oevibe.frequencies writes an optimized geometry back.
    moved = float(np.linalg.norm(_coordinates_of(mol) - before, axis=1).max())
    reference = tuple(float(value) for value in case["wavenumbers_cm1"][6:])
    if result.GetStatus() != oevibe.ConformerStatus.OK:
        return Spectrum(
            result.GetStatus(), result.GetMessage(), atoms, (), reference, moved
        )
    return Spectrum(
        result.GetStatus(),
        result.GetMessage(),
        atoms,
        tuple(float(value) for value in result.GetFrequencies()),
        reference,
        moved,
    )


@dataclasses.dataclass(frozen=True)
class SinglePoint:
    """One GFN0 reference and the two runs measured against it."""

    status: int
    message: str
    hartree: float  # at the geometry the reference was taken at
    reference: float
    at_the_minimum: float  # the same method at the molecule's xtb STANDARD geometry


@functools.cache
def _single_point(identifier: str) -> SinglePoint:
    """GFN0 at the reference's own geometry, and at the molecule's xtb minimum.

    The second energy is the discriminating one. A 1e-6 Eh claim is only worth making if the
    comparison can tell this geometry from another, and the xtb minimum is the nearest other
    geometry the committed data holds for the same molecule.
    """
    case = SINGLE_POINT_CASES[identifier]
    molecule = MOLECULES[case["title"]]

    def energy(geometry: str) -> float:
        mol = read_xyz_molecule(ACCEPTANCE_DIR / geometry)
        result = oevibe.energy(
            mol, method="gfn0-xtb", preset=PRESET, charge=molecule["charge"]
        ).Get(0)
        if result.GetStatus() != oevibe.ConformerStatus.OK:
            raise AssertionError(f"{identifier}: {result.GetMessage()}")
        return result.GetEnergy() / HARTREE_TO_KCAL

    mol = read_xyz_molecule(ACCEPTANCE_DIR / case["geometry"])
    result = oevibe.energy(
        mol, method="gfn0-xtb", preset=PRESET, charge=molecule["charge"]
    ).Get(0)
    if result.GetStatus() != oevibe.ConformerStatus.OK:
        return SinglePoint(
            result.GetStatus(),
            result.GetMessage(),
            float("nan"),
            case["energy_hartree"],
            float("nan"),
        )
    minimum = MINIMUM_GEOMETRIES.get(case["title"])
    return SinglePoint(
        result.GetStatus(),
        result.GetMessage(),
        result.GetEnergy() / HARTREE_TO_KCAL,
        case["energy_hartree"],
        energy(minimum) if minimum else float("nan"),
    )


@dataclasses.dataclass(frozen=True)
class ConstrainedRun:
    """One molecule optimized with its constraints, and again without them."""

    status: int
    message: str
    converged: bool
    iterations: int
    start: np.ndarray
    held: np.ndarray  # the constrained optimization's coordinates
    free: np.ndarray  # the same input optimized with no constraints at all
    free_converged: bool


@functools.cache
def _constrained_run(title: str) -> ConstrainedRun:
    """Optimize one constrained molecule's input twice, with and without its constraints.

    The unconstrained run is what makes the constrained one worth asserting. Every constraint
    target was measured on this very input, so holding it is also what doing nothing looks
    like; the free run measures how far the same quantity travels when nothing holds it.
    """
    molecule = CONSTRAINED[title]
    start = read_xyz_molecule(ACCEPTANCE_DIR / molecule["input"])
    # Read before the optimization, which writes its coordinates back into the molecule.
    start_coordinates = _coordinates_of(start)

    constraints = oevibe.ConstraintSet()
    frozen: list[int] = []
    for constraint in molecule["constraints"]:
        atoms = constraint["atoms"]
        if constraint["kind"] == "distance":
            constraints.AddDistance(atoms[0], atoms[1], constraint["value"])
        elif constraint["kind"] == "dihedral":
            constraints.AddDihedral(*atoms, constraint["value"])
        elif constraint["kind"] == "frozen":
            frozen.extend(atoms)
        else:
            raise AssertionError(f"unknown constraint kind {constraint['kind']!r}")
    if frozen:
        constraints.FreezeAtomIndices(frozen)

    held = oevibe.optimize(
        start,
        method="gfn2-xtb",
        preset=PRESET,
        charge=molecule["charge"],
        constraints=constraints,
    ).Get(0)
    free = oevibe.optimize(
        read_xyz_molecule(ACCEPTANCE_DIR / molecule["input"]),
        method="gfn2-xtb",
        preset=PRESET,
        charge=molecule["charge"],
    ).Get(0)
    empty = np.zeros_like(start_coordinates)
    stats = held.GetOptimizationStats() if held.HasOptimizationStats() else None
    return ConstrainedRun(
        status=held.GetStatus(),
        message=held.GetMessage(),
        converged=bool(stats.converged) if stats else False,
        iterations=int(stats.iterations) if stats else 0,
        start=start_coordinates,
        held=(
            _result_coordinates(held)
            if held.GetStatus() == oevibe.ConformerStatus.OK
            else empty
        ),
        free=(
            _result_coordinates(free)
            if free.GetStatus() == oevibe.ConformerStatus.OK
            else empty
        ),
        free_converged=(
            free.GetStatus() == oevibe.ConformerStatus.OK
            and bool(free.GetOptimizationStats().converged)
        ),
    )


# --- The settings every comparison below assumes ----------------------------------------


def test_every_reference_was_generated_at_the_settings_this_module_reruns(
    acceptance_manifest,
):
    """``PRESET`` is a constant here, so the data has to agree that it is the right one.

    ``test_acceptance_manifest.py`` pins each command line to the settings its own table
    calls for, which keeps a reference from being relabelled -- but it would accept the whole
    table being regenerated at a different accuracy, and then these runs would be comparing
    GFN2 at one accuracy with xtb at another and reporting the gap as an OEVibe defect. The
    xtb accuracy that STANDARD means is 1.0 (``scripts/acceptance/molecules.py``), and the
    Hamiltonian each kind reruns is in its ``--gfn`` argument.
    """
    hamiltonians = {"frequencies": "2", "single_point": "0"}
    checked = 0
    for case in acceptance_manifest["cases"]:
        expected = hamiltonians.get(case["kind"])
        if expected is None:
            continue
        command = case["command"]
        assert "--gfn" in command and "--acc" in command, (case["id"], command)
        assert command[command.index("--gfn") + 1] == expected, (case["id"], command)
        assert command[command.index("--acc") + 1] == PRESET_ACCURACY, (
            case["id"],
            command,
        )
        checked += 1
    assert checked == len(FREQUENCY_CASES) + len(SINGLE_POINT_CASES)
    assert checked, "the manifest holds no reference for this module to check"


# --- Frequencies (spec 12.3) ------------------------------------------------------------


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("identifier", _parameters(FREQUENCY_CASES, MISSING_FREQUENCY))
def test_the_frequency_analysis_returns_one_mode_per_vibration(identifier):
    """3N-6 modes on each side, both ascending, so mode i means the same mode in both.

    The two sides lay their modes out differently and the tolerance test below pairs them
    index for index, so the slice and the ordering it depends on are asserted here instead of
    being left to a tolerance miss to announce. ``test_acceptance_manifest.py`` pins the 3N
    length and the ascending order on the reference side; what is new here is that OEVibe's
    3N-6 list lines up with it.
    """
    _present(identifier, FREQUENCY_CASES, "frequency reference")
    spectrum = _spectrum(identifier)
    assert spectrum.status == oevibe.ConformerStatus.OK, spectrum.message
    assert len(spectrum.theirs) == 3 * spectrum.atoms - 6, (
        f"{identifier}: the manifest holds {len(spectrum.theirs)} vibrations past the six "
        f"rigid-body modes, not the {3 * spectrum.atoms - 6} a non-linear {spectrum.atoms}-"
        "atom molecule has"
    )
    assert len(spectrum.mine) == len(spectrum.theirs), (
        f"{identifier}: {len(spectrum.mine)} modes against xtb's {len(spectrum.theirs)}"
    )
    assert list(spectrum.mine) == sorted(spectrum.mine), identifier
    assert list(spectrum.theirs) == sorted(spectrum.theirs), identifier


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("identifier", _parameters(FREQUENCY_CASES, MISSING_FREQUENCY))
def test_the_hessian_is_taken_at_the_committed_xtb_geometry(identifier):
    """The reference is ``xtb --hess`` at one geometry, so OEVibe has to use that one.

    No frequency number can show that it did, and the committed data cannot show it either:
    asking for the optimization first changes nothing here, because OEVibe's optimizer accepts
    all five committed geometries without an accepted step. So the tolerance test would go on
    passing if a later change made the frequency verb relax its input, and the agreement would
    quietly become a statement about two different structures. ``optimize=False`` is checked
    here for what it does -- leave the molecule's coordinates alone -- rather than trusted
    because it is written in the call. Measured: the same call from a geometry that is not a
    minimum (imidazole's OMEGA input) moves an atom 2.8e-2 Angstrom, which this rejects.
    """
    _present(identifier, FREQUENCY_CASES, "frequency reference")
    spectrum = _spectrum(identifier)
    assert spectrum.status == oevibe.ConformerStatus.OK, spectrum.message
    assert spectrum.moved == 0.0, (
        f"{identifier}: the frequency run moved an atom {spectrum.moved:.3e} Angstrom, so "
        "the Hessian was not taken at the committed geometry xtb took its own at"
    )


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize(
    "identifier", _parameters(FREQUENCY_CASES, MISSING_FREQUENCY, "above the floor")
)
def test_the_modes_above_the_floor_match_xtb_hess(identifier, acceptance_manifest):
    """Spec 12.3: within the manifest's 2 cm^-1 for modes above its 50 cm^-1 floor.

    Which modes are compared is decided by xtb's values alone. Selecting on OEVibe's own
    frequencies would let a mode that came back wrong drop itself out of the comparison, and
    selecting on either side's magnitude would pull in the imaginary modes the floor exists to
    keep out -- spec 12.3 says "modes above 50 cm^-1", and -113 cm^-1 is not above 50. That
    second reading is not academic: measured, it fails urea on its -113.65 cm^-1 mode by
    3.39 cm^-1, which is the only thing either reading changes on this reference set.
    """
    _present(identifier, FREQUENCY_CASES, "frequency reference")
    spectrum = _spectrum(identifier)
    assert spectrum.status == oevibe.ConformerStatus.OK, spectrum.message
    tolerances = acceptance_manifest["tolerances"]
    floor = tolerances["frequency_floor_cm1"]
    assert len(spectrum.mine) == len(spectrum.theirs), (
        f"{identifier}: {len(spectrum.mine)} modes against xtb's {len(spectrum.theirs)}, so "
        "no mode can be paired with its reference"
    )
    pairs = [
        (abs(mine - theirs), mine, theirs)
        for mine, theirs in zip(spectrum.mine, spectrum.theirs)
        if theirs > floor
    ]
    assert pairs, f"{identifier}: xtb reports no mode above {floor} cm^-1 to compare"
    worst, mine, theirs = max(pairs)
    assert worst < tolerances["frequency_cm1"], (
        f"{identifier}: {mine:.3f} cm^-1 against xtb's {theirs:.3f} cm^-1, {worst:.3f} cm^-1 "
        f"apart over {len(pairs)} modes above the {floor} cm^-1 floor"
    )


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("identifier", _parameters(FREQUENCY_CASES, MISSING_FREQUENCY))
def test_the_imaginary_modes_below_the_floor_match_xtb_hess(
    identifier, acceptance_manifest
):
    """As many modes below -50 cm^-1 as xtb found, which for urea is two and not zero.

    Counted from ``GetFrequencies()`` rather than taken from ``GetImaginaryModeCount()``: that
    counts every negative frequency, while the manifest's field counts only those past the
    floor, and the two numbers would look comparable while meaning different things. Measured
    limit: on the committed set the two definitions happen to agree, because urea's two
    imaginary modes are both well past the floor and no reference has a negative frequency
    inside it, so this test cannot by itself catch a reader that reached for the wrong one.
    Counting here is what keeps the right definition in place.

    The count is asserted and the values are not, because spec 12.3's tolerance covers "modes
    above 50 cm^-1" and these are below -50. Measured on urea, the one case that has them:
    -316.86 against xtb's -316.04, and -117.04 against -113.65. The second is 3.39 cm^-1 out,
    which the 2 cm^-1 tolerance would fail if the comparison reached down here -- a saddle
    point's softest imaginary mode is not where the tblite/xtb agreement S5 measured holds,
    and spec 12.3 does not ask it to.
    """
    case = _present(identifier, FREQUENCY_CASES, "frequency reference")
    spectrum = _spectrum(identifier)
    assert spectrum.status == oevibe.ConformerStatus.OK, spectrum.message
    floor = acceptance_manifest["tolerances"]["frequency_floor_cm1"]
    mine = [value for value in spectrum.mine if value < -floor]
    assert len(mine) == case["imaginary_modes_below_floor"], (
        f"{identifier}: {len(mine)} modes below -{floor} cm^-1 ({mine}) against xtb's "
        f"{case['imaginary_modes_below_floor']}"
    )


def test_some_frequency_reference_has_an_imaginary_mode_below_the_floor(
    acceptance_manifest,
):
    """The imaginary-mode check has to be exercised by the reference set, not just defined.

    Every committed reference but urea is a minimum, so without a case that has imaginary
    modes the test above would assert 0 == 0 five times and pass on an engine that could not
    report an imaginary mode at all. If urea is ever dropped or re-optimized into a minimum,
    this fails and says so rather than letting the check quietly go vacuous.
    """
    cases = [
        case for case in acceptance_manifest["cases"] if case["kind"] == "frequencies"
    ]
    assert cases, "the manifest holds no frequency reference at all"
    with_imaginary = [
        case["id"] for case in cases if case["imaginary_modes_below_floor"] > 0
    ]
    assert with_imaginary, (
        "no committed frequency reference has a mode below the floor any more, so "
        "test_the_imaginary_modes_below_the_floor_match_xtb_hess now only ever compares zero "
        "with zero; the set needs a case that is not a minimum"
    )


# --- GFN0-xTB single points (spec 12.3) -------------------------------------------------


@pytest.mark.usefixtures("xtb_library")
@pytest.mark.parametrize(
    "identifier", _parameters(SINGLE_POINT_CASES, MISSING_SINGLE_POINT)
)
def test_the_gfn0_single_point_matches_xtb_sp(identifier, acceptance_manifest):
    """Spec 12.3: within 1e-6 Eh, "since they share the same library"."""
    _present(identifier, SINGLE_POINT_CASES, "GFN0 single-point reference")
    single_point = _single_point(identifier)
    assert single_point.status == oevibe.ConformerStatus.OK, single_point.message
    tolerance = acceptance_manifest["tolerances"]["single_point_hartree"]
    difference = single_point.hartree - single_point.reference
    assert abs(difference) < tolerance, (
        f"{identifier}: {single_point.hartree:.9f} Eh against xtb's "
        f"{single_point.reference:.9f} Eh, {difference:+.3e} Eh apart"
    )


@pytest.mark.usefixtures("xtb_library")
@pytest.mark.parametrize(
    "identifier", _parameters(SINGLE_POINT_CASES, MISSING_SINGLE_POINT)
)
def test_the_gfn0_tolerance_resolves_the_geometry_it_runs_at(
    identifier, acceptance_manifest
):
    """The 1e-6 Eh claim is only worth making if it can tell this geometry from another.

    The references are single points at the unoptimized input, and the committed data holds
    one other geometry for each of these molecules: xtb's own STANDARD minimum. Running the
    same method there has to miss the reference by more than the tolerance, which makes the
    agreement asserted above a statement about this geometry rather than about the molecule.
    """
    _present(identifier, SINGLE_POINT_CASES, "GFN0 single-point reference")
    single_point = _single_point(identifier)
    assert single_point.status == oevibe.ConformerStatus.OK, single_point.message
    assert math.isfinite(single_point.at_the_minimum), (
        f"{identifier}: the manifest holds no gas-phase {PRESET} optimization for this "
        "molecule, so there is no second geometry to separate the reference from"
    )
    tolerance = acceptance_manifest["tolerances"]["single_point_hartree"]
    difference = single_point.at_the_minimum - single_point.reference
    assert abs(difference) > tolerance, (
        f"{identifier}: GFN0 at the xtb minimum is {difference:+.3e} Eh from the reference at "
        f"the input geometry, inside the {tolerance} Eh tolerance, so this case cannot tell "
        "the two geometries apart"
    )


# --- Constrained optimizations (spec 12.3) ----------------------------------------------


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("title", _parameters(CONSTRAINED, MISSING_CONSTRAINED))
def test_the_constrained_optimization_does_real_work(title, acceptance_manifest):
    """A no-op would hold every constraint, so the run has to be an optimization first.

    Each target was measured on the input geometry this run starts from, so an optimizer that
    returned its input, or stopped before its first accepted step, would satisfy the criterion
    below perfectly. Convergence, at least one accepted iteration, and an atom that actually
    moved are all required before that criterion means anything.
    """
    _present(title, CONSTRAINED, "constrained molecule")
    run = _constrained_run(title)
    assert run.status == oevibe.ConformerStatus.OK, run.message
    assert run.converged, run.message
    assert run.iterations >= 1
    moved = float(np.linalg.norm(run.held - run.start, axis=1).max())
    tolerance = acceptance_manifest["tolerances"]["constraint_angstrom"]
    assert moved > tolerance, (
        f"{title}: the constrained optimization moved no atom more than {moved:.3e} "
        f"Angstrom, within the {tolerance} Angstrom constraint tolerance, so holding the "
        "constraints says nothing"
    )


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("title", _parameters(CONSTRAINED, MISSING_CONSTRAINED))
def test_every_constraint_is_held_to_the_spec_tolerance(title, acceptance_manifest):
    """Spec 12.3: every constraint held within 1e-4 Angstrom or 1e-3 degrees.

    A frozen atom's reference is its own position in the hash-pinned input, and a distance's
    or a dihedral's is the value the manifest measured there. No xtb run holds a constraint --
    the ``*-constrained`` optimization references are unconstrained and byte-identical to
    their twins -- so this criterion is self-referential by construction, which is why it is
    bracketed by the two tests either side of it.
    """
    molecule = _present(title, CONSTRAINED, "constrained molecule")
    run = _constrained_run(title)
    assert run.status == oevibe.ConformerStatus.OK, run.message
    tolerances = acceptance_manifest["tolerances"]
    deviations = _deviations(run.held, molecule, run.start, tolerances)
    assert deviations, f"{title}: the manifest records no constraint for this molecule"
    for deviation in deviations:
        assert deviation.amount < deviation.tolerance, (
            f"{title}: {deviation.label} moved {deviation.amount:.3e} from the value it was "
            f"constrained to, past the {deviation.tolerance} it is allowed"
        )


@pytest.mark.usefixtures("tblite_library")
@pytest.mark.parametrize("title", _parameters(CONSTRAINED, MISSING_CONSTRAINED))
def test_an_unconstrained_optimization_moves_every_constraint(
    title, acceptance_manifest
):
    """The same optimization without the constraints must break every one of them.

    This is what makes the criterion above an assertion about the constraints rather than
    about the starting geometry. If a quantity would have stayed at its target anyway, holding
    it to 1e-4 Angstrom measures nothing, and the suite should say so instead of reporting a
    pass. Measured today: 0.0090 Angstrom for aspirin's constrained distance, 0.130 Angstrom
    for its frozen atom, and 2.12 degrees for lidocaine's dihedral.
    """
    molecule = _present(title, CONSTRAINED, "constrained molecule")
    run = _constrained_run(title)
    assert run.free_converged, (
        f"{title}: the unconstrained optimization did not converge, so it cannot show what "
        "the constraints are holding against"
    )
    tolerances = acceptance_manifest["tolerances"]
    deviations = _deviations(run.free, molecule, run.start, tolerances)
    assert deviations, f"{title}: the manifest records no constraint for this molecule"
    for deviation in deviations:
        assert deviation.amount > deviation.tolerance, (
            f"{title}: unconstrained, {deviation.label} still ends within "
            f"{deviation.amount:.3e} of its target, inside the {deviation.tolerance} the "
            "constrained run is asked to hold it to, so that run proves nothing"
        )


@pytest.mark.parametrize("title", _parameters(CONSTRAINED, MISSING_CONSTRAINED))
def test_the_recorded_constraint_value_is_the_one_this_module_measures(
    title, acceptance_manifest
):
    """The manifest's targets and this module's measurements have to mean the same thing.

    ``test_acceptance_manifest.py`` already checks the recorded values against the xyz file
    read directly. What is unchecked until here is the step between: these tests apply and
    verify constraints through OpenEye atom indices on a molecule read by ``oechem``, and if
    that layout disagreed with the file order the manifest's indices refer to, every
    constraint would be placed on the wrong atoms -- and still held perfectly, because the
    target would be wrong in exactly the same way.
    """
    molecule = _present(title, CONSTRAINED, "constrained molecule")
    tolerances = acceptance_manifest["tolerances"]
    start = _coordinates_of(read_xyz_molecule(ACCEPTANCE_DIR / molecule["input"]))
    measured = [
        constraint for constraint in molecule["constraints"] if "value" in constraint
    ]
    assert measured, f"{title}: no constraint on this molecule records a target value"
    for constraint in measured:
        # A tenth of the constraint tolerance: close enough that the two are the same
        # quantity, loose enough for the single-precision coordinates oechem hands back.
        allowed = (
            0.1
            * tolerances[
                "constraint_degrees"
                if constraint["kind"] == "dihedral"
                else "constraint_angstrom"
            ]
        )
        difference = _measure(start, constraint) - constraint["value"]
        assert abs(difference) < allowed, (
            f"{title}: the {constraint['kind']} on atoms {constraint['atoms']} measures "
            f"{difference:+.3e} from the value the manifest recorded for the same input"
        )
