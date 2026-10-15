#!/usr/bin/env python
"""Generate the committed acceptance references with the xtb binary (spec 12.3).

Run from the repository root with the project's interpreter and the oevibe environment's xtb:

    .venv/bin/python scripts/acceptance/generate_references.py \
        --xtb "$MAMBA_ROOT_PREFIX/envs/oevibe/bin/xtb"

It writes tests/data/acceptance/, which is committed; the acceptance tests never call xtb. An
existing input.xyz is reused unless --regenerate-inputs is given, so the reference geometries do
not move when OMEGA changes.

Every xtb run gets XTBPATH pointing at the parameter directory beside the binary and no XTBHOME,
because GFN0-xTB's parameters are not compiled in and a stray .xtbrc would silently change the
settings. Neither path reaches the manifest: it records versions and settings, not this machine.

A molecule whose run fails is reported and skipped rather than aborting the whole generation, and
--only merges into the manifest already on disk, so a failed molecule can be redone on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molecules import DATA_DIR, MOLECULES, PRESET_LEVELS, AcceptanceMolecule

CONVERGED = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS")
TOTAL_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
XYZ_ENERGY = re.compile(r"energy:\s*(-?\d+\.\d+)")
VERSION_LINE = re.compile(r"(xtb version .*?)(?:\s+compiled by|$)")

TOLERANCES = {
    "rmsd_angstrom": 0.05,
    "energy_kcal_per_mol": 0.01,
    "step_ratio": 1.3,
    "frequency_cm1": 2.0,
    "frequency_floor_cm1": 50.0,
    "constraint_angstrom": 1e-4,
    "constraint_degrees": 1e-3,
    "single_point_hartree": 1e-6,
}


class GenerationError(Exception):
    """One case could not be generated; the run reports it and carries on."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_input(molecule: AcceptanceMolecule, path: Path) -> None:
    """Write a 3D structure for ``molecule``, prepared exactly as OEVibe would prepare it."""
    from openeye import oechem, oeomega

    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, molecule.smiles):
        raise SystemExit(f"cannot parse SMILES for {molecule.title}")
    options = oeomega.OEOmegaOptions()
    options.SetMaxConfs(1)
    options.SetStrictStereo(False)
    if oeomega.OEOmega(options).Build(mol) != oeomega.OEOmegaReturnCode_Success:
        raise SystemExit(f"OMEGA failed for {molecule.title}")
    oechem.OEAddExplicitHydrogens(mol)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = oechem.oemolostream(str(path))
    oechem.OEWriteMolecule(stream, mol)
    stream.close()


def xtb_environment(xtb: str) -> dict[str, str]:
    """:returns: The environment every xtb run gets.

    GFN0-xTB's parameters live in a file rather than in the binary, so without XTBPATH xtb stops
    with "Parameter file param_gfn0-xtb.txt not found!". The directory is derived from the binary
    so that no machine-specific path has to be supplied, and XTBHOME is dropped so that a .xtbrc
    found through it cannot change a setting the manifest claims to record.

    :raises SystemExit: When the parameter directory is not beside the binary.
    """
    parameters = Path(xtb).resolve().parents[1] / "share" / "xtb"
    if not (parameters / "param_gfn0-xtb.txt").is_file():
        raise SystemExit(f"{parameters} holds no xtb parameter files")
    environment = dict(os.environ)
    environment.pop("XTBHOME", None)
    environment["XTBPATH"] = str(parameters)
    return environment


def run_xtb(
    xtb: str,
    environment: dict[str, str],
    input_xyz: Path,
    arguments: list[str],
    workdir: Path,
) -> tuple[str, list[str]]:
    """:returns: xtb's stdout and the exact command line, run in an empty working directory.

    :raises GenerationError: When xtb exits non-zero.
    """
    shutil.copy(input_xyz, workdir / "input.xyz")
    command = [xtb, "input.xyz", *arguments]
    finished = subprocess.run(
        command,
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    if finished.returncode != 0:
        tail = "\n".join((finished.stdout + finished.stderr).splitlines()[-15:])
        raise GenerationError(f"xtb {' '.join(arguments)} failed:\n{tail}")
    return finished.stdout, ["xtb", "input.xyz", *arguments]


def read_xyz(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
    return atoms


def read_vibspectrum(path: Path) -> list[float]:
    """:returns: Every wavenumber in cm^-1, in file order, including the six near-zero modes.

    A vibspectrum row is ``mode [symmetry] wavenumber intensity selection-rules``; the symmetry
    column is blank for C1, so the wavenumber is the second number on the row either way.
    """
    wavenumbers = []
    for line in path.read_text().splitlines():
        if line.startswith(("$", "#")) or not line.strip():
            continue
        numbers = []
        for token in line.split():
            try:
                numbers.append(float(token))
            except ValueError:
                continue
        if len(numbers) >= 2:
            wavenumbers.append(numbers[1])
    return wavenumbers


def optimization_case(
    xtb: str,
    environment: dict[str, str],
    molecule: AcceptanceMolecule,
    input_xyz: Path,
    preset: str,
    solvated: bool,
    out_dir: Path,
    scratch: str | None,
) -> dict:
    level, accuracy = PRESET_LEVELS[preset]
    arguments = [
        "--opt",
        level,
        "--gfn",
        "2",
        "--acc",
        accuracy,
        "--chrg",
        str(molecule.charge),
        "--uhf",
        "0",
    ]
    if solvated:
        arguments += ["--alpb", "water"]
    case_id = f"{molecule.title}-{preset}" + ("-water" if solvated else "")
    with tempfile.TemporaryDirectory(dir=scratch) as workdir_name:
        workdir = Path(workdir_name)
        stdout, command = run_xtb(xtb, environment, input_xyz, arguments, workdir)
        optimized = workdir / "xtbopt.xyz"
        if not optimized.is_file():
            raise GenerationError(f"{case_id}: xtb wrote no xtbopt.xyz")
        geometry = out_dir / f"{case_id}.xyz"
        shutil.copy(optimized, geometry)
        cycles = CONVERGED.search(stdout)
        energy = XYZ_ENERGY.search(optimized.read_text().splitlines()[1])
    if cycles is None or energy is None:
        raise GenerationError(
            f"{case_id}: xtb did not report convergence and an energy"
        )
    return {
        "id": case_id,
        "title": molecule.title,
        "kind": "optimization",
        "preset": preset,
        "solvent": "water" if solvated else None,
        "command": command,
        "cycles": int(cycles.group(1)),
        "energy_hartree": float(energy.group(1)),
        "geometry": f"{molecule.title}/{geometry.name}",
        "geometry_sha256": sha256(geometry),
    }


def frequency_case(
    xtb: str,
    environment: dict[str, str],
    molecule: AcceptanceMolecule,
    out_dir: Path,
    scratch: str | None,
) -> dict:
    """A Hessian at xtb's own optimized STANDARD geometry, which is what OEVibe will compare at."""
    optimized = out_dir / f"{molecule.title}-standard.xyz"
    arguments = [
        "--hess",
        "--gfn",
        "2",
        "--acc",
        "1.0",
        "--chrg",
        str(molecule.charge),
        "--uhf",
        "0",
    ]
    with tempfile.TemporaryDirectory(dir=scratch) as workdir_name:
        workdir = Path(workdir_name)
        _, command = run_xtb(xtb, environment, optimized, arguments, workdir)
        spectrum = workdir / "vibspectrum"
        if not spectrum.is_file():
            raise GenerationError(f"{molecule.title}: xtb --hess wrote no vibspectrum")
        wavenumbers = read_vibspectrum(spectrum)
    expected = 3 * len(read_xyz(optimized))
    if len(wavenumbers) != expected:
        raise GenerationError(
            f"{molecule.title}: vibspectrum holds {len(wavenumbers)} modes, not {expected}"
        )
    floor = TOLERANCES["frequency_floor_cm1"]
    return {
        "id": f"{molecule.title}-hessian",
        "title": molecule.title,
        "kind": "frequencies",
        "geometry": f"{molecule.title}/{optimized.name}",
        "command": command,
        # Recorded rather than assumed away: the geometry an optimizer reaches is not always a
        # minimum, and a consumer that took every reference frequency to be real would be wrong.
        "imaginary_modes": sum(1 for value in wavenumbers[6:] if value < -floor),
        "wavenumbers_cm1": wavenumbers,
    }


def single_point_case(
    xtb: str,
    environment: dict[str, str],
    molecule: AcceptanceMolecule,
    input_xyz: Path,
    scratch: str | None,
) -> dict:
    """GFN0-xTB at the input geometry; OEVibe reaches the same library, so the tolerance is 1e-6 Eh."""
    arguments = [
        "--sp",
        "--gfn",
        "0",
        "--acc",
        "1.0",
        "--chrg",
        str(molecule.charge),
        "--uhf",
        "0",
    ]
    with tempfile.TemporaryDirectory(dir=scratch) as workdir_name:
        stdout, command = run_xtb(
            xtb, environment, input_xyz, arguments, Path(workdir_name)
        )
    energy = TOTAL_ENERGY.search(stdout)
    if energy is None:
        raise GenerationError(f"{molecule.title}: xtb --sp printed no total energy")
    return {
        "id": f"{molecule.title}-gfn0-sp",
        "title": molecule.title,
        "kind": "single_point",
        "method": "gfn0-xtb",
        "geometry": f"{molecule.title}/input.xyz",
        "command": command,
        "energy_hartree": float(energy.group(1)),
    }


def constraint_reference(molecule: AcceptanceMolecule, input_xyz: Path) -> list[dict]:
    """:returns: The constraints with the value each one must hold, measured on the input."""
    atoms = read_xyz(input_xyz)
    measured = []
    for constraint in molecule.constraints:
        indices = constraint["atoms"]
        entry: dict = {"kind": constraint["kind"], "atoms": list(indices)}
        if constraint["kind"] == "distance":
            a, b = (atoms[i][1:] for i in indices)
            entry["value"] = math.dist(a, b)
        elif constraint["kind"] == "dihedral":
            entry["value"] = dihedral_degrees([atoms[i][1:] for i in indices])
        measured.append(entry)
    return measured


def dihedral_degrees(points: list[tuple[float, ...]]) -> float:
    """:returns: The i-j-k-l torsion in degrees, in (-180, 180]."""
    import numpy as np

    p = [np.array(point, dtype=float) for point in points]
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def xtb_version(xtb: str) -> str:
    """:returns: The xtb banner's version line, which carries the 6.7.1 the spec pins.

    The tail of that line names the machine and the account xtb was built on; it is cut, because
    the manifest records which xtb produced the data, not where anyone's copy came from.
    """
    finished = subprocess.run(
        [xtb, "--version"], capture_output=True, text=True, check=False
    )
    for line in (finished.stdout + finished.stderr).splitlines():
        found = VERSION_LINE.search(" ".join(line.split()))
        if found is not None:
            return found.group(1)
    raise SystemExit("xtb --version printed no version line")


def tblite_version() -> str:
    """:returns: "<version> build <build>" for tblite in the oevibe environment.

    ``micromamba list`` without --json prints a table whose first line is a heading, so the
    package row is parsed from the JSON form instead. Older micromamba releases print a bare
    array there and current ones wrap it in an object, so both are accepted.
    """
    finished = subprocess.run(
        ["micromamba", "list", "-n", "oevibe", "tblite", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        raise SystemExit(
            f"micromamba list failed:\n{finished.stdout}\n{finished.stderr}"
        )
    payload = json.loads(finished.stdout)
    packages = payload["packages"] if isinstance(payload, dict) else payload
    for package in packages:
        if package.get("name") == "tblite":
            build = package.get("build_string", package.get("build", "?"))
            return f"{package['version']} build {build}"
    raise SystemExit("tblite is not installed in the oevibe environment")


def molecule_entry(molecule: AcceptanceMolecule, input_xyz: Path) -> dict:
    return {
        "title": molecule.title,
        "smiles": molecule.smiles,
        "charge": molecule.charge,
        "multiplicity": 1,
        "input": f"{molecule.title}/input.xyz",
        "input_sha256": sha256(input_xyz),
        "constraints": constraint_reference(molecule, input_xyz),
    }


def previous_manifest() -> dict:
    """:returns: The manifest already on disk, or empty lists when there is none.

    Generating the whole set takes minutes, so --only has to leave the molecules it skipped in
    place rather than publishing a manifest that covers one molecule.
    """
    path = DATA_DIR / "manifest.json"
    if not path.is_file():
        return {"molecules": [], "cases": []}
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtb", required=True, help="path to the xtb 6.7.1 binary")
    parser.add_argument("--regenerate-inputs", action="store_true")
    parser.add_argument(
        "--only", action="append", default=[], help="limit to these titles"
    )
    parser.add_argument(
        "--scratch", help="directory to run xtb in (default: the system temp)"
    )
    arguments = parser.parse_args()

    titles = {molecule.title for molecule in MOLECULES}
    unknown = sorted(set(arguments.only) - titles)
    if unknown:
        raise SystemExit(f"--only names no such molecule: {', '.join(unknown)}")

    environment = xtb_environment(arguments.xtb)
    carried = previous_manifest()
    entries = {entry["title"]: entry for entry in carried["molecules"]}
    cases: dict[str, list[dict]] = {}
    for case in carried["cases"]:
        cases.setdefault(case["title"], []).append(case)

    failures = []
    started = time.monotonic()
    for molecule in MOLECULES:
        if arguments.only and molecule.title not in arguments.only:
            continue
        out_dir = DATA_DIR / molecule.title
        out_dir.mkdir(parents=True, exist_ok=True)
        input_xyz = out_dir / "input.xyz"
        if arguments.regenerate_inputs or not input_xyz.exists():
            build_input(molecule, input_xyz)
        clock = time.monotonic()
        print(f"{molecule.title}: {len(read_xyz(input_xyz))} atoms", end="", flush=True)

        generated = []
        try:
            for preset in molecule.presets:
                generated.append(
                    optimization_case(
                        arguments.xtb,
                        environment,
                        molecule,
                        input_xyz,
                        preset,
                        False,
                        out_dir,
                        arguments.scratch,
                    )
                )
            if molecule.solvated:
                generated.append(
                    optimization_case(
                        arguments.xtb,
                        environment,
                        molecule,
                        input_xyz,
                        "standard",
                        True,
                        out_dir,
                        arguments.scratch,
                    )
                )
            if molecule.frequencies:
                generated.append(
                    frequency_case(
                        arguments.xtb, environment, molecule, out_dir, arguments.scratch
                    )
                )
            if molecule.single_point:
                generated.append(
                    single_point_case(
                        arguments.xtb,
                        environment,
                        molecule,
                        input_xyz,
                        arguments.scratch,
                    )
                )
        except GenerationError as error:
            failures.append(f"{molecule.title}: {error}")
            print(
                f", FAILED after {time.monotonic() - clock:.1f} s\n  {error}",
                flush=True,
            )
            continue

        entries[molecule.title] = molecule_entry(molecule, input_xyz)
        cases[molecule.title] = generated
        print(
            f", {len(generated)} cases in {time.monotonic() - clock:.1f} s",
            flush=True,
        )

    ordered = [molecule.title for molecule in MOLECULES if molecule.title in entries]
    manifest = {
        "xtb_version": xtb_version(arguments.xtb),
        "tblite_version": tblite_version(),
        # Recorded relative to the binary's own prefix, so the record stays machine-independent.
        "xtb_environment": {"XTBPATH": "${xtb prefix}/share/xtb", "XTBHOME": None},
        "tolerances": TOLERANCES,
        "molecules": [entries[title] for title in ordered],
        "cases": [case for title in ordered for case in cases.get(title, [])],
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"wrote {DATA_DIR / 'manifest.json'}: {len(manifest['cases'])} cases "
        f"in {time.monotonic() - started:.1f} s"
    )
    if failures:
        print(f"{len(failures)} molecules failed and are not in the manifest:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
