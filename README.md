# OEVibe

A C++17 library with Python bindings that computes energies, gradients, optimized
geometries, frequencies and relaxed torsion scans for OpenEye molecules, with
semiempirical (GFN2-xTB, GFN1-xTB, GFN0-xTB) and force-field (MMFF94, MMFF94s, Sage,
Parsley) engines. Molecules created with `openeye.oechem` in Python pass natively to C++
and back without serialization: no SMILES round trip and no manual pointer handling.

## Prerequisites

- **OpenEye C++ SDK** 2026.1 -- Headers and libraries (download from
  [OpenEye](https://www.eyesopen.com/))
- **OpenEye Python Toolkits** -- `pip install openeye-toolkits` (2026.1 or newer)
- **CMake** >= 3.21
- **SWIG** >= 4.2.1
- **Python** >= 3.12
- **numpy** >= 2 -- a runtime dependency of the package, not only a build dependency

## Getting Started

### 1. Configure the OpenEye SDK Path

Set the `OPENEYE_ROOT` environment variable to point to your OpenEye C++ SDK
installation (the directory containing `include/` and `lib/`):

```bash
export OPENEYE_ROOT=/path/to/openeye/sdk
```

The CMake presets read this variable automatically. You can also set
`Python3_EXECUTABLE` if you need a specific Python interpreter.

Alternatively, create a `CMakeUserPresets.json` (gitignored) to override
these values permanently for your local machine.

### 2. Build the C++ Library and SWIG Bindings

The `debug` preset builds the static C++ library and the C++ tests:

```bash
cmake --preset debug
cmake --build build-debug
```

The Python extension comes from the `python` preset, which links the
`openeye-toolkits` wheel's shared libraries and writes the compiled module into
`python/oevibe/`:

```bash
cmake --preset python
cmake --build --preset python
```

To build the C++ library in release mode:

```bash
cmake --preset release
cmake --build build-release
```

### 3. Install for Development

The built extension is importable straight from the source tree, which is what the test
commands below do:

```bash
PYTHONPATH=python python -c "import oevibe; print(oevibe.__version__)"
```

To install the Python package in editable mode instead, so changes to the C++ code are
reflected after rebuilding:

```bash
pip install --config-settings editable_mode=compat -e python/
```

The `editable_mode=compat` flag is required because scikit-build-core's default
editable mode uses import hooks that are incompatible with compiled SWIG
extension modules. The compat mode installs the package as a traditional
`.egg-link`, which works reliably with native extensions.

## Quick Start

```python
from openeye import oechem, oeomega

import oevibe

mol = oechem.OEMol()
oechem.OESmilesToMol(mol, "CCCCO")
options = oeomega.OEOmegaOptions()
options.SetMaxConfs(5)
oeomega.OEOmega(options).Build(mol)

# Optimize every conformer with GFN2-xTB in implicit water, in place.
results = oevibe.optimize(mol, method="gfn2-xtb", solvent="water")
for result in results:
    print(result.GetConformerIndex(), result.GetStatus(), result.GetEnergy())

# Frequencies and thermochemistry; weights come from the free energies at a temperature.
thermo = oevibe.frequencies(mol, method="gfn2-xtb")
print(dict(thermo.GetBoltzmannWeights(298.15)))

# A relaxed dihedral scan over one torsion, every 30 degrees.
scan = oevibe.scan(mol, dihedrals=[(0, 1, 2, 3)], step=30.0, method="gfn2-xtb")
for index in range(scan.GetNumPoints()):
    point = scan.GetPoint(index)
    if point.HasRelativeEnergy():
        print(point.GetAngles()[0], point.GetRelativeEnergy())
```

`gfn2-xtb` runs through libtblite, which OEVibe loads at runtime. Either set
`OEVIBE_TBLITE_LIBRARY` to the library file, or run inside the `oevibe` environment
described below, whose `$CONDA_PREFIX/lib` the engine searches with no variable at all.
Without either, the call raises `EngineUnavailableError` listing every path it tried.
The force-field methods (`mmff94`, `mmff94s`, `sage`, `parsley`) need nothing extra: they
are linked into the extension.

## Verbs and Options

| Verb | Result |
|------|--------|
| `oevibe.energy(mol, ...)` | Single-point energy for every conformer. |
| `oevibe.gradient(mol, ...)` | Energy and gradient for every conformer. |
| `oevibe.optimize(mol, ...)` | Optimizes every conformer and writes the geometry back. |
| `oevibe.frequencies(mol, ...)` | Optimizes first (by default), then the Hessian, frequencies and thermochemistry. |
| `oevibe.scan(mol, dihedrals, ...)` | Relaxed scan over one or two dihedrals, returning a `ScanResult`. |

Every verb takes the same keyword options:

| Option | Meaning |
|--------|---------|
| `method` | Method name or `oevibe.Method`, for example `"gfn2-xtb"` or `"mmff94s"`. |
| `solvent` | `None` for gas phase, a solvent name, or an `oevibe.Solvent`. |
| `engine` | Engine name overriding the method's default routing. |
| `preset` | `"fast"`, `"standard"` (the default), `"thorough"`, or an `oevibe.Preset`. |
| `properties` | Properties to compute: `"charges"`, `"dipole"`, `"orbital_energies"`, `"bond_orders"`. |
| `charge` | Total charge override. |
| `multiplicity` | Spin multiplicity override. |
| `keywords` | Engine-specific keywords, for example `{"accuracy": 0.5}`. |

`optimize` adds `constraints`, `algorithm` and `max_iterations`; `frequencies` adds
`optimize`, `temperature` and `pressure`; `scan` adds `step`, `angle_range` and
`max_optimizations`. Strings and `Method` / `Solvent` / `Preset` objects are
interchangeable everywhere. The verbs are thin wrappers over `oevibe.CalculationOptions`
and `oevibe.Calculation`, which are the full API: build a `CalculationOptions` once and
reuse the `Calculation` when running many molecules through the same settings.

Units are fixed across the whole API: energies in kcal/mol, coordinates in Angstrom,
gradients in kcal/mol/Angstrom, Hessians in kcal/mol/Angstrom^2, frequencies in cm^-1
with imaginary modes reported as negative values, entropies in cal/(mol K), temperature
in K, pressure in atm, angles in degrees, partial charges in e, dipoles in Debye and
orbital energies in eV. Hartree and Bohr exist only inside the engine adapters.

## Engines and the Environment

| Engine | Methods | Provided by |
|--------|---------|-------------|
| `oeff` | `mmff94`, `mmff94s` (Sheffield water), `sage`, `parsley` (gas phase, partial charges required) | OpenEye, linked into the extension |
| `tblite` | `gfn2-xtb`, `gfn1-xtb`, with ALPB and GBSA solvation and all four properties | libtblite, loaded at runtime |
| `xtb` | `gfn0-xtb` | libxtb, loaded at runtime |

libtblite and libxtb are loaded with `dlopen` at first use and are never linked, which is
what keeps the wheel MIT: no LGPL binary is distributed with it. Each engine looks for its
library in this order, taking the first that loads:

1. `OEVIBE_TBLITE_LIBRARY` or `OEVIBE_XTB_LIBRARY`, naming the library file itself.
2. `$CONDA_PREFIX/lib`.
3. The running interpreter's library directory, which the Python package reports as
   `oevibe.get_python_library_directory()`.
4. The bare library name, left to the system loader (`DYLD_LIBRARY_PATH`,
   `LD_LIBRARY_PATH`, the default search paths).

A C++ caller that constructs `TbliteEngine` or `XtbEngine` with an explicit path uses that
path ahead of all four. Whichever candidate loads first is kept for the life of the
process; the libraries are never unloaded.

The documented environment:

```bash
micromamba create -n oevibe -c conda-forge python=3.12 tblite xtb=6.7.1 numpy
micromamba run -n oevibe uv pip install openeye-toolkits oevibe
```

GFN0-xTB additionally needs xtb's `param_gfn0-xtb.txt`, which OEVibe looks for next to the
library it loaded (`<prefix>/share/xtb`), so the environment above needs no configuration.
`XTBPATH` and `XTBHOME` take precedence when they are set, so an unrelated xtb installation
in your shell environment silently supplies the parameter file instead; the test commands
below unset both for that reason. Do not set `XTBPATH` to make OEVibe work -- the fallback
already finds the file, and setting it masks the fallback.

`gfn-ff` is declared but is not available in v1: xtb's C API 1.0 writes its `gfnff_topo`
topology file into the process working directory and reads it back from there, with no way
to redirect it, and OEVibe never changes the working directory.

`oevibe.available_engines()` reports every engine, its version, its methods, and the
reason for anything that is unavailable -- including the full list of library paths that
were tried:

```python
for info in oevibe.available_engines():
    print(info.name, info.version, info.available, info.reason)
    for method in info.methods:
        print("   ", method.name, method.available, method.reason)
```

## Writing an Engine in Python

A Python engine subclasses `oevibe.PythonEngine` and `oevibe.PythonEngineSession` and is
registered with `oevibe.register_engine`, or is shipped as an `oevibe.engines` entry point
resolving to a zero-argument callable that returns the engine. Once registered it behaves
like a built-in: every verb can route to it, and so can `oevibe.as_oemolfunc`.
`tests/python/mmff_engine.py` is a complete worked example, re-exposing OpenEye's MMFF94
through the Python toolkit.

```python
class MyEngine(oevibe.PythonEngine):
    def GetName(self):
        return "my-engine"

    def GetVersion(self):
        return "1.0"

    def GetMethods(self):
        capabilities = oevibe.MethodCapabilities()
        capabilities.method = oevibe.Method("my-method")
        return [capabilities]

    def CreateSession(self, system, settings):
        return MySession(system.molecule)


class MySession(oevibe.PythonEngineSession):
    def evaluate(self, coords, request):
        # coords is a read-only (N, 3) float64 array in Angstrom, dense atom order.
        # Return exactly what was asked for: an unrequested field is a malformed
        # evaluation and becomes ENGINE_ERROR (spec 8.1).
        energy, gradient = my_potential(coords)
        if request.gradient:
            return oevibe.Evaluation(energy=energy, gradient=gradient)
        return oevibe.Evaluation(energy=energy)


oevibe.register_engine(MyEngine())
```

Two rules a plugin author has to know:

- The package keeps a strong reference to every registered engine for the life of the
  process, so the C++ registry never holds a dangling director. An engine is never
  collected once registered.
- A session is created per conformer and must not be shared. It may cache anything it
  likes about the one molecule it was created for.

## Working with OpenEye's Own Interfaces

The adapters run in both directions.

`oevibe.MolFuncEngine(name, version, method, potential)` wraps one OpenEye potential as a
single-method OEVibe engine, so OEVibe's optimizer, scans and thermochemistry can drive it:

```python
from openeye import oeff

engine = oevibe.MolFuncEngine("oeff-molfunc", "1.0", oevibe.Method("mmff94s-oe"),
                              oeff.OEMMFF(True))
oevibe.register_engine(engine)
results = oevibe.optimize(mol, method="mmff94s-oe")
```

`oevibe.as_oemolfunc(options)` goes the other way, presenting an OEVibe calculation as an
`openeye.oeff.OEMolFunc1` so OpenEye's optimizers can drive OEVibe:

```python
function = oevibe.as_oemolfunc(options)
assert function.Setup(mol)
coords = oechem.OEDoubleArray(3 * mol.GetMaxAtomIdx())
mol.GetCoords(coords)
optimized = oechem.OEDoubleArray(function.NumVar())
energy = oeff.OEBFGSOpt()(function, coords, optimized)
assert function.GetEvaluationError() is None
```

C++ callers use `OEVIBE::EngineMolFunc` from `include/oevibe/OEAdapters.h` instead, which
derives from `OEMolPotential::OEMolFunc2` and works with `OEBFGSOpt`, the OEOpt adaptors
and `OEGenericFF2::Add`.

Four limitations, each of them measured:

- **Never hand an `as_oemolfunc` adapter to an OpenEye API that copies the potential: the
  process aborts, with no Python exception first.** `OEMolFunc1::CreateCopy` is pure
  virtual in C++ but absent from every one of `oeff`'s potential proxies, so OpenEye's
  director raises `Swig::DirectorPureVirtualException` instead of dispatching to the
  Python override, and nothing catches it. The known case is composition: the adapter can
  be added to an `oeff.OEGenericFF` and used through it, but copy-constructing that holder
  kills the interpreter, where the same copy succeeds for one of OpenEye's own potentials.
  No runtime guard is possible -- the abort happens in C++ before any Python frame exists.
  Driving an optimizer directly, as above, is the supported path; all seven of `oeff`'s
  first-derivative optimizers take the adapter to a minimum.
- **A failed evaluation inside `as_oemolfunc` does not raise.** An exception leaving a
  director becomes a `Swig::DirectorMethodException` that OpenEye's optimizers do not
  catch, which would abort the process, so a failure instead warns, reports an infinite
  energy and contributes nothing to the gradient. The line search then rejects the step and
  the last good geometry survives. A caller driving an OpenEye optimizer must check
  `GetEvaluationError()` when the run finishes: it returns the first exception an
  evaluation raised, or `None`.
- **`oeff.OEGenericFF` cannot be used as a `MolFuncEngine` prototype.** Its C++
  `CreateCopy()` drops the components, so the resulting engine reports "atom type not
  determined" and no energy. It fails loudly rather than silently, but compose your
  potentials some other way.
- **Python-defined `OEMolFunc1` subclasses are not supported** (write an
  `oevibe.PythonEngine` instead). `MolFuncEngine` rejects them with a `TypeError`. The
  guard blocks every route reachable without deliberately mutating module state; a caller
  who rewrites `openeye` module state before OEVibe first reads it, or reaches into SWIG
  internals, can defeat it. That is accepted: the guard exists to stop a silently wrong
  answer, not as a security boundary.

## Running the Tests

The tests that need libtblite or libxtb read two variables naming the library files, and
skip with a stated reason when either is unset or does not name a file. They decide on the
variables alone, so set both even inside the `oevibe` environment, where the libraries
are:

```bash
export OEVIBE_TBLITE_LIBRARY=$CONDA_PREFIX/lib/libtblite.0.dylib
export OEVIBE_XTB_LIBRARY=$CONDA_PREFIX/lib/libxtb.6.dylib
```

On Linux the filenames are `libtblite.so.0` and `libxtb.so.6`.

C++ tests (built by the debug preset):

```bash
cmake --build --preset debug
env -u XTBPATH -u XTBHOME ctest --test-dir build-debug --timeout 600
```

Python tests:

```bash
cmake --build --preset python
env -u XTBPATH -u XTBHOME PYTHONPATH=python python -m pytest tests/python
```

`XTBPATH` and `XTBHOME` are unset for both so that GFN0-xTB reads the parameter file of
the libxtb under test rather than one belonging to some other xtb installation.

Linting and type checking:

```bash
ruff format --check python tests scripts/acceptance
ruff check python tests scripts/acceptance
mypy python/oevibe scripts/acceptance
```

### The Acceptance Suite

The acceptance suite benchmarks OEVibe against reference outputs from the `xtb` 6.7.1
binary: optimizations at each preset, finite-difference frequencies, constrained
optimizations and GFN0-xTB single points, over 21 drug-like molecules plus two
constrained cases. It is marker-gated and the default selection does not run it; it needs
both engine libraries, but not the xtb binary, which it never runs:

```bash
env -u XTBPATH -u XTBHOME PYTHONPATH=python python -m pytest tests/python -m acceptance
```

The references are committed under `tests/data/acceptance/`, with
`tests/data/acceptance/manifest.json` recording the xtb version, the exact command lines,
the settings and the SHA-256 of every reference file. `tests/python/test_acceptance_manifest.py`
checks the manifest against the committed files, so a reference edited without
regenerating the manifest fails there; it carries the `acceptance` marker too, so it runs
with the command above rather than in the default selection. Regenerating the references
needs the xtb 6.7.1 binary and takes tens of minutes:

```bash
python scripts/acceptance/generate_references.py --xtb "$CONDA_PREFIX/bin/xtb"
```

Eight strict xfails are expected, and nothing else: the RMSD and energy criteria of the
four metformin cases. They miss spec 12.3's energy tolerance by 7 to 11 times (0.072 to
0.107 kcal/mol against 0.01) and its RMSD tolerance by 1.4 to 2.2 times (0.070 to
0.112 Angstrom against 0.05) on a molecule whose biguanide torsions are nearly flat. The
convergence difference against xtb is known, and leaving it uninvestigated is a deliberate
decision rather than an oversight. Every other case meets all of its criteria, and a
metformin case that starts passing fails the suite, which is the point of recording them
strictly.

Do not add a blanket `-W error::DeprecationWarning` to any of these commands: importing
`openeye` under it hangs the interpreter forever instead of failing, because the warning
becomes an exception inside an OpenEye extension's module initialization. Scope the filter
instead, as in `-W error::DeprecationWarning:oevibe`. The DeprecationWarnings the suite
reports come from the SWIG runtime inside the `openeye-toolkits` wheels, not from OEVibe.

### What CI Does Not Run

Spec 12.2 left open whether CI's OpenEye license access extends to a test job with
conda-forge tblite and xtb. OEVibe takes the spec's documented fallback: the tblite-,
xtb- and acceptance-dependent tests run locally only, and no such CI job is added in v1,
because the licensing question is unresolved and adding the job is outside this release's
scope. CI builds and publishes wheels; it does not run the engine tests.

## Project Structure

```
oevibe/
    CMakeLists.txt                  # Build configuration
    CMakePresets.json               # CMake presets (copy to CMakeUserPresets.json)
    pyproject.toml                  # Package metadata + [tool.oe-build] config
    vrzn.toml                       # Version locations for vrzn
    include/oevibe/                 # Installed public headers
        oevibe.h                    # Umbrella header, version macros
        Calculation.h  Options.h  Result.h  Method.h  Preset.h  Solvent.h
        Constraints.h  Engine.h  EngineRegistry.h  MolecularSystem.h
        OEAdapters.h                # EngineMolFunc, MolFuncEngine
        Error.h  Units.h  Property.h
        engines/                    # OEFFEngine.h, TbliteEngine.h, XtbEngine.h,
                                    #   RuntimeLibraries.h
    src/                            # Implementation of the public headers
        core/                       # Optimizer, Hessian, frequencies, scans, write-back
        engines/                    # oeff, tblite and xtb engines, dlopen function tables
    swig/
        oevibe.i                    # SWIG interface with OEMolBase typemaps and directors
        CMakeLists.txt              # SWIG module build rules
    python/
        pyproject.toml              # Setuptools config for editable installs
        oevibe/
            __init__.py             # Package exports and the OpenEye compat layer
            _verbs.py               # energy, gradient, optimize, frequencies, scan
            engines.py              # PythonEngine, entry points, available_engines
            oemolfunc.py            # MolFuncEngine, as_oemolfunc
    scripts/
        build_python.py             # Build distributable wheels
        acceptance/                 # Reference generator and the molecule set
    tests/
        cpp/                        # C++ unit tests (GoogleTest)
        python/                     # Python tests, including the acceptance modules
        data/acceptance/            # Committed xtb references and their manifest
    .github/workflows/
        build-wheels.yml            # CI: multi-platform wheel builds
```

## Platform Notes

The Python extension must link the shared OpenEye libraries that ship inside the
`openeye-toolkits` wheel. A static OpenEye would put a second OpenEye runtime in the
process, and each runtime keeps its own generic-data tag registry, so molecules crossing
between them come back with their SD data relabeled. The `python` preset arranges the
shared build and CMake fails the configure if it does not hold. The Windows wheel job
therefore needs an OpenEye SDK that ships shared libraries.

## Adding New C++ API

New public C++ API goes in a header under `include/oevibe/`; new `.cpp` files are added to
the `OEVIBE_SOURCES` list in `CMakeLists.txt`. To reach Python, the declaration is wrapped
in `swig/oevibe.i` and its header is added to `_OEVIBE_SWIG_DEPENDS` in
`swig/CMakeLists.txt` so that editing the header regenerates the bindings. Any function
taking `OEMolBase&` or `const OEMolBase&` automatically uses the cross-runtime typemaps.

## Building Wheels

### Local Build

The `scripts/build_python.py` script builds a distributable wheel. It reads
project-specific settings from the `[tool.oe-build]` section of `pyproject.toml`,
so the script itself never needs modification.

```bash
python scripts/build_python.py --openeye-root /path/to/openeye/sdk --verbose
```

The built wheel will be placed in `dist/`. On macOS, the script automatically
runs `delocate` to bundle non-OpenEye dependencies and sets the correct RPATH
for OpenEye shared libraries.

Options:

```
--openeye-root PATH    Path to OpenEye C++ SDK (or set OPENEYE_ROOT env var)
--python PATH          Python executable to use
--clean                Clean dist/ before building
--upload               Upload to PyPI after building
--test-upload          Upload to TestPyPI instead
--verbose              Show build commands
```

If `--openeye-root` is not provided, the script checks the `OPENEYE_ROOT`
environment variable and then `CMakePresets.json` / `CMakeUserPresets.json`.

### CI Builds

The included GitHub Actions workflow (`.github/workflows/build-wheels.yml`)
builds wheels on:

- Linux x86_64 (Rocky Linux 8, manylinux_2_28)
- Linux aarch64 (Rocky Linux 9, manylinux_2_34)
- macOS arm64 (universal2)
- Windows x64

It triggers on version tags (`v*`) and `workflow_dispatch`. Wheels are published
to PyPI via trusted publishing on tag pushes.

**Required GitHub Variables** (configured in Settings > Variables > Actions):

| Variable | Example | Description |
|----------|---------|-------------|
| `OPENEYE_VERSION` | `2026.1.0` | OpenEye SDK version for CI |
| `SDK_BUCKET` | `openeye-sdks` | Cloud storage bucket name |
| `SDK_LINUX_X86_64` | `OpenEye-toolkits-...-x64.tar.gz` | Linux x86_64 SDK filename |
| `SDK_LINUX_AARCH64` | `OpenEye-toolkits-...-aarch64.tar.gz` | Linux aarch64 SDK filename |
| `SDK_MACOS` | `OpenEye-toolkits-...-universal.tar.gz` | macOS SDK filename |

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | GCP Workload Identity Federation provider |
| `GCP_SERVICE_ACCOUNT` | GCP service account for SDK downloads |

The OpenEye C++ SDK and license file are downloaded from a GCS bucket
(`SDK_BUCKET`) during CI.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `OEVIBE_BUILD_TESTS` | ON | Build C++ tests |
| `OEVIBE_BUILD_PYTHON` | OFF | Build Python SWIG bindings (the `python` preset turns it on) |
| `OEVIBE_UNIVERSAL2` | OFF | Build macOS universal2 binary |
| `OEVIBE_USE_STABLE_ABI` | ON | Use Python stable ABI (abi3) |

## Tools

This project uses several tools to manage the build, packaging, and versioning:

| Tool | Purpose |
|------|---------|
| [CMake](https://cmake.org/) | Build system for the C++ library and SWIG bindings |
| [SWIG](https://www.swig.org/) | Generates Python bindings from C++ headers |
| [scikit-build-core](https://scikit-build-core.readthedocs.io/) | Python build backend that delegates to CMake |
| [cmake-openeye](https://github.com/scott-arne/cmake-openeye) | CMake modules for finding the OpenEye SDK and building SWIG targets |
| [vrzn](https://github.com/scott-arne/vrzn) | Keeps version numbers in sync across all project files |
| [delocate](https://github.com/matthew-brett/delocate) | Bundles shared libraries into macOS wheels |
| [auditwheel](https://github.com/pypa/auditwheel) | Bundles shared libraries into Linux wheels |
| [pytest](https://docs.pytest.org/) | Test framework for the Python test suite |

## Version Management

This project uses [vrzn](https://github.com/scott-arne/vrzn) to keep version
numbers consistent across all project files. The `vrzn.toml` configuration
tracks nine version locations:

| File | Location Type |
|------|---------------|
| `pyproject.toml` | `[project] version` |
| `python/pyproject.toml` | `[project] version` |
| `python/oevibe/__init__.py` | `__version__` |
| `python/oevibe/__init__.py` | `__version_info__` |
| `CMakeLists.txt` | `project(... VERSION ...)` |
| `include/oevibe/oevibe.h` | `#define` version macros |
| `swig/oevibe.i` | `#define` version macros |
| `tests/python/test_oevibe.py` | `__version__` assertion |
| `tests/python/test_oevibe.py` | `__version_info__` assertion |

Common commands:

```bash
vrzn get          # Show current version in all tracked locations
vrzn bump patch   # Bump patch version (e.g. 0.1.0 -> 0.1.1)
vrzn bump minor   # Bump minor version (e.g. 0.1.0 -> 0.2.0)
vrzn set 1.0.0    # Set an explicit version everywhere
```

Install vrzn with:

```bash
pip install vrzn
```

## License

MIT
