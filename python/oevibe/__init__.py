"""
OEVibe - A C++/Python library for energy calculations with OpenEye molecules
"""

import hashlib
import importlib.machinery
import importlib.util
import os
import re
import shutil
import sys
import sysconfig
import warnings
from importlib import metadata
from pathlib import Path

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)


_OPENEYE_COMPAT_PRELOAD_PATHS: list[str] = []
_OPENEYE_COMPAT_EXTENSION_DIR: Path | None = None


def _user_cache_root():
    """Return the per-user cache root for OpenEye compatibility aliases."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "oevibe"
    return Path.home() / ".cache" / "oevibe"


def _runtime_openeye_version():
    """Return the installed OpenEye toolkit distribution version if available."""
    try:
        return metadata.version("openeye-toolkits")
    except metadata.PackageNotFoundError:
        return "unknown"


def _cache_key(oe_lib_dir, expected_libs, build_version, runtime_version):
    """Build a stable cache key for one OpenEye runtime library set."""
    key_data = "\n".join(
        [
            os.path.realpath(oe_lib_dir),
            build_version or "unknown",
            runtime_version or "unknown",
            *sorted(expected_libs),
        ]
    )
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:16]


def _runtime_shared_library_names(lib_names):
    """Return filenames that can participate in runtime dynamic loading."""
    return [
        lib_name
        for lib_name in lib_names
        if ".so" in lib_name or lib_name.endswith(".dylib") or lib_name.endswith(".dll")
    ]


def _is_openeye_runtime_library_name(lib_name):
    """Return whether a dependency belongs to the OpenEye runtime set."""
    return lib_name.startswith("liboe") or lib_name.startswith("libzstd.")


def _find_openeye_runtime_lib_dir(expected_libs=()):
    """Find the OpenEye runtime library directory without importing oechem."""
    search_locations = []
    openeye_module = sys.modules.get("openeye")
    openeye_path = getattr(openeye_module, "__path__", None)
    if openeye_path is not None:
        search_locations.extend(openeye_path)

    if not search_locations:
        try:
            openeye_spec = importlib.util.find_spec("openeye")
        except (ImportError, ValueError):
            openeye_spec = None
        if (
            openeye_spec is not None
            and openeye_spec.submodule_search_locations is not None
        ):
            search_locations.extend(openeye_spec.submodule_search_locations)

    expected_libs = set(_runtime_shared_library_names(expected_libs or ()))
    fallback_dir = None
    for package_root in search_locations:
        libs_root = Path(package_root) / "libs"
        if not libs_root.is_dir():
            continue

        # Importing openeye.libs eagerly imports oechem in some environments.
        # The runtime libraries are shipped below openeye/libs, so filesystem
        # discovery preserves the fresh-import condition.
        for root, _, files in os.walk(libs_root):
            file_set = set(files)
            if expected_libs and expected_libs.intersection(file_set):
                return root
            if fallback_dir is None and any(
                ".dylib" in lib_name or ".so" in lib_name or ".dll" in lib_name
                for lib_name in files
            ):
                fallback_dir = root

    return fallback_dir


def _library_family(lib_name):
    """Return the stable library family name for a versioned shared library."""
    match = re.match(r"(lib\w+?)(-[\d.]+)?(\.[\d.]*\w+)$", lib_name)
    if match is None:
        return None
    return match.group(1)


def _candidate_runtime_libraries(oe_lib_dir, expected_name):
    """Find runtime libraries with the same family as an expected filename."""
    family = _library_family(expected_name)
    if family is None:
        return []
    candidates = []
    for file_name in os.listdir(oe_lib_dir):
        candidate_path = os.path.join(oe_lib_dir, file_name)
        if not os.path.isfile(candidate_path):
            continue
        if file_name.startswith(f"{family}-") or file_name.startswith(f"{family}."):
            candidates.append(candidate_path)
    return sorted(candidates)


def _compatible_library_path(oe_lib_dir, expected_name):
    """Return a runtime library path and whether it needs an expected-name alias."""
    exact_path = os.path.join(oe_lib_dir, expected_name)
    if os.path.isfile(exact_path):
        return exact_path, False

    candidates = _candidate_runtime_libraries(oe_lib_dir, expected_name)
    if len(candidates) != 1:
        candidate_names = ", ".join(os.path.basename(path) for path in candidates)
        raise ImportError(
            f"Could not find a compatible OpenEye runtime library for "
            f"{expected_name!r} in {oe_lib_dir!r}. "
            f"Candidates: {candidate_names or 'none'}."
        )
    return candidates[0], True


def _extension_runtime_library_names(pkg_dir):
    """Return OpenEye runtime library names recorded by the extension."""
    extension_path = _find_extension_module_path(pkg_dir)
    if extension_path is None:
        return []

    if sys.platform == "darwin":
        return _mach_o_runtime_library_names(extension_path)
    if sys.platform.startswith("linux"):
        return _elf_runtime_library_names(extension_path)
    return []


def _mach_o_runtime_library_names(extension_path):
    """Return OpenEye dylib dependencies recorded in a Mach-O extension."""
    import subprocess

    try:
        result = subprocess.run(
            ["otool", "-L", str(extension_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return []

    dependencies = []
    for line in result.stdout.splitlines()[1:]:
        dependency = line.strip().split(" ", 1)[0]
        lib_name = os.path.basename(dependency)
        if _is_openeye_runtime_library_name(lib_name):
            dependencies.append(lib_name)
    return dependencies


def _elf_runtime_library_names(extension_path):
    """Return OpenEye shared-library dependencies recorded in an ELF extension."""
    import subprocess

    try:
        result = subprocess.run(
            ["readelf", "-d", str(extension_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return []

    dependencies = []
    for match in re.finditer(r"Shared library: \[(?P<name>[^\]]+)\]", result.stdout):
        lib_name = match.group("name")
        if _is_openeye_runtime_library_name(lib_name):
            dependencies.append(lib_name)
    return dependencies


def _ensure_cache_alias(cache_dir, expected_name, target_path):
    """Create or refresh an expected-name symlink in the user cache."""
    alias_path = cache_dir / expected_name
    if alias_path.is_symlink():
        if alias_path.resolve() == Path(target_path).resolve():
            return alias_path
        alias_path.unlink()
    elif alias_path.exists():
        raise ImportError(
            f"Cannot create OpenEye compatibility alias {alias_path}: "
            "a non-symlink file already exists at that path."
        )

    try:
        alias_path.symlink_to(target_path)
    except OSError as exc:
        raise ImportError(
            f"Could not create OpenEye compatibility alias "
            f"{alias_path} -> {target_path}: {exc}"
        ) from exc
    return alias_path


def _mirror_openeye_runtime_libs(oe_lib_dir, cache_dir):
    """Mirror the full OpenEye runtime library set into the cache directory.

    Relocating the extension so its ``$ORIGIN`` can resolve name-drift aliases
    also drops the build-time RPATH entry that points at the OpenEye library
    directory. Transitive dependencies that are not recorded as direct
    dependencies of the extension (for example ``liboecuda``) would then be
    unresolvable from the cache. Mirroring every shared library from that
    directory reproduces the directory the dropped RPATH entry pointed at,
    restoring a complete, ``$ORIGIN``-local view of the runtime set.

    Symlinks are used rather than copies, so no library is loaded here and the
    global symbol namespace is left untouched; ``ld.so`` only resolves a
    mirrored library if the extension's dependency chain actually needs it.

    :param oe_lib_dir: The OpenEye runtime library directory to mirror.
    :param cache_dir: The user cache directory the extension loads from.
    """
    for file_name in sorted(os.listdir(oe_lib_dir)):
        if not _runtime_shared_library_names([file_name]):
            continue
        source_path = os.path.join(oe_lib_dir, file_name)
        if not os.path.isfile(source_path):
            continue
        _ensure_cache_alias(cache_dir, file_name, source_path)


def _ensure_library_compat():
    """Prepare compatibility aliases when OpenEye library filenames drift.

    When oevibe is built with shared OpenEye libraries, the compiled extension
    records the exact versioned library filenames (e.g., liboechem-4.3.0.1.dylib).
    If the user upgrades openeye-toolkits, these filenames change and the dynamic
    linker fails to load the extension.

    This function creates expected-name aliases in a user-writable cache instead
    of mutating the installed package directory. When aliases are needed, the
    extension is later loaded from the same cache directory so its $ORIGIN lookup
    can find those aliases.
    """
    global _OPENEYE_COMPAT_EXTENSION_DIR, _OPENEYE_COMPAT_PRELOAD_PATHS

    _OPENEYE_COMPAT_PRELOAD_PATHS = []
    _OPENEYE_COMPAT_EXTENSION_DIR = None

    try:
        from . import _build_info
    except ImportError:
        return False

    if getattr(_build_info, "OPENEYE_LIBRARY_TYPE", "STATIC") != "SHARED":
        return False

    expected_libs = set(
        _runtime_shared_library_names(getattr(_build_info, "OPENEYE_EXPECTED_LIBS", []))
    )
    expected_libs.update(_extension_runtime_library_names(os.path.dirname(__file__)))
    expected_libs = sorted(expected_libs)
    if not expected_libs:
        return False

    oe_lib_dir = _find_openeye_runtime_lib_dir(expected_libs)
    if oe_lib_dir is None:
        return False

    if not os.path.isdir(oe_lib_dir):
        return False

    build_version = getattr(_build_info, "OPENEYE_BUILD_VERSION", None)
    runtime_version = _runtime_openeye_version()
    cache_dir = (
        _user_cache_root()
        / "openeye-libs"
        / _cache_key(oe_lib_dir, expected_libs, build_version, runtime_version)
    )

    preload_paths = []
    needs_cached_origin = False
    for expected_name in expected_libs:
        actual_path, needs_alias = _compatible_library_path(oe_lib_dir, expected_name)
        if needs_alias:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ImportError(
                    f"Could not create OpenEye compatibility cache directory "
                    f"{cache_dir}: {exc}"
                ) from exc
            alias_path = _ensure_cache_alias(cache_dir, expected_name, actual_path)
            preload_paths.append(str(alias_path))
            needs_cached_origin = True
        else:
            preload_paths.append(actual_path)

    _OPENEYE_COMPAT_PRELOAD_PATHS = preload_paths
    if needs_cached_origin:
        # The relocated extension loses its build-time RPATH entry into the
        # OpenEye library directory, so mirror the full runtime set into the
        # cache to keep transitive dependencies (e.g. liboecuda) resolvable.
        _mirror_openeye_runtime_libs(oe_lib_dir, cache_dir)
        _OPENEYE_COMPAT_EXTENSION_DIR = cache_dir

    return needs_cached_origin


def _extension_suffixes():
    """Return extension-module suffixes for the active Python interpreter."""
    return tuple(importlib.machinery.EXTENSION_SUFFIXES)


def _find_extension_module_path(pkg_dir):
    """Find the installed _oevibe extension file."""
    for suffix in _extension_suffixes():
        candidate = Path(pkg_dir) / f"_oevibe{suffix}"
        if candidate.is_file():
            return candidate
    for candidate in Path(pkg_dir).glob("_oevibe*"):
        if candidate.is_file() and str(candidate).endswith(_extension_suffixes()):
            return candidate
    return None


def _copy_if_stale(source_path, target_path):
    """Copy a file into the cache when size or mtime changed."""
    if (
        target_path.exists()
        and target_path.stat().st_size == source_path.stat().st_size
        and target_path.stat().st_mtime_ns == source_path.stat().st_mtime_ns
    ):
        return
    shutil.copy2(source_path, target_path)


def _copy_package_shared_sidecars(pkg_dir, cache_dir, extension_path):
    """Copy package-local shared library sidecars needed by cached extension."""
    for candidate in Path(pkg_dir).iterdir():
        name = candidate.name
        if not candidate.is_file() or candidate == extension_path:
            continue
        if (
            ".so" not in name
            and not name.endswith(".dylib")
            and not name.endswith(".dll")
            and not name.endswith(".pyd")
        ):
            continue
        _copy_if_stale(candidate, cache_dir / name)


def _load_cached_extension_if_needed():
    """Load _oevibe from the cache when OpenEye aliases live there."""
    cache_dir = _OPENEYE_COMPAT_EXTENSION_DIR
    if cache_dir is None:
        return

    module_name = f"{__name__}._oevibe"
    if module_name in sys.modules:
        return

    pkg_dir = os.path.dirname(__file__)
    extension_path = _find_extension_module_path(pkg_dir)
    if extension_path is None:
        return

    cached_extension_path = cache_dir / extension_path.name
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _copy_if_stale(extension_path, cached_extension_path)
        _copy_package_shared_sidecars(pkg_dir, cache_dir, extension_path)
    except OSError as exc:
        raise ImportError(
            f"Could not prepare cached oevibe extension in {cache_dir}: {exc}"
        ) from exc

    spec = importlib.util.spec_from_file_location(module_name, cached_extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {cached_extension_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _preload_shared_libs():
    """Preload OpenEye shared libraries so the C extension can find them.

    On Linux, the extension's RUNPATH (set at build time) normally handles
    dependency resolution, but preloading ensures libraries are available
    even if RUNPATH is stripped (e.g. by certain packaging tools).
    On macOS, @rpath references may not resolve without preloading.

    Only the libraries recorded in ``OPENEYE_EXPECTED_LIBS`` are loaded,
    and they are loaded with ``RTLD_GLOBAL`` so that cross-module C++
    symbol references resolve correctly. Loading the entire OpenEye
    library directory (which can contain 70+ unrelated shared objects)
    would pollute the global symbol namespace and cause segfaults in
    unrelated C extensions such as ``_sqlite3``.
    """
    import ctypes
    import sys

    if sys.platform not in ("linux", "darwin"):
        return

    try:
        from . import _build_info
    except ImportError:
        return

    if getattr(_build_info, "OPENEYE_LIBRARY_TYPE", "STATIC") != "SHARED":
        return

    expected_libs = _runtime_shared_library_names(
        getattr(_build_info, "OPENEYE_EXPECTED_LIBS", [])
    )
    if not expected_libs:
        return

    oe_lib_dir = _find_openeye_runtime_lib_dir(expected_libs)
    if oe_lib_dir is None:
        return

    if not os.path.isdir(oe_lib_dir):
        return

    paths = _OPENEYE_COMPAT_PRELOAD_PATHS
    if not paths:
        paths = [
            os.path.join(oe_lib_dir, lib_name)
            for lib_name in expected_libs
            if os.path.exists(os.path.join(oe_lib_dir, lib_name))
        ]

    for path in paths:
        if os.path.exists(path) or os.path.islink(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _preload_bundled_libs():
    """Preload libraries bundled by auditwheel from the .libs directory.

    auditwheel repair bundles non-manylinux dependencies (e.g. libraries
    from FetchContent or system packages) into a ``<package>.libs/``
    directory next to the package. The bundled copies have hashed filenames
    and must be loaded before the C extension to satisfy its DT_NEEDED
    entries.

    Libraries may have inter-dependencies, so we do multiple passes
    until no new libraries can be loaded. Libraries are loaded without
    ``RTLD_GLOBAL`` to avoid polluting the global symbol namespace.
    """
    import sys

    if sys.platform != "linux":
        return

    import ctypes

    pkg_name = __name__
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    site_dir = os.path.dirname(pkg_dir)
    for libs_name in (f"{pkg_name}.libs", f".{pkg_name}.libs"):
        libs_dir = os.path.join(site_dir, libs_name)
        if not os.path.isdir(libs_dir):
            continue
        remaining = [
            os.path.join(libs_dir, f)
            for f in sorted(os.listdir(libs_dir))
            if ".so" in f
        ]
        while remaining:
            failed = []
            for lib_path in remaining:
                try:
                    ctypes.CDLL(lib_path)
                except OSError:
                    failed.append(lib_path)
            if len(failed) == len(remaining):
                break
            remaining = failed


def _check_openeye_version():
    """Check that the OpenEye version matches what was used at build time."""
    try:
        from . import _build_info
    except ImportError:
        return

    if getattr(_build_info, "OPENEYE_LIBRARY_TYPE", "STATIC") != "SHARED":
        return

    build_version = getattr(_build_info, "OPENEYE_BUILD_VERSION", None)
    if not build_version:
        return

    try:
        from importlib import metadata

        runtime_version = metadata.version("openeye-toolkits")
    except metadata.PackageNotFoundError:
        warnings.warn(
            "openeye-toolkits package not found. "
            "Install with: pip install openeye-toolkits",
            ImportWarning,
        )
        return

    build_parts = build_version.split(".")[:2]
    runtime_parts = runtime_version.split(".")[:2]
    if build_parts != runtime_parts:
        warnings.warn(
            f"OpenEye version mismatch: oevibe was built with "
            f"OpenEye Toolkits {build_version} but runtime has {runtime_version}. "
            f"This may cause compatibility issues.",
            RuntimeWarning,
        )


_ensure_library_compat()
_preload_shared_libs()
_preload_bundled_libs()
_load_cached_extension_if_needed()
_check_openeye_version()

from ._verbs import energy, frequencies, gradient, optimize, scan
from .oevibe import (
    Algorithm,
    Calculation,
    CalculationOptions,
    ConformerResult,
    ConformerStatus,
    ConstraintSet,
    Engine,
    EngineRegistry,
    EngineSession,
    EngineSettings,
    EngineUnavailableError,
    Evaluation,
    EvaluationRequest,
    InputError,
    KeywordSpec,
    KeywordType,
    Method,
    MethodCapabilities,
    MolecularSystem,
    OEVibeError,
    OptimizationStats,
    OptimizerOptions,
    OptionsError,
    PartialChargePolicy,
    Preset,
    Property,
    Provenance,
    ResultSet,
    ScanOptions,
    ScanPoint,
    ScanPointStatus,
    ScanResult,
    Solvent,
    SolventModel,
    Thermochemistry,
    ThermoOptions,
    Timings,
    UnsupportedError,
    find_method,
    get_python_library_directory,
    set_python_library_directory,
    validate_evaluation,
)


def _python_library_directory() -> str:
    """Return the interpreter's library directory, where conda installs libtblite and libxtb.

    ``LIBDIR`` is the ``lib`` directory of the interpreter's own installation, which in a conda
    or micromamba environment is ``<prefix>/lib``. An interpreter that does not record it falls
    back to ``<sys.prefix>/lib``.

    :returns: The directory the runtime-loaded engines search after ``$CONDA_PREFIX/lib``.
    """
    libdir = sysconfig.get_config_var("LIBDIR")
    if isinstance(libdir, str) and libdir:
        return libdir
    return os.path.join(sys.prefix, "lib")


# The engines search for libtblite and libxtb only when a calculation first needs one, and
# nothing above reaches an engine, so the directory is in place before any search (spec 8.3).
set_python_library_directory(_python_library_directory())

__all__ = [
    "Algorithm",
    "Calculation",
    "CalculationOptions",
    "ConformerResult",
    "ConformerStatus",
    "ConstraintSet",
    "Engine",
    "EngineRegistry",
    "EngineSession",
    "EngineSettings",
    "EngineUnavailableError",
    "Evaluation",
    "EvaluationRequest",
    "InputError",
    "KeywordSpec",
    "KeywordType",
    "Method",
    "MethodCapabilities",
    "MolecularSystem",
    "OEVibeError",
    "OptimizationStats",
    "OptimizerOptions",
    "OptionsError",
    "PartialChargePolicy",
    "Preset",
    "Property",
    "Provenance",
    "ResultSet",
    "ScanOptions",
    "ScanPoint",
    "ScanPointStatus",
    "ScanResult",
    "Solvent",
    "SolventModel",
    "ThermoOptions",
    "Thermochemistry",
    "Timings",
    "UnsupportedError",
    "__version__",
    "__version_info__",
    "energy",
    "find_method",
    "frequencies",
    "get_python_library_directory",
    "gradient",
    "optimize",
    "scan",
    "set_python_library_directory",
    "validate_evaluation",
]
