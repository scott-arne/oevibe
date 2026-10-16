"""The Windows DLL search-path hook that makes ``_oevibe.pyd`` resolvable.

Nothing here tests Windows DLL resolution, which needs a Windows host. What it
tests is the Python-level contract the resolution depends on: that the hook
registers the openeye-toolkits library directory, that it stays inert
everywhere else and on a static build, and that it runs before anything
imports the extension.
"""

import inspect
import os
import sys

import oevibe


def test_hook_is_inert_off_windows(monkeypatch):
    """On this platform the extension resolves OpenEye through RPATH, not a DLL directory."""
    registered = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda path: registered.append(path), raising=False
    )
    monkeypatch.setattr(oevibe, "_OPENEYE_DLL_DIRECTORIES", [])

    oevibe._add_openeye_dll_directory()

    assert sys.platform != "win32"
    assert registered == []


def test_registers_the_openeye_library_directory(monkeypatch, tmp_path):
    """A Windows shared build registers the directory holding the toolkit DLLs."""
    registered = []
    handles: list = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        os,
        "add_dll_directory",
        lambda path: (registered.append(path), object())[1],
        raising=False,
    )
    monkeypatch.setattr(oevibe, "_OPENEYE_DLL_DIRECTORIES", handles)
    monkeypatch.setattr(oevibe._build_info, "OPENEYE_LIBRARY_TYPE", "SHARED")
    monkeypatch.setattr(
        oevibe, "_find_openeye_runtime_lib_dir", lambda libs: str(tmp_path)
    )

    oevibe._add_openeye_dll_directory()

    assert registered == [str(tmp_path)]
    # The handle has to outlive the call; releasing it unregisters the directory.
    assert len(handles) == 1


def test_static_build_registers_nothing(monkeypatch, tmp_path):
    """A static build carries its own OpenEye copy and must not touch the search path."""
    registered = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        os, "add_dll_directory", lambda path: registered.append(path), raising=False
    )
    monkeypatch.setattr(oevibe, "_OPENEYE_DLL_DIRECTORIES", [])
    monkeypatch.setattr(oevibe._build_info, "OPENEYE_LIBRARY_TYPE", "STATIC")
    monkeypatch.setattr(
        oevibe, "_find_openeye_runtime_lib_dir", lambda libs: str(tmp_path)
    )

    oevibe._add_openeye_dll_directory()

    assert registered == []


def test_hook_runs_before_the_extension_is_imported():
    """Registration after the import would be too late: the loader resolves DLLs eagerly.

    Checked against the source because on this platform the hook is a no-op, so no runtime
    observation can distinguish a correctly ordered call from one that never mattered.
    """
    source = inspect.getsource(oevibe)
    hook = source.index("\n_add_openeye_dll_directory()")
    for importer in ("\n_load_cached_extension_if_needed()", "\nfrom .oevibe import"):
        assert hook < source.index(importer), (
            f"_add_openeye_dll_directory() must come before {importer.strip()}"
        )
