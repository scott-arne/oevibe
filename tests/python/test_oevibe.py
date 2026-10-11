"""Package-level tests for oevibe."""

import oevibe


def test_version_available():
    assert hasattr(oevibe, "__version__")
    assert hasattr(oevibe, "__version_info__")
    assert oevibe.__version__ == "0.1.0"
    assert oevibe.__version_info__ == (0, 1, 0)


def test_exception_hierarchy():
    assert issubclass(oevibe.OEVibeError, RuntimeError)
    for name in (
        "OptionsError",
        "EngineUnavailableError",
        "InputError",
        "UnsupportedError",
    ):
        assert issubclass(getattr(oevibe, name), oevibe.OEVibeError)
