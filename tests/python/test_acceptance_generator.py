"""An interrupted promotion must not cost the committed acceptance references.

``scripts/acceptance/generate_references.py`` publishes ``tests/data/acceptance`` with two
renames. Between them the committed tree exists only under the ``.replaced`` name, and that
window is the one state in which a backup is the sole copy of the reference set Tasks 12 and 13
consume. These tests drive the two properties that make the window survivable: a promotion that
fails or is interrupted puts the committed tree back, and a later run refuses to touch a tree
left in that state rather than deleting the backup and publishing only what it just generated.

Nothing here runs xtb or touches the real ``tests/data/acceptance``; every test points the
generator's ``DATA_DIR`` at a tmp_path.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "acceptance")
)

import generate_references

# Two distinguishable trees: whichever one is on disk at the end says which rename took effect.
COMMITTED = '{"cases": "the complete committed set"}\n'
FRESH = '{"cases": "one molecule from an --only rerun"}\n'


@pytest.fixture
def committed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A stand-in for the committed reference tree, with the generator pointed at it."""
    data_dir = tmp_path / "acceptance"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(COMMITTED)
    monkeypatch.setattr(generate_references, "DATA_DIR", data_dir)
    return data_dir


def staging_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """:returns: A staging tree holding ``FRESH``, laid out as the generator lays one out."""
    staging = tmp_path / ".staging-test"
    staging.mkdir()
    (staging / "manifest.json").write_text(FRESH)
    return staging


def backup_of(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir.with_name(data_dir.name + ".replaced")


def rename_failing_on(source: pathlib.Path, error: BaseException):
    """:returns: A ``Path.rename`` that raises ``error`` when ``source`` is the one renamed.

    The interruption has to land on the second of the two renames specifically -- the one that
    moves staging into place -- because that is the instant at which the committed tree exists
    only under the ``.replaced`` name. The rollback renames a different directory, so it still
    runs.
    """
    original = pathlib.Path.rename

    def rename(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if self == source:
            raise error
        return original(self, target)

    return rename


# A SIGINT during promotion surfaces as KeyboardInterrupt, which is not an Exception, so the
# rollback has to catch it too -- a Ctrl-C is the likeliest way this window is ever hit.
@pytest.mark.parametrize(
    "interruption",
    [OSError("rename failed"), KeyboardInterrupt()],
    ids=["oserror", "sigint"],
)
def test_a_failed_promotion_puts_the_committed_references_back(
    committed: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    staging = staging_tree(tmp_path)
    monkeypatch.setattr(
        pathlib.Path, "rename", rename_failing_on(staging, interruption)
    )

    with pytest.raises(type(interruption)):
        generate_references.promote(staging)

    assert (committed / "manifest.json").read_text() == COMMITTED
    assert not backup_of(committed).exists()


def test_a_promotion_refuses_when_the_backup_is_the_only_copy(
    committed: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    replaced = backup_of(committed)
    committed.rename(replaced)  # exactly the state a killed promotion leaves behind
    staging = staging_tree(tmp_path)

    with pytest.raises(SystemExit) as refusal:
        generate_references.promote(staging)

    assert (replaced / "manifest.json").read_text() == COMMITTED
    assert not committed.exists()
    assert str(replaced) in str(refusal.value)


def test_a_run_refuses_before_generating_when_the_backup_is_the_only_copy(
    committed: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal has to precede generation, or a 26-second run is spent to reach it."""
    replaced = backup_of(committed)
    committed.rename(replaced)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_references.py",
            "--xtb",
            str(tmp_path / "no-such-xtb"),
            "--only",
            "urea",
        ],
    )

    with pytest.raises(SystemExit) as refusal:
        generate_references.main()

    # Not the "holds no xtb parameter files" exit: the state is checked before the binary is.
    assert str(replaced) in str(refusal.value)
    assert (replaced / "manifest.json").read_text() == COMMITTED
    assert list(tmp_path.glob(".staging-*")) == []


def test_a_normal_promotion_replaces_the_committed_references(
    committed: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    staging = staging_tree(tmp_path)

    generate_references.promote(staging)

    assert (committed / "manifest.json").read_text() == FRESH
    assert not staging.exists()
    assert not backup_of(committed).exists()


def test_a_normal_promotion_clears_a_leftover_backup(
    committed: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A ``.replaced`` beside an intact committed tree is stale, and is still cleaned up."""
    replaced = backup_of(committed)
    replaced.mkdir()
    (replaced / "manifest.json").write_text("stale\n")

    generate_references.promote(staging_tree(tmp_path))

    assert (committed / "manifest.json").read_text() == FRESH
    assert not replaced.exists()


def test_a_first_promotion_with_nothing_committed_publishes_the_staging_tree(
    committed: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """No committed tree and no backup is a first generation, not a crashed one."""
    (committed / "manifest.json").unlink()
    committed.rmdir()

    generate_references.promote(staging_tree(tmp_path))

    assert (committed / "manifest.json").read_text() == FRESH
