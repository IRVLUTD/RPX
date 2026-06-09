"""Smoke tests for every ``dataset_hub.cli`` command handler.

The handlers are thin wrappers around the library functions
(``pack_capture_tree``, ``build_frame_manifest``, etc.) but each one
still contains argparse-to-call wiring, return-code logic, and human-
readable print output. None of that was exercised by the library-level
tests, leaving ``dataset_hub/cli.py`` at 45% coverage (142 of 293
statements missing). These smoke tests invoke each handler through
the package's ``main([...])`` entry point against a tiny mock dataset
and assert the exit code, eliminating that coverage gap.

What is NOT tested here:

* ``upload`` and ``download`` — both hit HuggingFace. Skipped.
* ``stage-croissant`` — hits ``croissant_template.json`` package
  resource and is already covered by ``test_dataset_hub_croissant``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.cli import main as cli_main

pytest.importorskip("pyarrow", reason="pyarrow required for manifest builder")
pytest.importorskip("pandas", reason="pandas required for split-manifest writer")


# --------------------------------------------------------------------------- #
# Shared fixture: a packed-and-manifested mock dataset
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_root(tmp_path: Path) -> Path:
    """A tiny synthetic capture tree the CLI can pack/manifest/etc."""
    rc = cli_main(
        [
            "mock",
            "--out",
            str(tmp_path / "src"),
            "--multi",
            "2",
            "--single",
            "0",
            "--frames",
            "3",
            "--size",
            "16",
        ]
    )
    assert rc == 0
    return tmp_path / "src"


@pytest.fixture
def staged(mock_root: Path, tmp_path: Path) -> Path:
    """A staged dir with pack+manifest already run."""
    staging = tmp_path / "stage"
    assert (
        cli_main(["pack", "--src", str(mock_root), "--staging", str(staging), "--overwrite"]) == 0
    )

    splits_dir = tmp_path / "splits_in"
    splits_dir.mkdir()
    # Tiny scene_splits.json so the manifest step has rows to filter.
    scenes_json = splits_dir / "scene_splits.json"
    scenes_json.write_text(
        json.dumps(
            {
                "splits": {
                    "easy": ["scene1"],
                    "hard": ["scene2"],
                }
            }
        )
    )
    assert (
        cli_main(
            [
                "manifest",
                "--src",
                str(mock_root),
                "--staging",
                str(staging),
                "--splits",
                str(scenes_json),
            ]
        )
        == 0
    )
    return staging


# --------------------------------------------------------------------------- #
# `mock`
# --------------------------------------------------------------------------- #


def test_cli_mock_writes_a_capture_tree(tmp_path: Path):
    out = tmp_path / "src"
    rc = cli_main(
        [
            "mock",
            "--out",
            str(out),
            "--multi",
            "1",
            "--single",
            "0",
            "--frames",
            "2",
            "--size",
            "8",
        ]
    )
    assert rc == 0
    assert (out / "mos").is_dir()


# --------------------------------------------------------------------------- #
# `scan` — both table and --json output paths
# --------------------------------------------------------------------------- #


def test_cli_scan_table_mode(mock_root: Path, capsys):
    rc = cli_main(["scan", str(mock_root)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "scenes" in captured.out
    assert "per-modality totals" in captured.out


def test_cli_scan_json_mode(mock_root: Path, capsys):
    rc = cli_main(["scan", str(mock_root), "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    # Output is valid JSON
    payload = json.loads(captured.out)
    assert payload["totals"]["scenes"] >= 1


# --------------------------------------------------------------------------- #
# `pack`
# --------------------------------------------------------------------------- #


def test_cli_pack_creates_tar_shards(mock_root: Path, tmp_path: Path):
    staging = tmp_path / "stage"
    rc = cli_main(
        [
            "pack",
            "--src",
            str(mock_root),
            "--staging",
            str(staging),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert list(staging.rglob("*.tar"))


# --------------------------------------------------------------------------- #
# `manifest` — already exercised by `staged` fixture; assert the artefacts
# --------------------------------------------------------------------------- #


def test_cli_manifest_produces_parquet_and_current_json(staged: Path):
    assert (staged / "manifest" / "frames_v1.parquet").is_file()
    assert (staged / "manifest" / "current.json").is_file()
    assert (staged / "manifest" / "checksums.json").is_file()
    assert (staged / "manifest" / "file_checksums.json").is_file()


# --------------------------------------------------------------------------- #
# `stage-splits`
# --------------------------------------------------------------------------- #


def test_cli_stage_splits_copies_tier_files(staged: Path, tmp_path: Path):
    # Plant a source splits dir with the tier .txt files stage-splits expects.
    splits_src = tmp_path / "splits_src"
    splits_src.mkdir()
    (splits_src / "scene_splits.json").write_text(json.dumps({"splits": {}}))
    for tier in ("easy", "medium", "hard"):
        (splits_src / f"{tier}.txt").write_text("")

    rc = cli_main(
        [
            "stage-splits",
            "--staging",
            str(staged),
            "--splits-src",
            str(splits_src),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (staged / "splits" / "scene_splits.json").is_file()


# --------------------------------------------------------------------------- #
# `dataset-card`
# --------------------------------------------------------------------------- #


def test_cli_dataset_card_writes_readme(mock_root: Path, staged: Path):
    rc = cli_main(
        [
            "dataset-card",
            "--src",
            str(mock_root),
            "--staging",
            str(staged),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (staged / "README.md").is_file()


# --------------------------------------------------------------------------- #
# `verify`
# --------------------------------------------------------------------------- #


def test_cli_verify_succeeds_on_honest_tree(staged: Path, capsys):
    """Extract first so verify has an extracted/ tree to walk."""
    from rpx_benchmark.hub import _extract_snapshot_tars

    _extract_snapshot_tars(staged)
    rc = cli_main(["verify", str(staged)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "files OK" in captured.out


def test_cli_verify_returns_2_when_no_checksums_json(tmp_path: Path, capsys):
    """A snapshot dir with no checksums file → exit 2 (no defense available)."""
    snap = tmp_path / "legacy"
    (snap / "manifest").mkdir(parents=True)
    rc = cli_main(["verify", str(snap)])
    captured = capsys.readouterr()
    assert rc == 0  # the empty-report fallback returns 0
    assert "no file_checksums.json" in captured.err


def test_cli_verify_reports_corruption(staged: Path, capsys):
    """A byte flipped in an extracted file → exit 1, mismatch in output."""
    from rpx_benchmark.hub import _extract_snapshot_tars

    _extract_snapshot_tars(staged)
    # Tamper one extracted file
    target = next(iter((staged / "extracted").rglob("rgb/*")), None)
    if target is None:
        pytest.skip("no extracted RGB to tamper with")
    with target.open("r+b") as f:
        f.seek(64)
        f.write(b"\xff")
    rc = cli_main(["verify", str(staged)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "failures" in captured.out


# --------------------------------------------------------------------------- #
# `lossless-convert`
# --------------------------------------------------------------------------- #


def test_cli_lossless_convert_dry_run(mock_root: Path, tmp_path: Path, capsys):
    out = tmp_path / "v2"
    rc = cli_main(
        [
            "lossless-convert",
            "--src",
            str(mock_root),
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "DRY-RUN" in captured.out


def test_cli_lossless_convert_full_run(mock_root: Path, tmp_path: Path, capsys):
    out = tmp_path / "v2"
    rc = cli_main(
        [
            "lossless-convert",
            "--src",
            str(mock_root),
            "--out",
            str(out),
            "--workers",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "TOTAL" in captured.out
    # rgb/fisheye files should now be .webp; depth stays .png
    assert list(out.rglob("rgb/*.webp"))
    assert list(out.rglob("depth/*.png"))


def test_cli_lossless_convert_skip_flags_compose(mock_root: Path, tmp_path: Path):
    """All three --skip-* flags together should produce a pure-link copy."""
    out = tmp_path / "v2"
    rc = cli_main(
        [
            "lossless-convert",
            "--src",
            str(mock_root),
            "--out",
            str(out),
            "--workers",
            "1",
            "--skip-rgb-webp",
            "--skip-png-recompress",
            "--skip-cam-pose",
        ]
    )
    assert rc == 0
    # No conversions — rgb stays PNG.
    assert list(out.rglob("rgb/*.png"))
    assert not list(out.rglob("rgb/*.webp"))


# --------------------------------------------------------------------------- #
# `stage-croissant`
# --------------------------------------------------------------------------- #


def test_cli_stage_croissant_writes_json(staged: Path):
    rc = cli_main(
        [
            "stage-croissant",
            "--staging",
            str(staged),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (staged / "rpx_croissant.json").is_file()


# --------------------------------------------------------------------------- #
# Help and bad-command paths
# --------------------------------------------------------------------------- #


def test_cli_no_subcommand_exits_nonzero(capsys):
    with pytest.raises(SystemExit):
        cli_main([])
