"""CI smoke test for the build_esd_splits.py CLI driver.

The CLI lives outside the package (under ``experiments/scripts/``) so it
isn't covered by normal package tests. This file imports it via sys.path
injection and exercises the happy path + a failure path against a tiny
synthetic dataset built in ``tmp_path``.

The intent is regression-detection, not exhaustive scenario coverage —
per-feature math is tested in ``test_esd.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make the script importable. Repo layout: <repo>/benchmark/tests/<this file>
# and <repo>/experiments/scripts/build_esd_splits.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "experiments" / "scripts"
if not (_SCRIPTS_DIR / "build_esd_splits.py").is_file():
    pytest.skip(f"CLI script not at expected path: {_SCRIPTS_DIR}", allow_module_level=True)
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_esd_splits  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixture helpers — write the smallest dataset that exercises every loader
# --------------------------------------------------------------------------- #


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def _build_one_phase(phase_dir: Path, n_frames: int = 3, h: int = 8, w: int = 8) -> None:
    for sub in ("rgb", "depth", "cam_pose", "sam2/masks"):
        (phase_dir / sub).mkdir(parents=True, exist_ok=True)
    for t in range(n_frames):
        stem = f"{t:05d}"
        Image.fromarray(np.full((h, w, 3), 128, np.uint8)).save(phase_dir / "rgb" / f"{stem}.png")
        Image.fromarray(np.full((h, w), 1500, np.uint16)).save(phase_dir / "depth" / f"{stem}.png")
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[0:2, 0:2] = 1
        Image.fromarray(mask).save(phase_dir / "sam2/masks" / f"{stem}.png")
        np.savez(
            phase_dir / "cam_pose" / f"{stem}.npz",
            position=np.array([0.01 * t, 0.0, 0.0]),
            orientation=_identity_quat(),
        )


def _build_dataset(root: Path, *, n_scenes: int = 2, n_phases: int = 3) -> Path:
    for s in range(n_scenes):
        for p in range(n_phases):
            _build_one_phase(root / f"scene{s + 1}" / str(p))
    return root


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_cli_clean_run_writes_json_csv_and_log(tmp_path):
    data_root = _build_dataset(tmp_path / "data")
    out_dir = tmp_path / "out"
    json_path = out_dir / "phase_esd_splits.json"
    log_path = out_dir / "phase_esd_splits.log"

    rc = build_esd_splits.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(json_path),
            "--workers",
            "1",
        ]
    )
    assert rc == 0

    # JSON: schema fields present, all 6 (scene, phase) rows extracted.
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] >= 2
    assert len(payload["feature_names"]) == 31
    assert payload["summary"]["ok"] is True
    assert payload["summary"]["n_entries"] == 6
    assert payload["summary"]["n_failed"] == 0
    assert len(payload["phases"]) == 6
    # Each row has all 27 features populated.
    sample_row = next(iter(payload["phases"].values()))
    assert set(sample_row["features"].keys()) == set(payload["feature_names"])
    # feature_stats block carries one entry per feature.
    assert len(payload["feature_stats"]) == 31

    # CSV: schema-version comment header + DictWriter header + 6 rows.
    csv_path = json_path.with_suffix(".csv")
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("# schema_version=")
    assert lines[1].startswith("scene_id,phase,n_frames_total,n_frames_used,")
    assert len(lines) == 2 + 6  # comment + header + 6 data rows

    # Per-feature stats CSV alongside.
    stats_csv = out_dir / "phase_esd_splits_feature_stats.csv"
    assert stats_csv.is_file()
    assert len(stats_csv.read_text().splitlines()) == 1 + 31  # header + features

    # Log file mirror exists and is non-empty.
    assert log_path.is_file()
    assert log_path.stat().st_size > 0


def test_cli_returns_2_on_missing_data_root(tmp_path):
    rc = build_esd_splits.main(
        [
            "--data-root",
            str(tmp_path / "does_not_exist"),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2


def test_cli_returns_1_on_per_phase_failure(tmp_path):
    data_root = _build_dataset(tmp_path / "data", n_scenes=1, n_phases=2)
    # Corrupt one mask so its phase fails extraction. The other phase still works.
    bad_mask = data_root / "scene1" / "1" / "sam2/masks" / "00000.png"
    bad_mask.write_bytes(b"not a png")

    out_json = tmp_path / "out" / "esd.json"
    rc = build_esd_splits.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(out_json),
            "--workers",
            "1",
        ]
    )
    assert rc == 1  # partial-failure exit code

    payload = json.loads(out_json.read_text())
    assert payload["summary"]["ok"] is False
    assert payload["summary"]["n_failed"] == 1
    assert payload["summary"]["n_entries"] == 1
    failure = payload["summary"]["failures"][0]
    # The CLI catches whatever the per-phase worker raised — could be
    # PIL's `UnidentifiedImageError`, our own `DatasetError`, or another
    # variant depending on which step ate the corrupt PNG. Assert the
    # weaker contract: "the error surfaced as a traceback in the
    # failures list" (which is what consumers actually rely on).
    assert "traceback" in failure
    assert failure["traceback"]  # non-empty traceback string
