"""Tests for ``scripts/pose_comprehensive_metrics.py``.

Pin the per-pair metrics + AUC math against simple analytic cases so a
refactor can't silently change rotation / translation conventions or
the AUC normalisation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# pose_comprehensive_metrics lives under scripts/, not in the rpx_benchmark package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pose_comprehensive_metrics import (  # noqa: E402
    auc_pose_error,
    compute_run,
    rotation_error_deg,
    translation_angular_deg,
    translation_l2,
)

from rpx_benchmark.pose_metrics import evaluate_rcpe  # noqa: E402

# ────────────────────────  per-pair metrics  ──────────────────────────────


def test_rotation_error_deg_identity_is_zero():
    R = np.eye(3)
    assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-9)


def test_rotation_error_deg_matches_known_angle():
    """A 30° rotation about Z should yield exactly 30° geodesic error."""
    theta = np.deg2rad(30.0)
    R_pred = np.eye(3)
    R_gt = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    assert rotation_error_deg(R_pred, R_gt) == pytest.approx(30.0, abs=1e-6)


def test_rotation_error_deg_caps_at_180():
    """A 180° rotation about any axis yields exactly 180° error."""
    R_pred = np.eye(3)
    R_gt = np.diag([1, -1, -1]).astype(np.float64)  # 180° about X
    assert rotation_error_deg(R_pred, R_gt) == pytest.approx(180.0, abs=1e-6)


def test_translation_l2_known_distance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    assert translation_l2(a, b) == pytest.approx(1.0, abs=1e-9)


def test_translation_angular_deg_orthogonal():
    """Orthogonal unit vectors → 90° angular error."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert translation_angular_deg(a, b) == pytest.approx(90.0, abs=1e-6)


def test_translation_angular_deg_scale_invariant():
    """Same direction, different magnitudes → 0° angular error."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([5.0, 0.0, 0.0])
    assert translation_angular_deg(a, b) == pytest.approx(0.0, abs=1e-9)


def test_translation_angular_deg_zero_norm_returns_zero():
    """Zero-norm input should not blow up — return 0 by contract."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert translation_angular_deg(a, b) == 0.0


# ────────────────────────  AUC  ────────────────────────────────────────────


def test_auc_perfect_predictions_is_one():
    """When every pair has 0° error, AUC is 1.0 at every threshold."""
    errors = np.zeros(100)
    auc = auc_pose_error(errors, thresholds=(5.0, 10.0, 20.0))
    for k in ("auc_5deg", "auc_10deg", "auc_20deg"):
        assert auc[k] == pytest.approx(1.0, abs=1e-9), f"{k}: {auc[k]}"


def test_auc_all_above_threshold_is_zero():
    """When every pair fails by way more than the threshold, AUC is 0."""
    errors = np.full(100, 100.0)  # 100° error everywhere
    auc = auc_pose_error(errors, thresholds=(5.0, 10.0, 20.0))
    for k in ("auc_5deg", "auc_10deg", "auc_20deg"):
        assert auc[k] == pytest.approx(0.0, abs=1e-9), f"{k}: {auc[k]}"


def test_auc_empty_input_is_zero():
    auc = auc_pose_error(np.array([]), thresholds=(5.0,))
    assert auc["auc_5deg"] == 0.0


def test_auc_monotone_in_threshold():
    """AUC at higher thresholds should be ≥ AUC at lower ones (more pairs
    pass at higher thresholds, so cumulative fraction is higher)."""
    rng = np.random.default_rng(5_062_026)
    errors = rng.uniform(0, 30, size=500)
    auc = auc_pose_error(errors, thresholds=(5.0, 10.0, 20.0))
    assert auc["auc_5deg"] <= auc["auc_10deg"] + 1e-9
    assert auc["auc_10deg"] <= auc["auc_20deg"] + 1e-9


def test_compute_run_reads_published_npy_pose_manifest(tmp_path: Path):
    pose_dir = tmp_path / "scenes" / "scene004" / "0" / "cam_pose"
    pose_dir.mkdir(parents=True)
    np.save(pose_dir / "00000.npy", np.array([0, 0, 0, 0, 0, 0, 1.0]))
    # Raw T265 +Y/+Z are OpenCV -Y/-Z. The prediction below is expressed in
    # OpenCV coordinates and must compare exactly after the fixed basis change.
    np.save(pose_dir / "00005.npy", np.array([1, -2, -3, 0, 0, 0, 1.0]))

    manifest = {
        "root": str(tmp_path),
        "samples": [{
            "id": "scene004__0__00000__00005",
            "scene_id": "scene004",
            "phase": 0,
            "pose_a": "scenes/scene004/0/cam_pose/00000.npy",
            "pose_b": "scenes/scene004/0/cam_pose/00005.npy",
            "metadata": {
                "frame": "00000",
                "frame_b": "00005",
                "pair_stride": 5,
                "pair_type": "intra_phase",
            },
        }],
    }
    manifest_path = tmp_path / "pairs_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    predictions_path = tmp_path / "predictions.csv"
    fieldnames = [
        "sample_id", "scene_id", "phase", "phase_b", "frame_a", "frame_b",
        "R00", "R01", "R02", "R10", "R11", "R12",
        "R20", "R21", "R22", "tx", "ty", "tz",
    ]
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "sample_id": "scene004__0__00000__00005",
            "scene_id": "scene004", "phase": "0",
            "phase_b": "0",
            "frame_a": "00000", "frame_b": "00005",
            "R00": 1, "R01": 0, "R02": 0,
            "R10": 0, "R11": 1, "R12": 0,
            "R20": 0, "R21": 0, "R22": 1,
            "tx": 1, "ty": 2, "tz": 3,
        })

    result = compute_run(predictions_path, manifest_path)

    assert len(result["per_pair"]) == 1
    assert result["per_pair"][0]["rotation_error_deg"] == pytest.approx(0.0)
    assert result["per_pair"][0]["translation_error_m"] == pytest.approx(0.0)
    assert result["per_pair"][0]["translation_l2"] == pytest.approx(0.0)
    rcpe = evaluate_rcpe(result["per_pair"])
    assert rcpe["n_pairs"] == 1
    assert rcpe["aggregated"]["translation_error_m"] == pytest.approx(0.0)
