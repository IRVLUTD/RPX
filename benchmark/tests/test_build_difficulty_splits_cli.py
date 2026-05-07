"""CI smoke test for build_difficulty_splits.py (Stage 2 CLI).

Exercises the happy path (synthetic 27-feature CSV → 5-method splits + tier
files) plus failure paths (missing CSV, MI weights file). Per-method math is
covered in test_esd_scoring.py.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "experiments" / "scripts"
if not (_SCRIPTS_DIR / "build_difficulty_splits.py").is_file():
    pytest.skip(f"CLI script not at expected path: {_SCRIPTS_DIR}", allow_module_level=True)
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_difficulty_splits  # noqa: E402

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_synthetic_csv(path: Path, n_scenes: int = 5) -> None:
    """Write a Stage-1-shaped CSV with three phases per scene (n_scenes×3 rows)."""
    rng = np.random.default_rng(0)
    fieldnames = ["scene_id", "phase", "n_frames_total", "n_frames_used", *FEATURE_NAMES]
    with path.open("w", newline="") as f:
        f.write("# schema_version=2 n_features=27 generator=test\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in range(n_scenes):
            for p in range(3):
                row = {
                    "scene_id": f"scene{s + 1}.fixture",
                    "phase": p,
                    "n_frames_total": 250,
                    "n_frames_used": 250,
                }
                # Random but feature-typical values to avoid degenerate
                # constant-column tertiles.
                for name in FEATURE_NAMES:
                    row[name] = float(rng.uniform(0, 1))
                writer.writerow(row)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_cli_writes_json_csv_and_tier_files(tmp_path):
    # The full method set (mean_pn, median_pn, max_pn, pca_pc1, kmeans3)
    # is only produced when sklearn is installed — without it, only the
    # pn-family methods are emitted. Skip rather than assert a stale set.
    pytest.importorskip("sklearn")

    csv_path = tmp_path / "phase_esd_splits.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=10)  # 30 entries → 10 per tier

    rc = build_difficulty_splits.main(
        [
            "--features",
            str(csv_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    json_path = out_dir / "phase_difficulty.json"
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == build_difficulty_splits.SCHEMA_VERSION
    assert payload["primary_method"] == "mean_pn"
    assert payload["scoring_version"] == "effort_stratified_v1"
    assert payload["summary"]["n_entries"] == 30
    # All 5 methods landed (sklearn is in the test env).
    assert set(payload["available_methods"]) == {
        "mean_pn",
        "median_pn",
        "max_pn",
        "pca_pc1",
        "kmeans3",
    }
    # Tertile counts ≈ 10/10/10.
    counts = payload["summary"]["tertile_counts"]
    assert sum(counts.values()) == 30
    assert all(8 <= counts[t] <= 12 for t in ("easy", "medium", "hard"))

    # Per-row schema: every entry has scores/tiers for every method.
    sample = next(iter(payload["phases"].values()))
    for key in (
        "scene_id",
        "phase",
        "n_frames_used",
        "raw_features",
        "pn_features",
        "contributions",
        "scores",
        "percentiles",
        "tiers",
        "tier",
        "score",
        "percentile",
    ):
        assert key in sample
    assert set(sample["raw_features"]) == set(FEATURE_NAMES)
    assert set(sample["pn_features"]) == set(FEATURE_NAMES)
    assert set(sample["contributions"]) == set(FEATURE_NAMES)
    assert set(sample["scores"]) == {"mean_pn", "median_pn", "max_pn", "pca_pc1", "kmeans3"}

    # Provenance carries the input hash.
    assert len(payload["provenance"]["input_sha256"]) == 64

    # Flat CSV: comment header + DictWriter header + 30 data rows.
    csv_out = out_dir / "phase_difficulty.csv"
    lines = csv_out.read_text().splitlines()
    assert lines[0].startswith("# schema_version=")
    assert "scene_id" in lines[1]
    assert len(lines) == 2 + 30

    # easy/medium/hard.txt: union covers every entry exactly once.
    seen = set()
    for tier in ("easy", "medium", "hard"):
        for line in (out_dir / f"{tier}.txt").read_text().splitlines():
            line = line.strip()
            if line:
                assert line not in seen
                seen.add(line)
    assert len(seen) == 30

    # Provenance JSON exists and matches the payload.
    prov = json.loads((out_dir / "splits_provenance.json").read_text())
    assert prov["input_sha256"] == payload["provenance"]["input_sha256"]


def test_cli_emits_structured_metric_fields(tmp_path):
    """v2 schema must include consensus_tier, confidence, weight_flip_rate,
    per_category on every row, plus matching counts in summary."""
    csv_path = tmp_path / "phase_esd_splits.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=10)
    # Run with a small n_perturb to keep the test fast.
    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(out_dir),
                "--confidence-n-perturb",
                "100",
            ]
        )
        == 0
    )

    payload = json.loads((out_dir / "phase_difficulty.json").read_text())

    # Top-level: schema bumped + new keys present.
    assert payload["schema_version"] >= 2
    assert payload["confidence_labels"] == ["high", "medium", "low"]
    assert "feature_categories" in payload
    assert len(payload["feature_categories"]) == 10

    # Summary block carries new aggregates.
    s = payload["summary"]
    assert set(s["consensus_tier_counts"]) == {"easy", "medium", "hard"}
    assert sum(s["consensus_tier_counts"].values()) == 30
    assert set(s["confidence_counts"]) == {"high", "medium", "low"}
    assert sum(s["confidence_counts"].values()) == 30
    assert 0 <= s["primary_vs_consensus_match_pct"] <= 100

    # Every row carries the new per-row fields.
    for _key, row in payload["phases"].items():
        assert row["consensus_tier"] in {"easy", "medium", "hard"}
        assert row["confidence"] in {"high", "medium", "low"}
        assert 0.0 <= row["weight_flip_rate"] <= 1.0
        assert set(row["per_category"]) == set(payload["feature_categories"])
        for v in row["per_category"].values():
            assert 0.0 <= v <= 1.0

    # CSV mirrors the new columns.
    csv_lines = (out_dir / "phase_difficulty.csv").read_text().splitlines()
    header = csv_lines[1]
    for col in (
        "consensus_tier",
        "confidence",
        "weight_flip_rate",
        "cat_annotation_effort",
        "cat_fisheye_stereo",
    ):
        assert col in header, f"missing column {col} in flat CSV header"


def test_cli_default_weights_are_effort_stratified_and_sum_to_one(tmp_path):
    csv_path = tmp_path / "x.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=5)

    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    payload = json.loads((out_dir / "phase_difficulty.json").read_text())
    weights = payload["provenance"]["weights"]
    assert len(weights) == len(FEATURE_NAMES)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Default scoring is effort_stratified — iter features carry alpha/2 each.
    assert payload["scoring_version"] == "effort_stratified_v1"
    assert weights["iter_mean"] > weights["depth_invalid"]
    assert weights["iter_mean"] == pytest.approx(weights["iter_max"])


def test_cli_writes_scene_splits_json_with_correct_tertile_counts(tmp_path):
    """The shipped scene_splits.json keeps all 3 phases of a scene together
    and produces ⌊N/3⌋ Easy + ⌊N/3⌋ Medium + remainder Hard."""
    csv_path = tmp_path / "x.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=33)  # 33 scenes → 11/11/11

    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    payload = json.loads((out_dir / "scene_splits.json").read_text())
    assert payload["n_scenes"] == 33
    counts = payload["tier_counts"]
    assert counts["easy"] == 11
    assert counts["medium"] == 11
    assert counts["hard"] == 11

    splits = payload["splits"]
    # No scene appears in more than one tier.
    seen = set()
    for tier in ("easy", "medium", "hard"):
        for sid in splits[tier]:
            assert sid not in seen
            seen.add(sid)
    assert len(seen) == 33

    # Per-phase tier breakdown is preserved per scene.
    assert "scene_detail" in payload
    detail = payload["scene_detail"]
    assert len(detail) == 33
    sample_sid = next(iter(detail))
    sample = detail[sample_sid]
    assert sample["scene_tier"] in {"easy", "medium", "hard"}
    assert isinstance(sample["scene_score"], float)
    assert len(sample["phase_scores"]) == 3
    assert len(sample["phase_tiers"]) == 3
    assert sample["phase_indices"] == [0, 1, 2]
    for t in sample["phase_tiers"]:
        assert t in {"easy", "medium", "hard"}

    # n_scenes=100 → 33/33/34 (canonical paper spec).
    csv_path100 = tmp_path / "x100.csv"
    out_dir100 = tmp_path / "out100"
    _write_synthetic_csv(csv_path100, n_scenes=100)
    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path100),
                "--out-dir",
                str(out_dir100),
            ]
        )
        == 0
    )
    payload100 = json.loads((out_dir100 / "scene_splits.json").read_text())
    assert payload100["tier_counts"]["easy"] == 33
    assert payload100["tier_counts"]["medium"] == 33
    assert payload100["tier_counts"]["hard"] == 34


def test_cli_uniform_weighting_when_explicitly_requested(tmp_path):
    csv_path = tmp_path / "x.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=5)

    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(out_dir),
                "--weighting",
                "uniform",
            ]
        )
        == 0
    )

    payload = json.loads((out_dir / "phase_difficulty.json").read_text())
    assert payload["scoring_version"] == "uniform_v1"
    weights = payload["provenance"]["weights"]
    # All features carry the same weight.
    assert weights["iter_mean"] == pytest.approx(weights["depth_invalid"])
    assert weights["iter_mean"] == pytest.approx(1.0 / len(FEATURE_NAMES))


def test_cli_with_mi_weights_uses_mi_v1_scoring(tmp_path):
    csv_path = tmp_path / "x.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=5)

    # Hand-roll a tiny MI weights artifact concentrating mass on one feature.
    mi_path = tmp_path / "mi.json"
    mi_path.write_text(
        json.dumps(
            {
                "weights": {FEATURE_NAMES[0]: 1.0, FEATURE_NAMES[1]: 0.5},
                "calibration_model_set": ["dummy_model"],
                "metric_used": "MI",
            }
        )
    )

    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(out_dir),
                "--weights-mi",
                str(mi_path),
            ]
        )
        == 0
    )

    payload = json.loads((out_dir / "phase_difficulty.json").read_text())
    assert payload["scoring_version"] == "mi_v1"
    weights = payload["provenance"]["weights"]
    # Mass should be concentrated on the two features in the artifact.
    assert weights[FEATURE_NAMES[0]] > weights[FEATURE_NAMES[2]]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_cli_primary_choice_drives_tier_files(tmp_path):
    csv_path = tmp_path / "x.csv"
    out_dir = tmp_path / "out"
    _write_synthetic_csv(csv_path, n_scenes=10)

    # Two runs differ only in --primary; tier txt files should align with the
    # chosen method's tier column in each case.
    assert (
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(out_dir),
                "--primary",
                "max_pn",
            ]
        )
        == 0
    )
    payload = json.loads((out_dir / "phase_difficulty.json").read_text())
    assert payload["primary_method"] == "max_pn"

    # Sanity: each row's `tier` matches `tiers["max_pn"]`.
    for row in payload["phases"].values():
        assert row["tier"] == row["tiers"]["max_pn"]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_cli_returns_2_on_missing_features_csv(tmp_path):
    rc = build_difficulty_splits.main(
        [
            "--features",
            str(tmp_path / "does_not_exist.csv"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_cli_rejects_csv_missing_feature_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    # Write a CSV with no feature columns.
    csv_path.write_text("# schema_version=2\nscene_id,phase\nsceneA,0\n")
    with pytest.raises(SystemExit):
        build_difficulty_splits.main(
            [
                "--features",
                str(csv_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
