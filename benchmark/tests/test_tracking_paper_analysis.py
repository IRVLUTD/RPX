from __future__ import annotations

from pathlib import Path

import pandas as pd

from rpx_benchmark.paper_tracking_analysis import (
    analyze_ego_tracking_cells,
    analyze_tracking_cells,
)


def _write_cells(root: Path, phase_shift: float) -> list[Path]:
    rows_by_tier: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    tiers = ["easy"] * 33 + ["medium"] * 33 + ["hard"] * 34
    for scene_index, tier in enumerate(tiers):
        baseline = scene_index / 250.0
        for phase in range(3):
            shift = phase_shift * phase
            rows_by_tier[tier].append(
                {
                    "model": "sam2",
                    "task": "object_tracking",
                    "split": tier,
                    "scene": f"scene{scene_index:03d}",
                    "phase": phase,
                    "metric:mota": 0.3 + baseline + shift,
                    "metric:idf1": 0.4 + baseline * 0.7 + shift * 0.8,
                    "metric:hota": 0.2 + baseline * 0.4 + shift * 0.5,
                    "metric:deta": 0.25 + baseline * 0.3 + shift * 0.4,
                    "metric:assa": 0.15 + baseline * 0.2 + shift * 0.3,
                    "metric:idsw": 20.0 + scene_index * 0.3 - shift * 10,
                }
            )
    paths = []
    for tier, rows in rows_by_tier.items():
        path = root / f"{tier}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths.append(path)
    return paths


def test_phase_invariant_tracking_cube_has_phi_one(tmp_path: Path) -> None:
    analysis, cells = analyze_tracking_cells(_write_cells(tmp_path, phase_shift=0.0))
    assert len(cells) == 300
    assert analysis["overall_phase_manova"]["phi_wilks"] == 1.0
    assert analysis["jedi_status"] == "bounds_missing"


def test_phase_shift_reduces_tracking_phi(tmp_path: Path) -> None:
    analysis, _ = analyze_tracking_cells(_write_cells(tmp_path, phase_shift=0.03))
    assert analysis["overall_phase_manova"]["phi_wilks"] < 1.0
    assert analysis["overall_phase_manova"]["p_value"] < 0.05


def test_ego_tracking_aggregates_one_clip_per_scene(tmp_path: Path) -> None:
    paths = []
    offset = 0
    for split, count in (("easy", 33), ("medium", 33), ("hard", 34)):
        rows = []
        for index in range(count):
            value = 0.5 + (offset + index) / 1000
            rows.append(
                {
                    "model": "xmem",
                    "task": "object_tracking",
                    "dataset_protocol": "ego",
                    "split": split,
                    "scene": f"scene{offset + index:03d}",
                    "phase": 0,
                    "metric:mota": value,
                    "metric:idf1": value,
                    "metric:hota": value,
                    "metric:deta": value,
                    "metric:assa": value,
                    "metric:idsw": 10 - value,
                    "latency_ms": 20.0,
                }
            )
        offset += count
        path = tmp_path / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths.append(path)

    analysis, cells = analyze_ego_tracking_cells(paths)

    assert len(cells) == 100
    assert analysis["dataset_protocol"] == "ego"
    assert analysis["split_cells"] == {"easy": 33, "medium": 33, "hard": 34}
    assert analysis["hardware_means"]["latency_ms"] == 20.0
