"""The shared phase-cell recording contract.

Both per-frame tasks (via ``cells_from_per_sample``) and per-clip tasks
(via ``cell_from_metrics`` / ``d1v_cell_row``) must emit the *same* cell
schema so Image Depth and Video Depth cells merge into one ``cells.parquet`` and feed
the same cross-phase / phase×difficulty Φ analysis.
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import Difficulty, Phase
from rpx_benchmark.cell_log import (
    FIXED_COLUMNS,
    cell_from_metrics,
    cells_from_per_sample,
)
from rpx_benchmark.data.clip_dataset import PhaseClip
from rpx_benchmark.tasks.video_depth import (
    VIDEO_DEPTH_TASK,
    video_depth_cell_row,
)

H = W = 8
_K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# cell_from_metrics
# --------------------------------------------------------------------------- #


def test_cell_from_metrics_has_full_fixed_schema():
    cell = cell_from_metrics(
        {"absrel": 0.1, "rmse": 0.2, "opw": float("nan")},
        model_name="DA3",
        task=VIDEO_DEPTH_TASK,
        scene_id="scene_a",
        phase=Phase.CLUTTER,
        n_samples=10,
        difficulty=Difficulty.HARD,
    )
    for col in FIXED_COLUMNS:
        assert col in cell
    assert cell["phase"] == "clutter"
    assert cell["difficulty"] == "hard"
    assert cell["scene_id"] == "scene_a"
    assert cell["n_samples"] == 10
    assert cell["metric:absrel"] == 0.1
    # NaN metric (e.g. OPW without a flow backend) is preserved as NaN.
    assert np.isnan(cell["metric:opw"])


def test_cell_from_metrics_difficulty_none():
    cell = cell_from_metrics(
        {"absrel": 0.1}, model_name="m", task=VIDEO_DEPTH_TASK,
        scene_id="s", phase="clean", n_samples=3,
    )
    assert cell["difficulty"] is None
    assert cell["phase"] == "clean"


def test_schema_parity_per_frame_vs_per_clip():
    # A per-frame cell and a per-clip cell share identical fixed columns.
    per_sample = [
        {"absrel": 0.1, "rmse": 0.2, "scene": "s1", "phase": Phase.CLUTTER,
         "difficulty": Difficulty.HARD, "id": "s1_clu_0"},
    ]
    pf_cell = cells_from_per_sample(per_sample, model_name="m", task="monocular_depth")[0]
    pc_cell = cell_from_metrics(
        {"absrel": 0.1, "rmse": 0.2}, model_name="m", task=VIDEO_DEPTH_TASK,
        scene_id="s1", phase=Phase.CLUTTER, n_samples=1, difficulty=Difficulty.HARD,
    )
    assert set(FIXED_COLUMNS) <= set(pf_cell) and set(FIXED_COLUMNS) <= set(pc_cell)
    # Same difficulty/phase encoding so cross-split grouping lines up.
    assert pf_cell["difficulty"] == pc_cell["difficulty"] == "hard"
    assert pf_cell["phase"] == pc_cell["phase"] == "clutter"


# --------------------------------------------------------------------------- #
# d1v_cell_row — clip -> canonical cell, end to end
# --------------------------------------------------------------------------- #


def _clip(scene="scene_b", phase_idx=1, t=4, difficulty=Difficulty.MEDIUM, poses=True):
    depth = np.full((t, H, W), 2.0, dtype=np.float32)
    valid = np.ones((t, H, W), dtype=bool)
    return PhaseClip(
        rgb_seq=np.zeros((t, H, W, 3), dtype=np.uint8),
        depth_gt_seq=depth,
        valid_mask_seq=valid,
        scene_id=scene,
        phase_idx=phase_idx,
        frame_indices=np.arange(t),
        intrinsics=_K,
        poses=np.stack([np.eye(4) for _ in range(t)]) if poses else None,
        difficulty=difficulty,
    )


def test_d1v_cell_row_is_canonical_and_tagged():
    clip = _clip()
    pred = np.full((4, H, W), 2.0, dtype=np.float32)
    row = video_depth_cell_row(pred, clip, model_name="DA3 (video)")

    assert row["task"] == VIDEO_DEPTH_TASK == "video_depth"
    assert row["model_name"] == "DA3 (video)"
    assert row["scene_id"] == "scene_b"
    assert row["phase"] == "interaction"
    assert row["difficulty"] == "medium"   # cross-split tag present
    assert row["n_samples"] == 4           # clip length T
    # Image Depth per-frame metrics + temporal metrics both present.
    assert "metric:absrel" in row and "metric:fscore_5cm" in row
    assert "metric:tae" in row and "metric:tgm" in row
    # poses present -> TAE finite; no flow backend -> OPW NaN.
    assert np.isfinite(row["metric:tae"])
    assert np.isnan(row["metric:opw"])


def test_d1v_cell_row_without_poses_has_nan_tae():
    clip = _clip(poses=False)
    pred = np.full((4, H, W), 2.0, dtype=np.float32)
    row = video_depth_cell_row(pred, clip, model_name="m")
    assert np.isnan(row["metric:tae"])
