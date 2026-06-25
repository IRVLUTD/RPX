"""End-to-end test for the Video Depth runner.

Exercises the per-clip pipeline with a fake adapter and a synthetic
two-clip manifest. Proves:

* download → ``VideoDepthDataset`` → per-clip ``model.predict`` → metric
  suite → cell-log writes complete without GPU or real model weights;
* relative-depth models get per-clip ``(s, t)`` alignment applied
  before metric computation;
* metric-depth models pass through unchanged;
* the cell-log row carries the correct ``scene_id`` / ``phase`` /
  ``frame_budget`` fields, ready for downstream JEDI / Φ aggregation.

No network access, no GPU — synthetic everything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.api import (
    BenchmarkModel,
    TaskType,
    VideoDepthGroundTruth,
    VideoDepthPrediction,
    VideoSample,
)
from rpx_benchmark.tasks._video_pipeline import VideoTaskRunConfig, run_video_pipeline


# --------------------------------------------------------------------------- #
# Fake data — two scenes × three phases × 4 frames each
# --------------------------------------------------------------------------- #


def _make_fake_manifest(root: Path) -> Path:
    """Write a tiny extractive manifest + RGB/depth PNGs for two clips."""
    import imageio.v3 as iio

    H, W, T = 8, 8, 4
    samples = []
    for scene_id in ["scene_a", "scene_b"]:
        for phase in [0, 1, 2]:
            frames_dir = root / "extracted" / "scenes" / scene_id / str(phase) / "rgb"
            depth_dir = root / "extracted" / "scenes" / scene_id / str(phase) / "depth"
            frames_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            frame_files, depth_files = [], []
            for t in range(T):
                rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
                # Depth in mm (D435 convention): 500–4000 mm range
                depth_mm = (500 + np.random.rand(H, W) * 3500).astype(np.uint16)
                rgb_p = frames_dir / f"{t:05d}.png"
                d_p = depth_dir / f"{t:05d}.png"
                iio.imwrite(rgb_p, rgb)
                iio.imwrite(d_p, depth_mm)
                frame_files.append(str(rgb_p.relative_to(root)))
                depth_files.append(str(d_p.relative_to(root)))
            samples.append({
                "scene_id": scene_id,
                "phase": phase,
                "difficulty": "easy",
                "frame_filenames": frame_files,
                "depth_filenames": depth_files,
            })

    manifest = {
        "task": TaskType.VIDEO_DEPTH.value,
        "split": "easy",
        "root": str(root),
        "samples": samples,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


# --------------------------------------------------------------------------- #
# Fake video-depth model — returns GT * scale + shift (deterministic)
# --------------------------------------------------------------------------- #


class _FakeVideoDepthModel(BenchmarkModel):
    """Returns GT-shaped predictions so metrics are well-defined.

    Two flavours via ``depth_output_kind``:
    * ``"metric"`` — returns ``gt * 1.05`` (5% bias; should score
      reasonably but not perfectly).
    * ``"relative"`` — returns ``gt * 3 + 0.7`` (deliberate affine
      distortion that alignment must undo).
    """

    task = TaskType.VIDEO_DEPTH
    name = "fake-video-depth"

    def __init__(self, output_kind: str = "metric") -> None:
        self.depth_output_kind = output_kind

    def setup(self) -> None:
        return None

    def predict(self, batch):
        out = []
        for sample in batch:
            gt = sample.ground_truth.depth_map_seq.astype(np.float32)
            if self.depth_output_kind == "relative":
                pred = gt * 3.0 + 0.7  # affine distortion
            else:
                pred = gt * 1.05  # small metric bias
            out.append(VideoDepthPrediction(depth_map_seq=pred))
        return out


# --------------------------------------------------------------------------- #
# The actual tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_dataset_root(tmp_path, monkeypatch):
    """Build a fake manifest and stub download_split to return its path."""
    manifest_path = _make_fake_manifest(tmp_path)

    def _fake_download(task, split, repo_id, cache_dir=None, revision=None):
        return manifest_path

    # Patch download_split in the video pipeline module
    from rpx_benchmark.tasks import _video_pipeline as vp

    monkeypatch.setattr(vp, "download_split", _fake_download)
    return tmp_path


def test_video_pipeline_runs_metric_model(fake_dataset_root, tmp_path):
    """End-to-end smoke for a metric-depth video model."""
    out_dir = tmp_path / "out_metric"
    cfg = VideoTaskRunConfig(
        model=_FakeVideoDepthModel(output_kind="metric"),
        split="easy",
        device="cpu",
        output_dir=str(out_dir),
    )
    result, _dr, paths = run_video_pipeline(
        task=TaskType.VIDEO_DEPTH,
        primary_metric="absrel",
        cfg=cfg,
    )

    # All declared outputs present.
    assert paths["json"].exists()
    assert paths["cells"].exists()
    assert paths["markdown"].exists()

    # Result has at least the per-frame error metric keys.
    assert result.aggregated, "result.aggregated is empty"
    assert "absrel" in result.aggregated

    # Cell log has the expected number of cells (2 scenes × 3 phases = 6).
    import pyarrow.parquet as pq

    cells_df = pq.read_table(paths["cells"]).to_pandas()
    assert len(cells_df) == 6
    assert set(cells_df["scene_id"]) == {"scene_a", "scene_b"}
    assert set(cells_df["phase"]) == {"clutter", "interaction", "clean"}
    assert (cells_df["frame_budget"] == 0).all()
    # Model was 5% biased; AbsRel should be ~0.05 ± a bit
    assert (cells_df["metric:absrel"] > 0).all()
    assert (cells_df["metric:absrel"] < 0.2).all()


def test_video_pipeline_aligns_relative_model(fake_dataset_root, tmp_path):
    """Relative-depth model gets per-clip (s, t) alignment applied,
    so its AbsRel comes back near zero even though the raw prediction
    was scaled by 3× and shifted by 0.7 m.
    """
    out_dir = tmp_path / "out_relative"
    cfg = VideoTaskRunConfig(
        model=_FakeVideoDepthModel(output_kind="relative"),
        split="easy",
        device="cpu",
        output_dir=str(out_dir),
    )
    result, _dr, paths = run_video_pipeline(
        task=TaskType.VIDEO_DEPTH,
        primary_metric="absrel",
        cfg=cfg,
    )

    import pyarrow.parquet as pq

    cells_df = pq.read_table(paths["cells"]).to_pandas()
    # After per-clip (s, t) alignment, AbsRel should be ~0 (the fake
    # model's "shape" is exactly GT, just affine-distorted).
    assert (cells_df["metric:absrel"] < 0.01).all(), (
        f"alignment failed — AbsRel still high: {cells_df['metric:absrel'].tolist()}"
    )


def test_video_pipeline_frame_budget(fake_dataset_root, tmp_path):
    """``frame_budget=2`` with stride sampling cuts each 4-frame clip to 2;
    the cell-log row records the budget for downstream filtering.
    """
    out_dir = tmp_path / "out_budget"
    cfg = VideoTaskRunConfig(
        model=_FakeVideoDepthModel(output_kind="metric"),
        split="easy",
        device="cpu",
        output_dir=str(out_dir),
        frame_budget=2,
        sampling="stride",
    )
    _result, _dr, paths = run_video_pipeline(
        task=TaskType.VIDEO_DEPTH,
        primary_metric="absrel",
        cfg=cfg,
    )

    import pyarrow.parquet as pq

    cells_df = pq.read_table(paths["cells"]).to_pandas()
    assert (cells_df["frame_budget"] == 2).all()


# --------------------------------------------------------------------------- #
# Frame-as-video integration: Image Depth adapter shape → Video Depth cell log row
# --------------------------------------------------------------------------- #


class _FakePerFrameAdapter:
    """Mimics scripts/depth_models/* — per-frame callable.

    Returns a depth map that's 1.02× the value-of-channel-0 squared,
    so it's deterministic and metric-shaped without needing GT context.
    """

    def __call__(self, rgb):
        if isinstance(rgb, (list, tuple)):
            return [(r[..., 0].astype(np.float32) ** 2) * 1.02 + 0.3 for r in rgb]
        return (rgb[..., 0].astype(np.float32) ** 2) * 1.02 + 0.3


def test_frame_as_video_runs_through_full_pipeline(fake_dataset_root, tmp_path):
    """The shim that wraps any per-frame adapter must produce a cell-log
    row populated with every Video Depth metric (spatial + temporal) when run
    through the full pipeline. This proves the team's first real
    adapter (DA-V2 in per-clip mode) will work end-to-end without any
    further runner changes.
    """
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent / "scripts")
    )
    from video_depth_models import FrameDepthAsVideo

    model = FrameDepthAsVideo(
        adapter=_FakePerFrameAdapter(),
        name="fake-frame-as-video",
        depth_output_kind="metric",
        frame_batch=2,
    )

    out_dir = tmp_path / "out_frame_as_video"
    cfg = VideoTaskRunConfig(
        model=model,
        split="easy",
        device="cpu",
        output_dir=str(out_dir),
    )
    _result, _dr, paths = run_video_pipeline(
        task=TaskType.VIDEO_DEPTH,
        primary_metric="absrel",
        cfg=cfg,
    )

    import pyarrow.parquet as pq

    cells_df = pq.read_table(paths["cells"]).to_pandas()
    assert len(cells_df) == 6  # 2 scenes × 3 phases

    # Every Video Depth metric must be present on the row — spatial + temporal.
    metric_cols = [c for c in cells_df.columns if c.startswith("metric:")]
    expected_metrics = {"absrel", "rmse", "delta1", "tae", "opw"}
    for m in expected_metrics:
        assert f"metric:{m}" in metric_cols, (
            f"Cell log missing metric:{m} — Video Depth pipeline regressed. "
            f"Found: {sorted(metric_cols)}"
        )
