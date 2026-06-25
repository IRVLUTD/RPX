"""Tests for the make_numpy_video_depth_model factory.

Closes the architectural gap: Image Depth has had make_numpy_depth_model
since day one; Video Depth was missing the analogue. This factory lets
a user wrap a plain ``(T, H, W, 3) uint8 → (T, H, W) float32`` callable
as a BenchmarkableModel that goes through the same InputAdapter /
OutputAdapter pipeline as every other task.
"""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.adapters import make_numpy_video_depth_model
from rpx_benchmark.api import (
    TaskType,
    VideoDepthGroundTruth,
    VideoDepthPrediction,
    VideoSample,
)
from rpx_benchmark.exceptions import AdapterError


def _make_clip(T: int = 4, H: int = 8, W: int = 8) -> VideoSample:
    rgb = (np.random.rand(T, H, W, 3) * 255).astype(np.uint8)
    gt = VideoDepthGroundTruth(
        depth_map_seq=np.ones((T, H, W), dtype=np.float32),
        valid_mask_seq=np.ones((T, H, W), dtype=bool),
        frame_indices=np.arange(T, dtype=np.int32),
    )
    return VideoSample(
        id="fake",
        rgb_seq=rgb,
        ground_truth=gt,
        metadata={"frame_indices": list(range(T))},
    )


def test_factory_returns_benchmarkable_model_with_video_depth_task():
    """The wrapped model must declare task=VIDEO_DEPTH so the runner's
    dataset/task validation passes.
    """
    def fn(rgb_seq):
        T, H, W, _ = rgb_seq.shape
        return np.full((T, H, W), 2.0, dtype=np.float32)

    bm = make_numpy_video_depth_model(fn, name="mine")
    assert bm.task is TaskType.VIDEO_DEPTH
    assert bm.name == "mine"
    assert bm.depth_output_kind == "metric"  # default


def test_factory_propagates_depth_output_kind():
    """Relative-depth callables must produce a model the runner
    recognises so per-clip (s, t) alignment fires.
    """
    def fn(rgb_seq):
        T, H, W, _ = rgb_seq.shape
        return np.ones((T, H, W), dtype=np.float32)

    bm = make_numpy_video_depth_model(fn, depth_output_kind="relative")
    assert bm.depth_output_kind == "relative"


def test_factory_predict_returns_video_depth_prediction():
    """The output adapter must wrap the callable's numpy output into
    VideoDepthPrediction (the type the metric calculators expect).
    """
    def fn(rgb_seq):
        T, H, W, _ = rgb_seq.shape
        return np.full((T, H, W), 1.5, dtype=np.float32)

    bm = make_numpy_video_depth_model(fn)
    bm.setup()
    [pred] = bm.predict([_make_clip()])
    assert isinstance(pred, VideoDepthPrediction)
    assert pred.depth_map_seq.shape == (4, 8, 8)
    assert pred.depth_map_seq.dtype == np.float32
    assert np.all(pred.depth_map_seq == 1.5)


def test_factory_resizes_per_frame_when_shape_mismatches():
    """Per-frame resizing parallels the Image Depth factory's behaviour:
    when (H', W') != (H, W), we resize per-frame instead of erroring.
    T must still match — frame-count divergence is a contract bug.
    """
    def fn(rgb_seq):
        T, H, W, _ = rgb_seq.shape
        # Return at half resolution; output adapter should resize back.
        return np.full((T, H // 2, W // 2), 1.0, dtype=np.float32)

    bm = make_numpy_video_depth_model(fn)
    bm.setup()
    [pred] = bm.predict([_make_clip(T=4, H=8, W=8)])
    assert pred.depth_map_seq.shape == (4, 8, 8)


def test_factory_rejects_callable_that_drops_frames():
    """Returning fewer frames than the input clip is a contract bug,
    not something the output adapter is allowed to paper over."""
    def fn(rgb_seq):
        T, H, W, _ = rgb_seq.shape
        return np.zeros((T - 1, H, W), dtype=np.float32)  # drops one frame

    bm = make_numpy_video_depth_model(fn)
    bm.setup()
    with pytest.raises(AdapterError, match="frames"):
        bm.predict([_make_clip(T=4)])


def test_factory_accepts_t_1_h_w_squeeze():
    """A common callable shape is (T, 1, H, W) — the output adapter
    should auto-squeeze the channel dim.
    """
    def fn(rgb_seq):
        T, H, W, _ = rgb_seq.shape
        return np.full((T, 1, H, W), 3.0, dtype=np.float32)

    bm = make_numpy_video_depth_model(fn)
    bm.setup()
    [pred] = bm.predict([_make_clip()])
    assert pred.depth_map_seq.shape == (4, 8, 8)
    assert np.all(pred.depth_map_seq == 3.0)


def test_factory_rejects_2d_or_5d_output():
    """Bad-shape outputs must error loudly, not corrupt the cell log."""
    def bad_fn(rgb_seq):
        return np.zeros((8, 8), dtype=np.float32)  # 2-D, no T

    bm = make_numpy_video_depth_model(bad_fn)
    bm.setup()
    with pytest.raises(AdapterError, match=r"\(T, H, W\)"):
        bm.predict([_make_clip()])
