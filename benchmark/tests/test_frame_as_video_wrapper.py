"""Tests for the FrameDepthAsVideo wrapper.

Proves the shim that turns any per-frame depth adapter into a Video Depth
:class:`BenchmarkModel`:

* per-frame outputs land in the right slot of the stacked sequence
  (frame ``t`` of the input → frame ``t`` of the output);
* frame_batch chunking is correct when ``T`` doesn't divide evenly
  by the chunk size;
* the adapter contract violation (wrong-count return, wrong-shape
  return) raises ``AdapterError`` with a hint instead of corrupting
  the sequence silently;
* the wrapper inherits ``depth_output_kind`` so the runner applies
  per-clip ``(s, t)`` alignment automatically for relative models.

No GPU, no real weights — a callable Python adapter is enough to
exercise every code path in the shim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable so video_depth_models is reachable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from video_depth_models import FrameDepthAsVideo

from rpx_benchmark.api import (
    TaskType,
    VideoDepthGroundTruth,
    VideoSample,
)
from rpx_benchmark.exceptions import AdapterError

# --------------------------------------------------------------------------- #
# Fake per-frame adapters
# --------------------------------------------------------------------------- #


class _IdentityAdapter:
    """Returns each frame's first channel as float32 depth.

    Used as a deterministic stand-in for a real per-frame model. The
    output is ``(H, W) float32`` of the same shape as the input,
    matching every Image Depth adapter's contract.
    """

    def __call__(self, rgb):
        if isinstance(rgb, (list, tuple)):
            return [r[..., 0].astype(np.float32) for r in rgb]
        return rgb[..., 0].astype(np.float32)


class _OffByOneAdapter:
    """Returns one fewer depth map than asked — contract violation."""

    def __call__(self, rgb):
        rgbs = list(rgb) if isinstance(rgb, (list, tuple)) else [rgb]
        # Deliberately drop the last frame
        return [r[..., 0].astype(np.float32) for r in rgbs[:-1]]


class _BadShapeAdapter:
    """Returns 3-D arrays instead of 2-D — contract violation."""

    def __call__(self, rgb):
        rgbs = list(rgb) if isinstance(rgb, (list, tuple)) else [rgb]
        return [np.zeros((1, *r.shape[:2]), dtype=np.float32) for r in rgbs]


# --------------------------------------------------------------------------- #
# Synthetic VideoSample
# --------------------------------------------------------------------------- #


def _make_video_sample(T: int = 6, H: int = 8, W: int = 8) -> VideoSample:
    """Build a VideoSample whose RGB channel 0 = frame index (uint8).

    Pairs nicely with ``_IdentityAdapter``: the output sequence's frame
    ``t`` should be a ``(H, W)`` float32 array uniformly equal to
    ``t``, so the test can verify both ordering and float-cast.
    """
    rgb_seq = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        rgb_seq[t, :, :, 0] = t  # frame index in channel 0
    gt = VideoDepthGroundTruth(
        depth_map_seq=np.ones((T, H, W), dtype=np.float32),
        valid_mask_seq=np.ones((T, H, W), dtype=bool),
        frame_indices=np.arange(T, dtype=np.int32),
    )
    return VideoSample(
        id="fake",
        rgb_seq=rgb_seq,
        ground_truth=gt,
        metadata={"frame_indices": list(range(T))},
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_wrapper_preserves_frame_ordering():
    """Output sequence frame t must equal input sequence frame t."""
    wrapper = FrameDepthAsVideo(
        adapter=_IdentityAdapter(), name="fake-id", frame_batch=4
    )
    sample = _make_video_sample(T=6)
    [pred] = wrapper.predict([sample])
    assert pred.depth_map_seq.shape == (6, 8, 8)
    assert pred.depth_map_seq.dtype == np.float32
    for t in range(6):
        # Identity adapter returns channel 0; we set channel 0 = t.
        assert (pred.depth_map_seq[t] == float(t)).all(), (
            f"Frame {t}: identity wrapper broke ordering"
        )


def test_wrapper_handles_uneven_frame_batch():
    """T=7 with frame_batch=4 → chunks of 4 + 3. Order must be preserved
    across the chunk boundary.
    """
    wrapper = FrameDepthAsVideo(
        adapter=_IdentityAdapter(), name="fake-id", frame_batch=4
    )
    sample = _make_video_sample(T=7)
    [pred] = wrapper.predict([sample])
    assert pred.depth_map_seq.shape == (7, 8, 8)
    for t in range(7):
        assert (pred.depth_map_seq[t] == float(t)).all(), (
            f"Frame {t}: uneven-chunk wrapper broke ordering"
        )


def test_wrapper_rejects_off_by_one_adapter():
    """Adapter that returns the wrong count → AdapterError, not silent
    corruption of the sequence.
    """
    wrapper = FrameDepthAsVideo(
        adapter=_OffByOneAdapter(), name="fake-bad-count", frame_batch=4
    )
    sample = _make_video_sample(T=6)
    with pytest.raises(AdapterError, match="depth maps for a chunk"):
        wrapper.predict([sample])


def test_wrapper_rejects_bad_shape_adapter():
    """Adapter that returns 3-D arrays → AdapterError."""
    wrapper = FrameDepthAsVideo(
        adapter=_BadShapeAdapter(), name="fake-bad-shape", frame_batch=4
    )
    sample = _make_video_sample(T=4)
    with pytest.raises(AdapterError, match=r"\(H, W\) float32"):
        wrapper.predict([sample])


def test_wrapper_inherits_depth_output_kind():
    """The runner reads depth_output_kind to decide alignment. The
    wrapper must propagate it from the constructor.
    """
    metric = FrameDepthAsVideo(
        adapter=_IdentityAdapter(),
        name="fake-metric",
        depth_output_kind="metric",
    )
    relative = FrameDepthAsVideo(
        adapter=_IdentityAdapter(),
        name="fake-relative",
        depth_output_kind="relative",
    )
    assert metric.depth_output_kind == "metric"
    assert relative.depth_output_kind == "relative"


def test_wrapper_task_is_video_depth():
    """The runner refuses (model, dataset) task mismatches. The
    wrapper must declare itself as VIDEO_DEPTH.
    """
    wrapper = FrameDepthAsVideo(adapter=_IdentityAdapter(), name="fake")
    assert wrapper.task is TaskType.VIDEO_DEPTH


def test_wrapper_setup_is_noop():
    """Adapters in scripts/depth_models/ load weights at __init__; the
    wrapper's setup must be safe to call (the runner always calls it).
    """
    wrapper = FrameDepthAsVideo(adapter=_IdentityAdapter(), name="fake")
    wrapper.setup()  # no exception
