from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.api import (
    VideoDepthGroundTruth,
    VideoDepthPrediction,
    VideoSample,
)
from rpx_benchmark.exceptions import ConfigError
from rpx_benchmark.tasks._video_pipeline import _apply_clip_alignment


def _sample(gt: np.ndarray) -> VideoSample:
    return VideoSample(
        id="scene000__0",
        rgb_seq=np.zeros((*gt.shape, 3), dtype=np.uint8),
        ground_truth=VideoDepthGroundTruth(
            depth_map_seq=gt.astype(np.float32),
            valid_mask_seq=np.ones_like(gt, dtype=bool),
            frame_indices=np.arange(gt.shape[0], dtype=np.int32),
        ),
    )


def test_gemdepth_disparity_alignment_recovers_metric_depth() -> None:
    gt = np.array(
        [
            [[0.5, 0.75], [1.0, 1.5]],
            [[2.0, 2.5], [3.0, 4.0]],
        ],
        dtype=np.float32,
    )
    # Official GemDepth evaluation fits a * raw_inverse + b to 1 / GT.
    a, b = 2.5, 0.15
    raw_inverse = ((1.0 / gt) - b) / a

    aligned = _apply_clip_alignment(
        VideoDepthPrediction(depth_map_seq=raw_inverse),
        _sample(gt),
        mode="ls_disparity",
    )

    np.testing.assert_allclose(aligned.depth_map_seq, gt, rtol=1e-5, atol=1e-5)


def test_video_native_alignment_rejects_unknown_mode() -> None:
    gt = np.full((2, 2, 2), 1.0, dtype=np.float32)
    with pytest.raises(ConfigError, match="native alignment mode"):
        _apply_clip_alignment(
            VideoDepthPrediction(depth_map_seq=np.full_like(gt, 0.5)),
            _sample(gt),
            mode="invented",
        )
