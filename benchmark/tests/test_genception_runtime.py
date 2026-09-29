from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from genception_runtime import (  # noqa: E402
    GenCeptionRuntime,
    restore_genception_grid,
    restore_prediction_grid,
    segmentation_video_to_masks,
    segmentation_video_to_scores,
)


class FakePipeline:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"video": self.output}

    @staticmethod
    def decode_depth(video):
        values = np.asarray(video)
        return values[..., 0]


def test_restore_prediction_grid_restores_time_and_space() -> None:
    source = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    restored = restore_prediction_grid(source, target_thw=(5, 4, 6))
    assert restored.shape == (5, 4, 6)
    assert restored.dtype == np.float32


def test_genception_short_clip_uses_unpadded_prefix() -> None:
    prediction = np.arange(81, dtype=np.float32)[:, None, None]
    restored = restore_genception_grid(prediction, input_thw=(3, 1, 1))
    np.testing.assert_array_equal(restored[:, 0, 0], [0, 1, 2])


def test_refvos_rgb_decode_supports_float_and_uint8() -> None:
    floating = np.asarray([[[[0.4, 0.6, 0.2]]]], dtype=np.float32)
    assert segmentation_video_to_scores(floating).item() == pytest.approx(0.6)
    assert segmentation_video_to_masks(floating).item()
    integer = np.asarray([[[[0, 127, 255]]]], dtype=np.uint8)
    assert segmentation_video_to_scores(integer).item() == pytest.approx(1.0)


def test_runtime_depth_and_prompt_cache_preserve_input_shape(tmp_path: Path) -> None:
    output = np.ones((1, 3, 2, 4, 3), dtype=np.float32)
    encoder_calls = []

    def encoder(text: str) -> np.ndarray:
        encoder_calls.append(text)
        return np.zeros((1, 226, 4096), dtype=np.float32)

    runtime = GenCeptionRuntime(
        "1.3b", tmp_path, tmp_path, pipeline=FakePipeline(output), prompt_encoder=encoder
    )
    video = np.zeros((2, 5, 7, 3), dtype=np.uint8)
    scores_a = runtime.predict_mask_scores(video, "red cup")
    scores_b = runtime.predict_mask_scores(video, "red cup")
    assert scores_a.shape == (2, 5, 7)
    np.testing.assert_array_equal(scores_a, scores_b)
    assert encoder_calls == ["red cup"]


def test_runtime_rejects_bad_prompt_embedding_shape(tmp_path: Path) -> None:
    runtime = GenCeptionRuntime(
        "14b",
        tmp_path,
        tmp_path,
        pipeline=FakePipeline(np.zeros((1, 1, 2, 2, 3), dtype=np.float32)),
        prompt_encoder=lambda _: np.zeros((1, 8, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="1, 226, 4096"):
        runtime.predict_masks(np.zeros((1, 2, 2, 3), dtype=np.uint8), "object")
