from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tracking_models.grounded_sam2_tracker import GroundedSAM2Tracker  # noqa: E402
from tracking_models.sam3_1_tracker import SAM31Tracker  # noqa: E402
from tracking_text_runtime import TextPrompt  # noqa: E402


def test_grounded_sam2_merge_assigns_highest_positive_logit() -> None:
    logits = torch.tensor(
        [
            [[[1.0, -1.0], [0.2, -2.0]]],
            [[[0.1, 2.0], [0.8, -3.0]]],
        ]
    )
    result = GroundedSAM2Tracker._merge([4, 9], logits, (2, 2))
    np.testing.assert_array_equal(result, np.asarray([[4, 9], [9, 0]]))


def test_grounded_sam2_selects_one_distinct_region_per_prompt() -> None:
    prompts = (
        TextPrompt(1, "blue bottle", "1", "bottle"),
        TextPrompt(2, "white bottle", "2", "other_bottle"),
    )
    candidates = [
        {
            "prompt": "blue bottle",
            "source_mask_index": 1,
            "score": 0.9,
            "box": [0, 0, 10, 10],
        },
        {
            "prompt": "white bottle",
            "source_mask_index": 2,
            "score": 0.8,
            "box": [0, 0, 10, 10],
        },
        {
            "prompt": "white bottle",
            "source_mask_index": 2,
            "score": 0.7,
            "box": [20, 20, 30, 30],
        },
        {
            "prompt": "blue bottle",
            "source_mask_index": 1,
            "score": 0.6,
            "box": [40, 40, 50, 50],
        },
    ]

    selected, rejected = GroundedSAM2Tracker._select_one_to_one_detections(candidates, prompts)

    assert [(item["prompt"], item["box"]) for item in selected] == [
        ("blue bottle", [0, 0, 10, 10]),
        ("white bottle", [20, 20, 30, 30]),
    ]
    assert {item["rejection_reason"] for item in rejected} == {
        "region_claimed_by_other_prompt",
        "lower_score_for_same_prompt",
    }


def test_sam31_merge_offsets_instances_and_accepts_singleton_channel() -> None:
    target = np.zeros((2, 2), dtype=np.int32)
    confidence = np.full((2, 2), -np.inf, dtype=np.float32)
    masks = np.asarray(
        [
            [[[True, True], [False, False]]],
            [[[False, True], [True, False]]],
        ]
    )
    SAM31Tracker._merge(
        target,
        confidence,
        masks,
        np.asarray([0.4, 0.9]),
        np.asarray([12, 13]),
    )
    np.testing.assert_array_equal(target, np.asarray([[12, 13], [13, 0]]))


def test_sam31_merge_rejects_wrong_mask_shape() -> None:
    with pytest.raises(RuntimeError, match="returned mask shape"):
        SAM31Tracker._merge(
            np.zeros((2, 2), dtype=np.int32),
            np.full((2, 2), -np.inf, dtype=np.float32),
            np.zeros((1, 3, 3), dtype=bool),
            np.asarray([1.0]),
            np.asarray([1]),
        )
