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


def test_sam31_filters_base_predictor_kwargs_unsupported_by_multiplex() -> None:
    calls = []

    class Model:
        def init_state(
            self,
            resource_path,
            offload_video_to_cpu=False,
            async_loading_frames=False,
        ):
            calls.append((resource_path, offload_video_to_cpu, async_loading_frames))
            return {"ready": True}

    tracker = object.__new__(SAM31Tracker)
    tracker.predictor = type("Predictor", (), {"model": Model()})()

    ignored = tracker._patch_init_state_compatibility()
    result = tracker.predictor.model.init_state(
        resource_path="frames",
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
        video_loader_type="async",
    )

    assert result == {"ready": True}
    assert calls == [("frames", True, True)]
    assert ignored == ("offload_state_to_cpu", "video_loader_type")


def test_sam31_empty_points_ends_only_the_affected_prompt(monkeypatch, tmp_path) -> None:
    class Predictor:
        def __init__(self) -> None:
            self.closed_sessions: list[str] = []

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": f"session-{request['resource_path']}"}
            if request["type"] == "close_session":
                self.closed_sessions.append(request["session_id"])
            return {}

        def handle_stream_request(self, request):
            del request
            yield {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.asarray([7]),
                    "out_probs": np.asarray([0.9]),
                    "out_binary_masks": np.asarray([[[[True, False], [False, False]]]]),
                },
            }
            raise RuntimeError("No points are provided; please add points first")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tracker = object.__new__(SAM31Tracker)
    tracker.predictor = Predictor()
    tracker.checkpoint_sha256 = "test-checkpoint"
    tracker._ignored_init_state_arguments = ()

    predictions, latencies = tracker.track(
        video_dir=tmp_path,
        frame_shape=(2, 2),
        frame_count=3,
        text_prompts=(TextPrompt(1, "red cup", "1", "cup"),),
    )

    np.testing.assert_array_equal(predictions[0], np.asarray([[1, 0], [0, 0]]))
    np.testing.assert_array_equal(predictions[1], np.zeros((2, 2), dtype=np.int32))
    np.testing.assert_array_equal(predictions[2], np.zeros((2, 2), dtype=np.int32))
    assert latencies[0] > 0
    assert latencies[1:] == [0.0, 0.0]
    assert len(tracker.predictor.closed_sessions) == 1
    record = tracker.prediction_metadata()["prompts"][0]
    assert record["propagation_status"] == "terminated_empty_points"
    assert record["last_output_frame_index"] == 0
    assert record["first_unprocessed_frame_index"] == 1
    assert record["unprocessed_frame_count"] == 2


def test_sam31_does_not_hide_unrelated_runtime_errors(monkeypatch, tmp_path) -> None:
    class Predictor:
        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": "session"}
            return {}

        def handle_stream_request(self, request):
            del request
            if False:
                yield
            raise RuntimeError("CUDA failure")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tracker = object.__new__(SAM31Tracker)
    tracker.predictor = Predictor()
    tracker.checkpoint_sha256 = "test-checkpoint"
    tracker._ignored_init_state_arguments = ()

    with pytest.raises(RuntimeError, match="CUDA failure"):
        tracker.track(
            video_dir=tmp_path,
            frame_shape=(2, 2),
            frame_count=1,
            text_prompts=(TextPrompt(1, "red cup", "1", "cup"),),
        )
