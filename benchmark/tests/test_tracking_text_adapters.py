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


def test_grounded_sam2_merge_assigns_highest_positive_logit() -> None:
    logits = torch.tensor(
        [
            [[[1.0, -1.0], [0.2, -2.0]]],
            [[[0.1, 2.0], [0.8, -3.0]]],
        ]
    )
    result = GroundedSAM2Tracker._merge([4, 9], logits, (2, 2))
    np.testing.assert_array_equal(result, np.asarray([[4, 9], [9, 0]]))


def test_grounded_sam2_derives_tight_half_open_gt_boxes() -> None:
    mask = np.asarray([[0, 4, 4], [9, 4, 0], [9, 0, 0]], dtype=np.int32)

    assert GroundedSAM2Tracker._object_boxes(mask) == [
        (4, [1.0, 0.0, 3.0, 2.0]),
        (9, [0.0, 1.0, 1.0, 3.0]),
    ]


def test_sam31_derives_normalized_xywh_gt_boxes() -> None:
    mask = np.asarray([[0, 4, 4, 0], [0, 4, 0, 0]], dtype=np.int32)

    assert SAM31Tracker._object_boxes(mask) == [
        (4, [1.0, 0.0, 3.0, 2.0], [0.25, 0.0, 0.5, 1.0]),
    ]


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


def test_sam31_multiplexes_gt_boxes_and_handles_empty_points(monkeypatch, tmp_path) -> None:
    class Predictor:
        def __init__(self) -> None:
            self.closed_sessions: list[str] = []
            self.add_requests: list[dict] = []
            self.start_requests: list[dict] = []
            self.propagate_requests: list[dict] = []

        def handle_request(self, request):
            if request["type"] == "start_session":
                self.start_requests.append(request)
                return {"session_id": f"session-{request['resource_path']}"}
            if request["type"] == "close_session":
                self.closed_sessions.append(request["session_id"])
            if request["type"] == "add_prompt":
                self.add_requests.append(request)
            return {}

        def handle_stream_request(self, request):
            self.propagate_requests.append(request)
            yield {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.asarray([7, 9]),
                    "out_probs": np.asarray([0.9, 0.8]),
                    "out_binary_masks": np.asarray(
                        [
                            [[[True, False], [False, False]]],
                            [[[False, False], [False, True]]],
                        ]
                    ),
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
        first_frame_mask=np.asarray([[7, 0], [0, 9]], dtype=np.int32),
        frame_count=3,
    )

    np.testing.assert_array_equal(predictions[0], np.asarray([[7, 0], [0, 9]]))
    np.testing.assert_array_equal(predictions[1], np.zeros((2, 2), dtype=np.int32))
    np.testing.assert_array_equal(predictions[2], np.zeros((2, 2), dtype=np.int32))
    assert latencies[0] > 0
    assert latencies[1:] == [0.0, 0.0]
    assert len(tracker.predictor.start_requests) == 1
    assert len(tracker.predictor.closed_sessions) == 1
    assert len(tracker.predictor.add_requests) == 2
    assert len(tracker.predictor.propagate_requests) == 1
    assert tracker.predictor.add_requests[0]["points"] == [
        [0.0, 0.0],
        [0.5, 0.5],
    ]
    assert tracker.predictor.add_requests[0]["point_labels"] == [2, 3]
    assert tracker.predictor.add_requests[0]["obj_id"] == 7
    assert tracker.predictor.add_requests[0]["text"] is None
    assert tracker.predictor.add_requests[1]["obj_id"] == 9
    assert tracker.prediction_metadata()["multiplex"] == {
        "session_count_per_clip": 1,
        "propagation_count_per_clip": 1,
        "object_identity": "explicit_ground_truth_obj_id",
    }
    for record in tracker.prediction_metadata()["objects"]:
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
            first_frame_mask=np.asarray([[1, 0], [0, 0]], dtype=np.int32),
            frame_count=1,
        )
