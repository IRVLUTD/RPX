"""Native SAM 3.1 GT-box-initialized video tracking adapter for RPX."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

SAM31_MODEL_ID = "facebook/sam3.1"
SAM31_MODEL_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_SOURCE_REVISION = "6dbb02bd38288df755dfa1378000a861e65b84f6"
SAM31_EMPTY_POINTS_ERROR = "No points are provided; please add points first"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SAM31Tracker:
    model_name = "sam3.1"
    model_id = SAM31_MODEL_ID
    model_revision = SAM31_MODEL_REVISION
    source_revision = SAM31_SOURCE_REVISION
    checkpoint_filename = "sam3.1_multiplex.pt"
    adapter_label = "SAM 3.1"
    prompt_type = "box"
    tracking_mode = "native-tight-gt-box-multiplex-video-segmentation"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 requires CUDA.")
        from sam3.model_builder import build_sam3_predictor

        checkpoint = Path(
            hf_hub_download(
                repo_id=SAM31_MODEL_ID,
                revision=SAM31_MODEL_REVISION,
                filename=self.checkpoint_filename,
            )
        )
        self.predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint),
            compile=False,
            warm_up=False,
            use_fa3=False,
            async_loading_frames=True,
        )
        self._ignored_init_state_arguments = self._patch_init_state_compatibility()
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        model = getattr(self.predictor, "model", None)
        self.parameter_count = (
            int(sum(parameter.numel() for parameter in model.parameters()))
            if model is not None
            else 0
        )
        self._metadata: dict[str, Any] = {}

    def _patch_init_state_compatibility(self) -> tuple[str, ...]:
        """Filter base-predictor kwargs unsupported by the multiplex model.

        At the pinned upstream revision, ``Sam3BasePredictor.start_session``
        always forwards ``offload_state_to_cpu``.  SAM 3.1's multiplex
        ``init_state`` does not accept that keyword, despite using the same
        base predictor.  Filter only arguments absent from the concrete model
        signature; supported arguments and model behavior are unchanged.
        """

        original = self.predictor.model.init_state
        signature = inspect.signature(original)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return ()
        supported = set(signature.parameters)
        base_arguments = {
            "resource_path",
            "offload_video_to_cpu",
            "offload_state_to_cpu",
            "async_loading_frames",
            "video_loader_type",
        }
        ignored = tuple(sorted(base_arguments - supported))

        def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
            filtered = {key: value for key, value in kwargs.items() if key in supported}
            return original(*args, **filtered)

        self.predictor.model.init_state = compatible_init_state
        return ignored

    @staticmethod
    def _merge(
        target: np.ndarray,
        confidence: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        ids: np.ndarray,
    ) -> None:
        for mask, score, object_id in zip(masks, scores, ids, strict=True):
            binary = np.asarray(mask, dtype=bool).squeeze()
            if binary.shape != target.shape:
                raise RuntimeError(
                    f"SAM 3.1 returned mask shape {binary.shape}; expected {target.shape}."
                )
            replace = binary & (float(score) > confidence)
            target[replace] = int(object_id)
            confidence[replace] = float(score)

    def prediction_metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _object_boxes(
        first_frame_mask: np.ndarray,
    ) -> list[tuple[int, list[float], list[float]]]:
        """Return GT object IDs, pixel XYXY boxes, and normalized XYWH boxes."""

        height, width = first_frame_mask.shape
        boxes: list[tuple[int, list[float], list[float]]] = []
        for value in np.unique(first_frame_mask):
            object_id = int(value)
            if object_id == 0:
                continue
            ys, xs = np.where(first_frame_mask == object_id)
            if xs.size == 0:
                continue
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1
            boxes.append(
                (
                    object_id,
                    [float(x0), float(y0), float(x1), float(y1)],
                    [
                        x0 / width,
                        y0 / height,
                        (x1 - x0) / width,
                        (y1 - y0) / height,
                    ],
                )
            )
        return boxes

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        frame_shape = first_frame_mask.shape
        predictions = [np.zeros(frame_shape, dtype=np.int32) for _ in range(frame_count)]
        confidences = [np.full(frame_shape, -np.inf, dtype=np.float32) for _ in range(frame_count)]
        latencies = [0.0] * frame_count
        boxes = self._object_boxes(first_frame_mask)
        prompt_records: list[dict[str, Any]] = [
            {
                "object_id": object_id,
                "box_xyxy_pixels": box_xyxy,
                "box_xywh_normalized": box_xywh_normalized,
                "propagation_status": "pending",
            }
            for object_id, box_xyxy, box_xywh_normalized in boxes
        ]
        session_id = None
        last_output_frame_index: int | None = None
        try:
            response = self.predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_dir),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = response["session_id"]
            torch.cuda.synchronize()
            started = time.perf_counter()

            # SAM's native box encoding is two corner points carrying labels 2
            # and 3.  Supplying each box this way reaches SAM 3.1's explicit
            # object-ID refinement path, so every GT identity is added to one
            # multiplex state before the single video propagation below.
            for object_id, _, box_xywh_normalized in boxes:
                x, y, width, height = box_xywh_normalized
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": None,
                        "points": [[x, y], [x + width, y + height]],
                        "point_labels": [2, 3],
                        "clear_old_points": True,
                        "obj_id": object_id,
                        "rel_coordinates": True,
                    }
                )

            try:
                for response in self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": frame_count,
                    }
                ):
                    torch.cuda.synchronize()
                    frame_index = int(response["frame_index"])
                    last_output_frame_index = frame_index
                    output = response["outputs"]
                    output_ids = np.asarray(output["out_obj_ids"], dtype=np.int32)
                    scores = np.asarray(
                        output.get("out_probs", np.ones(len(output_ids))),
                        dtype=np.float32,
                    )
                    self._merge(
                        predictions[frame_index],
                        confidences[frame_index],
                        np.asarray(output["out_binary_masks"]),
                        scores,
                        output_ids,
                    )
                    now = time.perf_counter()
                    latencies[frame_index] = (now - started) * 1000
                    started = now
            except RuntimeError as error:
                if str(error) != SAM31_EMPTY_POINTS_ERROR:
                    raise
                first_unprocessed_frame = (
                    0 if last_output_frame_index is None else last_output_frame_index + 1
                )
                for record in prompt_records:
                    record.update(
                        {
                            "propagation_status": "terminated_empty_points",
                            "propagation_error": str(error),
                            "last_output_frame_index": last_output_frame_index,
                            "first_unprocessed_frame_index": first_unprocessed_frame,
                            "unprocessed_frame_count": max(
                                0, frame_count - first_unprocessed_frame
                            ),
                            "recovery": (
                                "Unprocessed frames remain background; the clip "
                                "is retained and subsequent clips continue."
                            ),
                        }
                    )
                print(
                    "WARNING: SAM 3.1 multiplex propagation lost all tracker "
                    f"points after frame {last_output_frame_index}; continuing "
                    "with background for the remaining frames.",
                    flush=True,
                )
            else:
                for record in prompt_records:
                    record["propagation_status"] = "complete"
        finally:
            if session_id is not None:
                with contextlib.suppress(Exception):
                    self.predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )
        self._metadata = {
            "initialization": "tight GT bounding boxes on frame 0",
            "prompt_type": "box",
            "box_format": {
                "adapter_input": "normalized_xyxy_corner_points_labels_2_3",
                "pixel_reference": "half_open_xyxy",
                "source": "ground_truth_first_frame_instance_mask",
            },
            "multiplex": {
                "session_count_per_clip": 1,
                "propagation_count_per_clip": 1,
                "object_identity": "explicit_ground_truth_obj_id",
            },
            "model": {
                "repo": SAM31_MODEL_ID,
                "revision": SAM31_MODEL_REVISION,
                "checkpoint_sha256": self.checkpoint_sha256,
            },
            "compatibility": {
                "filtered_unsupported_init_state_arguments": list(
                    self._ignored_init_state_arguments
                )
            },
            "objects": prompt_records,
        }
        return predictions, latencies
