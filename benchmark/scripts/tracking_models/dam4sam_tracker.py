"""Official DAM4SAM adapter for the RPX D3 multi-object protocol."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

DAM4SAM_MODEL_ID = "facebook/sam2.1-hiera-large"
DAM4SAM_MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
DAM4SAM_SOURCE_REVISION = "9c954504b39ebca4c412f207be0787c26bfac85a"
DAM4SAM_CHECKPOINT = "sam2.1_hiera_large.pt"
DAM4SAM_TRACKER_NAME = "sam21pp-L"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DAM4SAMTracker:
    """Run one official DAM4SAM state per RPX object with shared model weights.

    The linked DAM4SAM release is a single-target tracker. RPX evaluates it as
    a multi-object VOS baseline by maintaining an independent official DAM
    state for each frame-zero object. All states share the same SAM2.1-L
    predictor weights; only their temporal memories are independent.
    """

    model_name = "dam4sam"
    model_id = DAM4SAM_MODEL_ID
    model_revision = DAM4SAM_MODEL_REVISION
    source_revision = DAM4SAM_SOURCE_REVISION
    checkpoint_filename = DAM4SAM_CHECKPOINT
    adapter_label = "DAM4SAM"
    prompt_type = "mask"
    tracking_mode = "independent-sot-instances-multi-object-mask-vos"
    native_multi_object = False

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production DAM4SAM adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the DAM4SAM adapter.")

        checkpoint = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename=self.checkpoint_filename,
                revision=self.model_revision,
                cache_dir=os.environ.get("HF_HOME"),
            )
        )
        os.environ["DAM4SAM_CHECKPOINT"] = str(checkpoint.resolve())

        from dam4sam_tracker import DAM4SAMTracker as OfficialDAM4SAMTracker

        self.official_tracker_class = OfficialDAM4SAMTracker
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        self._template: Any | None = None
        self.parameter_count = 0

    @staticmethod
    def _shared_tracker(template: Any, tracker_class: type[Any]) -> Any:
        """Create an independent DAM state while reusing immutable model weights."""
        tracker = tracker_class.__new__(tracker_class)
        for attribute in (
            "checkpoint",
            "model_cfg",
            "input_image_size",
            "img_mean",
            "img_std",
            "predictor",
        ):
            setattr(tracker, attribute, getattr(template, attribute))
        tracker.tracking_times = []
        return tracker

    @staticmethod
    def _merge_object_masks(
        object_ids: list[int],
        masks: list[np.ndarray],
        confidences: list[float],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Resolve overlaps by official predicted IoU, then stable object order."""
        if not (len(object_ids) == len(masks) == len(confidences)):
            raise ValueError("object IDs, masks and confidences must have equal lengths.")
        output = np.zeros(shape, dtype=np.int32)
        best = np.full(shape, -np.inf, dtype=np.float32)
        for object_id, mask, confidence in zip(
            object_ids, masks, confidences, strict=True
        ):
            binary = np.asarray(mask, dtype=bool)
            if binary.shape != shape:
                raise RuntimeError(
                    f"DAM4SAM returned mask shape {binary.shape}; expected {shape}."
                )
            score = float(confidence)
            replace = binary & (score > best)
            output[replace] = object_id
            best[replace] = score
        return output

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        initial = np.asarray(first_frame_mask)
        if initial.ndim != 2 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("first_frame_mask must be a 2-D integer instance mask.")
        object_ids = [int(value) for value in np.unique(initial) if value > 0]
        if not object_ids:
            raise ValueError("first_frame_mask contains no positive object IDs.")

        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(
                f"DAM4SAM received {len(frames)} frames; expected {frame_count}."
            )

        first_image = Image.open(frames[0]).convert("RGB")
        trackers: list[Any] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for index, object_id in enumerate(object_ids):
                if index == 0:
                    tracker = self.official_tracker_class(DAM4SAM_TRACKER_NAME)
                    self._template = tracker
                    self.parameter_count = int(
                        sum(
                            parameter.numel()
                            for parameter in tracker.predictor.parameters()
                        )
                    )
                else:
                    assert self._template is not None
                    tracker = self._shared_tracker(
                        self._template, self.official_tracker_class
                    )
                tracker.initialize(first_image, (initial == object_id).astype(np.uint8))
                trackers.append(tracker)

        predictions: list[np.ndarray] = [initial.astype(np.int32, copy=True)]
        latencies_ms: list[float] = [0.0] * frame_count

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for frame_index, frame_path in enumerate(frames[1:], start=1):
                image = Image.open(frame_path).convert("RGB")
                frame_masks: list[np.ndarray] = []
                confidences: list[float] = []
                torch.cuda.synchronize()
                started = time.perf_counter()
                for tracker in trackers:
                    result = tracker.track(image)
                    frame_masks.append(np.asarray(result["pred_mask"], dtype=np.uint8))
                    confidences.append(float(result.get("pred_iou", 0.0)))
                torch.cuda.synchronize()
                latencies_ms[frame_index] = (time.perf_counter() - started) * 1000.0
                predictions.append(
                    self._merge_object_masks(
                        object_ids,
                        frame_masks,
                        confidences,
                        initial.shape,
                    )
                )

        first_image.close()
        del trackers
        self._template = None
        torch.cuda.empty_cache()
        return predictions, latencies_ms
