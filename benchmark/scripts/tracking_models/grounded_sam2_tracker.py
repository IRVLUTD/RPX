"""Grounded-SAM2 ablation using GT boxes with its official SAM 2.1 backend."""

from __future__ import annotations

import contextlib
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SAM21_MODEL_ID = "facebook/sam2.1-hiera-large"
SAM21_MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
SAM21_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM21_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDED_SAM2_SOURCE_REVISION = "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GroundedSAM2Tracker:
    model_name = "grounded-sam2"
    model_id = SAM21_MODEL_ID
    model_revision = SAM21_MODEL_REVISION
    source_revision = GROUNDED_SAM2_SOURCE_REVISION
    checkpoint_filename = SAM21_CHECKPOINT
    adapter_label = "Grounded-SAM2"
    prompt_type = "box"
    tracking_mode = "grounded-sam2-sam2.1-backend-tight-gt-box-initialized"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM2 requires CUDA.")
        from sam2.build_sam import build_sam2_video_predictor
        sam_checkpoint = Path(
            hf_hub_download(
                repo_id=SAM21_MODEL_ID,
                filename=SAM21_CHECKPOINT,
                revision=SAM21_MODEL_REVISION,
            )
        )
        self.video_predictor = build_sam2_video_predictor(
            SAM21_CONFIG, str(sam_checkpoint), device="cuda", apply_postprocessing=True
        )
        self.video_predictor.non_overlap_masks = True
        self.checkpoint_path = str(sam_checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(sam_checkpoint)
        self.sam_checkpoint_path = str(sam_checkpoint.resolve())
        self.sam_checkpoint_sha256 = _sha256(sam_checkpoint)
        self.parameter_count = int(
            sum(parameter.numel() for parameter in self.video_predictor.parameters())
        )
        self._metadata: dict[str, Any] = {}

    @staticmethod
    def _merge(
        object_ids: Sequence[int], logits: torch.Tensor, shape: tuple[int, int]
    ) -> np.ndarray:
        if len(object_ids) == 0:
            return np.zeros(shape, dtype=np.int32)
        values = logits[:, 0].detach().float().cpu().numpy()
        best = values.argmax(axis=0)
        scores = values.max(axis=0)
        ids = np.asarray(object_ids, dtype=np.int32)
        output = np.zeros(shape, dtype=np.int32)
        foreground = scores > 0
        output[foreground] = ids[best[foreground]]
        return output

    def prediction_metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _object_boxes(first_frame_mask: np.ndarray) -> list[tuple[int, list[float]]]:
        boxes: list[tuple[int, list[float]]] = []
        for value in np.unique(first_frame_mask):
            object_id = int(value)
            if object_id == 0:
                continue
            ys, xs = np.where(first_frame_mask == object_id)
            if xs.size:
                boxes.append(
                    (
                        object_id,
                        [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max() + 1),
                            float(ys.max() + 1),
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
        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(
                f"Grounded-SAM2 received {len(frames)} frames; expected {frame_count}."
            )
        boxes = self._object_boxes(first_frame_mask)

        state = None
        predictions: list[np.ndarray | None] = [None] * frame_count
        latencies = [0.0] * frame_count
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.video_predictor.init_state(
                    video_path=str(video_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=True,
                )
                started = time.perf_counter()
                for object_id, box in boxes:
                    self.video_predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        box=np.asarray(box, dtype=np.float32),
                    )
                torch.cuda.synchronize()
                latencies[0] = (time.perf_counter() - started) * 1000
                if not boxes:
                    predictions = [np.zeros(frame_shape, dtype=np.int32) for _ in frames]
                else:
                    iterator = self.video_predictor.propagate_in_video(state)
                    while True:
                        started = time.perf_counter()
                        try:
                            frame_index, object_ids, mask_logits = next(iterator)
                        except StopIteration:
                            break
                        torch.cuda.synchronize()
                        latencies[frame_index] = (time.perf_counter() - started) * 1000
                        predictions[frame_index] = self._merge(object_ids, mask_logits, frame_shape)
        finally:
            if state is not None:
                with contextlib.suppress(Exception):
                    self.video_predictor.reset_state(state)
            torch.cuda.empty_cache()
        missing = [index for index, value in enumerate(predictions) if value is None]
        if missing:
            raise RuntimeError(f"Grounded-SAM2 returned no masks for frames {missing[:10]}")
        self._metadata = {
            "initialization": "tight GT bounding boxes on frame 0",
            "prompt_type": "box",
            "grounding_dino_bypassed": True,
            "model_label_note": (
                "Grounded-SAM2 ablation using its SAM 2.1 video backend; "
                "GroundingDINO is bypassed because initialization is supplied by GT boxes."
            ),
            "sam2_model": {
                "repo": SAM21_MODEL_ID,
                "revision": SAM21_MODEL_REVISION,
                "checkpoint_sha256": self.sam_checkpoint_sha256,
            },
            "objects": [
                {
                    "object_id": object_id,
                    "box_xyxy_pixels": box,
                }
                for object_id, box in boxes
            ],
        }
        return [value for value in predictions if value is not None], latencies
