"""GroundingDINO text detections followed by official SAM 2.1 tracking."""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tracking_text_runtime import TextPrompt

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
    model_id = GROUNDING_MODEL_ID
    model_revision = GROUNDING_MODEL_REVISION
    source_revision = GROUNDED_SAM2_SOURCE_REVISION
    checkpoint_filename = "model.safetensors"
    adapter_label = "Grounded-SAM2"
    prompt_type = "text"
    tracking_mode = "scene-vocabulary-text-detection-then-box-prompted-vos"
    vocabulary_name = "RPX-primary-color-plus-canonical-name-v1"
    vocabulary_size = 70
    box_threshold = 0.35
    text_threshold = 0.25
    cross_prompt_iou_threshold = 0.70

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM2 requires CUDA.")
        from sam2.build_sam import build_sam2_video_predictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        cache = os.environ.get("HF_HOME")
        grounding_checkpoint = Path(
            hf_hub_download(
                repo_id=GROUNDING_MODEL_ID,
                filename=self.checkpoint_filename,
                revision=GROUNDING_MODEL_REVISION,
                cache_dir=cache,
            )
        )
        sam_checkpoint = Path(
            hf_hub_download(
                repo_id=SAM21_MODEL_ID,
                filename=SAM21_CHECKPOINT,
                revision=SAM21_MODEL_REVISION,
                cache_dir=cache,
            )
        )
        self.processor = AutoProcessor.from_pretrained(
            GROUNDING_MODEL_ID, revision=GROUNDING_MODEL_REVISION, cache_dir=cache
        )
        self.grounder = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_MODEL_ID, revision=GROUNDING_MODEL_REVISION, cache_dir=cache
            )
            .cuda()
            .eval()
        )
        self.video_predictor = build_sam2_video_predictor(
            SAM21_CONFIG, str(sam_checkpoint), device="cuda", apply_postprocessing=True
        )
        self.video_predictor.non_overlap_masks = True
        self.checkpoint_path = str(grounding_checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(grounding_checkpoint)
        self.sam_checkpoint_path = str(sam_checkpoint.resolve())
        self.sam_checkpoint_sha256 = _sha256(sam_checkpoint)
        self.parameter_count = int(
            sum(parameter.numel() for parameter in self.grounder.parameters())
            + sum(parameter.numel() for parameter in self.video_predictor.parameters())
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
    def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
        left_top_x = max(float(left[0]), float(right[0]))
        left_top_y = max(float(left[1]), float(right[1]))
        right_bottom_x = min(float(left[2]), float(right[2]))
        right_bottom_y = min(float(left[3]), float(right[3]))
        intersection = max(0.0, right_bottom_x - left_top_x) * max(0.0, right_bottom_y - left_top_y)
        left_area = max(0.0, float(left[2]) - float(left[0])) * max(
            0.0, float(left[3]) - float(left[1])
        )
        right_area = max(0.0, float(right[2]) - float(right[0])) * max(
            0.0, float(right[3]) - float(right[1])
        )
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _select_one_to_one_detections(
        cls,
        candidates: Sequence[dict[str, Any]],
        prompts: Sequence[TextPrompt],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Greedily assign at most one distinct image region to each prompt.

        GroundingDINO is queried independently for each RPX prompt, so the raw
        candidate lists are not mutually exclusive.  A global score ordering
        lets a prompt fall back to its next candidate when its best box was
        already claimed, while rejecting near-identical regions across labels.
        Missing detections remain honest false negatives.
        """

        prompt_keys = {(int(prompt.mask_index), prompt.prompt_text) for prompt in prompts}
        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        claimed_prompts: set[tuple[int, str]] = set()
        ordered = sorted(
            (dict(candidate) for candidate in candidates),
            key=lambda candidate: (
                -float(candidate["score"]),
                int(candidate["source_mask_index"]),
                tuple(float(value) for value in candidate["box"]),
            ),
        )
        for candidate in ordered:
            prompt_key = (
                int(candidate["source_mask_index"]),
                str(candidate["prompt"]),
            )
            if prompt_key not in prompt_keys:
                rejected.append({**candidate, "rejection_reason": "unknown_prompt"})
                continue
            if prompt_key in claimed_prompts:
                rejected.append({**candidate, "rejection_reason": "lower_score_for_same_prompt"})
                continue
            conflict = next(
                (
                    accepted
                    for accepted in selected
                    if cls._box_iou(candidate["box"], accepted["box"])
                    >= cls.cross_prompt_iou_threshold
                ),
                None,
            )
            if conflict is not None:
                rejected.append(
                    {
                        **candidate,
                        "rejection_reason": "region_claimed_by_other_prompt",
                        "conflicts_with_prompt": conflict["prompt"],
                        "conflict_iou": cls._box_iou(candidate["box"], conflict["box"]),
                    }
                )
                continue
            selected.append(candidate)
            claimed_prompts.add(prompt_key)
        return selected, rejected

    def track(
        self,
        video_dir: Path,
        frame_shape: tuple[int, int],
        frame_count: int,
        text_prompts: Sequence[TextPrompt],
    ) -> tuple[list[np.ndarray], list[float]]:
        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(
                f"Grounded-SAM2 received {len(frames)} frames; expected {frame_count}."
            )
        first_image = Image.open(frames[0]).convert("RGB")
        candidates: list[dict[str, Any]] = []
        torch.cuda.synchronize()
        detection_started = time.perf_counter()
        with torch.inference_mode():
            for prompt in text_prompts:
                inputs = self.processor(
                    images=first_image,
                    text=prompt.prompt_text + ".",
                    return_tensors="pt",
                ).to("cuda")
                outputs = self.grounder(**inputs)
                result = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[first_image.size[::-1]],
                )[0]
                for box, score, label in zip(
                    result["boxes"], result["scores"], result["labels"], strict=True
                ):
                    candidates.append(
                        {
                            "prompt": prompt.prompt_text,
                            "source_mask_index": prompt.mask_index,
                            "label": str(label),
                            "score": float(score.detach().cpu()),
                            "box": [float(value) for value in box.detach().cpu()],
                        }
                    )
        detections, rejected_detections = self._select_one_to_one_detections(
            candidates, text_prompts
        )
        selected_prompt_keys = {
            (int(detection["source_mask_index"]), str(detection["prompt"]))
            for detection in detections
        }
        missing_prompts = [
            {
                "prompt": prompt.prompt_text,
                "source_mask_index": prompt.mask_index,
                "source_catalog_id": prompt.source_catalog_id,
                "object_id": prompt.object_id,
            }
            for prompt in text_prompts
            if (prompt.mask_index, prompt.prompt_text) not in selected_prompt_keys
        ]
        torch.cuda.synchronize()
        detection_ms = (time.perf_counter() - detection_started) * 1000

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
                for predicted_id, detection in enumerate(detections, 1):
                    self.video_predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=predicted_id,
                        box=np.asarray(detection["box"], dtype=np.float32),
                    )
                    detection["predicted_track_id"] = predicted_id
                if not detections:
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
            first_image.close()
            if state is not None:
                with contextlib.suppress(Exception):
                    self.video_predictor.reset_state(state)
            torch.cuda.empty_cache()
        latencies[0] += detection_ms
        missing = [index for index, value in enumerate(predictions) if value is None]
        if missing:
            raise RuntimeError(f"Grounded-SAM2 returned no masks for frames {missing[:10]}")
        self._metadata = {
            "initialization": "RPX fixed scene vocabulary -> GroundingDINO boxes -> SAM2",
            "grounding_model": {
                "repo": GROUNDING_MODEL_ID,
                "revision": GROUNDING_MODEL_REVISION,
            },
            "sam2_model": {
                "repo": SAM21_MODEL_ID,
                "revision": SAM21_MODEL_REVISION,
                "checkpoint_sha256": self.sam_checkpoint_sha256,
            },
            "grounding_checkpoint_sha256": self.checkpoint_sha256,
            "selection_protocol": {
                "name": "global_greedy_one_prompt_one_region",
                "cross_prompt_iou_threshold": self.cross_prompt_iou_threshold,
                "candidate_count": len(candidates),
                "prompt_count": len(text_prompts),
                "selected_track_count": len(detections),
                "missing_prompt_count": len(missing_prompts),
            },
            "prompts": [
                {
                    "prompt": prompt.prompt_text,
                    "source_mask_index": prompt.mask_index,
                    "source_catalog_id": prompt.source_catalog_id,
                    "object_id": prompt.object_id,
                }
                for prompt in text_prompts
            ],
            "detections": detections,
            "rejected_detections": rejected_detections,
            "missing_prompts": missing_prompts,
            "frames": [
                {
                    "frame_index": frame_index,
                    "frame": f"{frame_index:05d}",
                    "tracks": [
                        {
                            "track_id": int(detection["predicted_track_id"]),
                            "class_name": str(detection["prompt"]),
                            "source_mask_index": int(detection["source_mask_index"]),
                        }
                        for detection in detections
                    ],
                }
                for frame_index in range(frame_count)
            ],
        }
        return [value for value in predictions if value is not None], latencies
