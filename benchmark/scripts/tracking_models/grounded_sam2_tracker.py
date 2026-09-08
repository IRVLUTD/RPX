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

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM2 requires CUDA.")
        from sam2.build_sam import build_sam2_video_predictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        cache = os.environ.get("HF_HOME")
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
        self.grounder = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_MODEL_ID, revision=GROUNDING_MODEL_REVISION, cache_dir=cache
        ).cuda().eval()
        self.video_predictor = build_sam2_video_predictor(
            SAM21_CONFIG, str(sam_checkpoint), device="cuda", apply_postprocessing=True
        )
        self.video_predictor.non_overlap_masks = True
        self.checkpoint_path = str(sam_checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(sam_checkpoint)
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

    def track(
        self,
        video_dir: Path,
        frame_shape: tuple[int, int],
        frame_count: int,
        text_prompts: Sequence[TextPrompt],
    ) -> tuple[list[np.ndarray], list[float]]:
        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(f"Grounded-SAM2 received {len(frames)} frames; expected {frame_count}.")
        first_image = Image.open(frames[0]).convert("RGB")
        detections: list[dict[str, Any]] = []
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
                    detections.append(
                        {
                            "prompt": prompt.prompt_text,
                            "source_mask_index": prompt.mask_index,
                            "label": str(label),
                            "score": float(score.detach().cpu()),
                            "box": [float(value) for value in box.detach().cpu()],
                        }
                    )
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
                        predictions[frame_index] = self._merge(
                            object_ids, mask_logits, frame_shape
                        )
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
                "checkpoint_sha256": self.checkpoint_sha256,
            },
            "prompts": [prompt.prompt_text for prompt in text_prompts],
            "detections": detections,
        }
        return [value for value in predictions if value is not None], latencies
