"""Official SAM2Long memory-tree adapter for the RPX D3 protocol."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

SAM2LONG_MODEL_ID = "facebook/sam2.1-hiera-large"
SAM2LONG_MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
SAM2LONG_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM2LONG_CONFIG = "sam2.1_hiera_l.yaml"
SAM2LONG_CONFIG_DIR = "/opt/rpx-models/sam2long/sam2/configs/sam2.1"
SAM2LONG_NUM_PATHWAYS = 3
SAM2LONG_IOU_THRESHOLD = 0.1
SAM2LONG_UNCERTAINTY = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SAM2LongTracker:
    """Run official SAM2Long using its constrained memory-tree search."""

    model_name = "sam2long"
    model_id = SAM2LONG_MODEL_ID
    model_revision = SAM2LONG_MODEL_REVISION
    checkpoint_filename = SAM2LONG_CHECKPOINT
    config_name = SAM2LONG_CONFIG
    config_directory = SAM2LONG_CONFIG_DIR
    adapter_label = "SAM2Long"
    prompt_type = "mask"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production SAM2Long adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the SAM2Long adapter.")

        config_dir = Path(os.environ.get("SAM2LONG_CONFIG_DIR", self.config_directory)).resolve()
        config_path = config_dir / self.config_name
        if not config_path.is_file():
            raise RuntimeError(
                f"SAM2Long config is missing: {config_path}. "
                "Use the pinned cumulative SAM2Long Docker image."
            )

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2_video_predictor

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(config_dir), version_base="1.3")

        checkpoint = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename=self.checkpoint_filename,
                revision=self.model_revision,
                cache_dir=os.environ.get("HF_HOME"),
            )
        )
        self.predictor = build_sam2_video_predictor(
            self.config_name,
            str(checkpoint),
            device=device,
            apply_postprocessing=True,
        )
        self.predictor.non_overlap_masks = True
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.parameter_count = int(
            sum(parameter.numel() for parameter in self.predictor.parameters())
        )

    @staticmethod
    def _combine_masks(
        object_ids: list[int], mask_logits: torch.Tensor, shape: tuple[int, int]
    ) -> np.ndarray:
        logits = mask_logits.detach().float().cpu().numpy()
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.shape != (len(object_ids), *shape):
            raise RuntimeError(
                f"SAM2Long returned mask shape {logits.shape}; "
                f"expected {(len(object_ids), *shape)}."
            )
        best_index = logits.argmax(axis=0)
        best_score = logits.max(axis=0)
        output = np.zeros(shape, dtype=np.int32)
        foreground = best_score > 0.0
        ids = np.asarray(object_ids, dtype=np.int32)
        output[foreground] = ids[best_index[foreground]]
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

        per_object_logits: list[list[torch.Tensor]] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for object_id in object_ids:
                state = self.predictor.init_state(
                    video_path=str(video_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )
                state["num_pathway"] = SAM2LONG_NUM_PATHWAYS
                state["iou_thre"] = SAM2LONG_IOU_THRESHOLD
                state["uncertainty"] = SAM2LONG_UNCERTAINTY
                try:
                    self.predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        mask=initial == object_id,
                    )

                    output_ids, output_logits = self.predictor.propagate_in_video(state)
                    if [int(value) for value in output_ids] != [object_id]:
                        raise RuntimeError(
                            f"SAM2Long returned object IDs {output_ids}; expected {[object_id]}."
                        )
                    if len(output_logits) != frame_count:
                        raise RuntimeError(
                            f"SAM2Long returned {len(output_logits)} frames; "
                            f"expected {frame_count}."
                        )
                    per_object_logits.append(
                        [value.detach().float().cpu() for value in output_logits]
                    )
                finally:
                    self.predictor.reset_state(state)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        predictions = [initial.astype(np.int32, copy=True)]
        for frame_index in range(1, frame_count):
            logits = torch.cat([values[frame_index] for values in per_object_logits], dim=0)
            predictions.append(self._combine_masks(object_ids, logits, initial.shape))
        per_frame_ms = elapsed_ms / max(1, frame_count - 1)
        torch.cuda.empty_cache()
        return predictions, [0.0, *([per_frame_ms] * (frame_count - 1))]
