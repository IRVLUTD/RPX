"""Official SAM 2++ unified-prompt adapter for the RPX D3 protocol."""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download

SAM2_PLUS_MODEL_ID = "MCG-NJU/SAM2-Plus"
SAM2_PLUS_MODEL_REVISION = "c3c534e30469d8788123287a484488567c5115d4"
SAM2_PLUS_CHECKPOINT = "checkpoint_phase123.pt"
SAM2_PLUS_CONFIG = "sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml"
SAM2_PLUS_CONFIG_DIR = "/opt/rpx-models/sam2_plus/sam2_plus/configs/sam2.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SAM2PlusTracker:
    """Run official SAM 2++ with first-frame RPX instance-mask prompts."""

    model_name = "sam2-plus"
    model_id = SAM2_PLUS_MODEL_ID
    model_revision = SAM2_PLUS_MODEL_REVISION
    checkpoint_filename = SAM2_PLUS_CHECKPOINT
    config_name = SAM2_PLUS_CONFIG
    config_directory = SAM2_PLUS_CONFIG_DIR
    adapter_label = "SAM 2++"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production SAM 2++ adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the SAM 2++ adapter.")

        config_dir = Path(os.environ.get("SAM2_PLUS_CONFIG_DIR", self.config_directory)).resolve()
        config_path = config_dir / self.config_name
        if not config_path.is_file():
            raise RuntimeError(
                f"SAM 2++ config is missing: {config_path}. "
                "Use the pinned cumulative SAM 2++ Docker image."
            )

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2_plus.build_sam import build_sam2_video_predictor_plus

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
        self.predictor = build_sam2_video_predictor_plus(
            self.config_name,
            str(checkpoint),
            device=device,
            apply_postprocessing=True,
            task="mask",
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
        object_ids: Sequence[int], mask_logits: torch.Tensor, shape: tuple[int, int]
    ) -> np.ndarray:
        ids = np.asarray([int(value) for value in object_ids], dtype=np.int32)
        logits = mask_logits.detach().float().cpu().numpy()
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.shape != (len(ids), *shape):
            raise RuntimeError(
                f"SAM 2++ returned mask shape {logits.shape}; "
                f"expected {(len(ids), *shape)}."
            )
        best_index = logits.argmax(axis=0)
        best_score = logits.max(axis=0)
        output = np.zeros(shape, dtype=np.int32)
        foreground = best_score > 0.0
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

        state = None
        predictions: list[np.ndarray | None] = [None] * frame_count
        latencies_ms = [0.0] * frame_count
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.predictor.init_state(
                    video_path=str(video_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )
                for object_id in object_ids:
                    self.predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        mask=initial == object_id,
                    )

                predictions[0] = initial.astype(np.int32, copy=True)
                torch.cuda.synchronize()
                iterator = self.predictor.propagate_in_video(state)
                while True:
                    started = time.perf_counter()
                    try:
                        frame_index, output_ids, mask_logits, _, _ = next(iterator)
                    except StopIteration:
                        break
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if frame_index == 0:
                        continue
                    predictions[frame_index] = self._combine_masks(
                        output_ids, mask_logits, initial.shape
                    )
                    latencies_ms[frame_index] = elapsed_ms
        finally:
            if state is not None:
                with contextlib.suppress(Exception):
                    self.predictor.reset_state(state)
            torch.cuda.empty_cache()

        missing = [index for index, mask in enumerate(predictions) if mask is None]
        if missing:
            raise RuntimeError(
                f"SAM 2++ did not return predictions for frames: {missing[:10]}"
            )
        return [mask for mask in predictions if mask is not None], latencies_ms
