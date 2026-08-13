"""SAM 2 video-object-segmentation adapter for RPX D3."""

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

SAM2_MODEL_ID = "facebook/sam2-hiera-large"
SAM2_MODEL_REVISION = "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251"
SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"


class SAM2Tracker:
    """Run official SAM 2 from ground-truth masks on the first frame."""

    model_name = "sam2"
    model_id = SAM2_MODEL_ID
    model_revision = SAM2_MODEL_REVISION
    checkpoint_filename = SAM2_CHECKPOINT
    config_name = SAM2_CONFIG
    adapter_label = "SAM 2"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError(
                f"The production {self.adapter_label} adapter requires device='cuda'."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable to the {self.adapter_label} adapter.")

        from sam2.build_sam import build_sam2_video_predictor

        checkpoint = hf_hub_download(
            repo_id=self.model_id,
            filename=self.checkpoint_filename,
            revision=self.model_revision,
            cache_dir=os.environ.get("HF_HOME"),
        )
        self.checkpoint_path = str(Path(checkpoint).resolve())
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        self.predictor = build_sam2_video_predictor(
            self.config_name,
            checkpoint,
            device=device,
            apply_postprocessing=True,
        )
        self.predictor.non_overlap_masks = True
        self.device = device
        self.parameter_count = int(
            sum(parameter.numel() for parameter in self.predictor.parameters())
        )

    @staticmethod
    def _autocast_context():
        # Meta's official SAM 2 examples use BF16 on CUDA.
        return torch.autocast("cuda", dtype=torch.bfloat16)

    @staticmethod
    def _combine_masks(
        object_ids: Sequence[int],
        mask_logits: torch.Tensor,
        shape: tuple[int, int],
    ) -> np.ndarray:
        ids = np.asarray([int(value) for value in object_ids], dtype=np.int32)
        logits = mask_logits[:, 0].detach().float().cpu().numpy()
        if logits.shape[1:] != shape:
            raise RuntimeError(
                f"SAM 2 returned mask shape {logits.shape[1:]}; expected {shape}."
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
        """Return one instance-ID mask and timed propagation latency per frame."""

        initial = np.asarray(first_frame_mask)
        if initial.ndim != 2 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("first_frame_mask must be a 2-D integer instance mask.")
        object_ids = [int(value) for value in np.unique(initial) if value > 0]
        if not object_ids:
            raise ValueError("first_frame_mask contains no positive object IDs.")

        state = None
        predictions: list[np.ndarray | None] = [None] * frame_count
        latencies_ms: list[float] = [0.0] * frame_count
        try:
            with torch.inference_mode(), self._autocast_context():
                state = self.predictor.init_state(
                    video_path=str(video_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=True,
                )
                for object_id in object_ids:
                    self.predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        mask=initial == object_id,
                    )

                # The initialization frame is supplied by the benchmark, so
                # preserve it exactly instead of scoring SAM's reconstruction.
                predictions[0] = initial.astype(np.int32, copy=True)
                torch.cuda.synchronize()
                iterator = self.predictor.propagate_in_video(state)
                while True:
                    start = time.perf_counter()
                    try:
                        frame_index, output_ids, mask_logits = next(iterator)
                    except StopIteration:
                        break
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if frame_index == 0:
                        continue
                    predictions[frame_index] = self._combine_masks(
                        output_ids,
                        mask_logits,
                        initial.shape,
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
                f"{self.adapter_label} did not return predictions for frames: {missing[:10]}"
            )
        return [mask for mask in predictions if mask is not None], latencies_ms
