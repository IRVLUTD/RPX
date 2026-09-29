"""Text-initialized GenCeption RefVOS adapters for RPX tracking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from genception_runtime import (
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    GenCeptionRuntime,
    load_rgb_video,
    runtime_from_environment,
)


class GenCeptionTracker:
    model_id = UPSTREAM_REPOSITORY
    source_revision = UPSTREAM_REVISION
    prompt_type = "text"
    tracking_mode = "native-refvos-text-initialized"
    adapter_label = "GenCeption RefVOS"
    variant = ""

    def __init__(
        self,
        device: str = "cuda",
        *,
        runtime: GenCeptionRuntime | None = None,
    ) -> None:
        del device
        self.model_name = f"genception-{self.variant}"
        self.model_revision = UPSTREAM_REVISION
        self.runtime = runtime or runtime_from_environment(self.variant)
        self.runtime.setup()
        self._metadata: dict[str, Any] = {}

    def prediction_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def track(
        self,
        video_dir: Path,
        frame_shape: tuple[int, int],
        frame_count: int,
        text_prompts: Sequence[Any],
    ) -> tuple[list[np.ndarray], list[float]]:
        paths = sorted(video_dir.glob("*.jpg"))
        if len(paths) != frame_count:
            raise RuntimeError(f"GenCeption received {len(paths)} frames; expected {frame_count}")
        video = load_rgb_video(paths)
        if tuple(video.shape[1:3]) != tuple(frame_shape):
            raise RuntimeError(
                f"staged RGB shape {video.shape[1:3]} does not match {frame_shape}"
            )
        labels = np.zeros(video.shape[:3], dtype=np.int32)
        confidence = np.full(video.shape[:3], -np.inf, dtype=np.float32)
        elapsed_ms = 0.0
        records = []
        for prompt in text_prompts:
            started = time.perf_counter()
            scores = self.runtime.predict_mask_scores(video, prompt.prompt_text)
            elapsed_ms += (time.perf_counter() - started) * 1000.0
            foreground = scores >= self.runtime.segmentation_threshold
            replace = foreground & (scores > confidence)
            labels[replace] = int(prompt.mask_index)
            confidence[replace] = scores[replace]
            records.append(
                {
                    "mask_index": int(prompt.mask_index),
                    "prompt_text": str(prompt.prompt_text),
                    "source_catalog_id": str(prompt.source_catalog_id),
                    "object_id": str(prompt.object_id),
                }
            )
        per_frame_ms = elapsed_ms / frame_count
        self._metadata = {
            "adapter": "official_genception_refvos",
            "variant": self.variant,
            "source_revision": UPSTREAM_REVISION,
            "prompt_count": len(records),
            "prompts": records,
            "overlap_policy": "highest_reconstructed_foreground_score",
            "fixed_model_frames": 81,
        }
        return [frame for frame in labels], [per_frame_ms] * frame_count


class GenCeption13BTracker(GenCeptionTracker):
    variant = "1.3b"


class GenCeption14BTracker(GenCeptionTracker):
    variant = "14b"
