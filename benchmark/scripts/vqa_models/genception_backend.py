"""GenCeption RefVOS backend for RPX regular and in-context grounding VQA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from genception_runtime import (
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    GenCeptionRuntime,
    runtime_from_environment,
)
from PIL import Image


@dataclass(frozen=True)
class GenCeptionCheckpoint:
    repo_id: str
    revision: str
    variant: str
    backend: str = "jax-genception"


CHECKPOINTS = {
    "genception-1.3b": GenCeptionCheckpoint(
        UPSTREAM_REPOSITORY, UPSTREAM_REVISION, "1.3b"
    ),
    "genception-14b": GenCeptionCheckpoint(
        UPSTREAM_REPOSITORY, UPSTREAM_REVISION, "14b"
    ),
}


class GenCeptionVQARunner:
    """Convert native RefVOS masks to the benchmark's normalized bbox JSON."""

    def __init__(
        self,
        model_key: str,
        image_root: str | Path,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 1,
        *,
        runtime: GenCeptionRuntime | None = None,
    ) -> None:
        del image_root, gpu_memory_utilization
        if max_num_seqs != 1:
            raise ValueError("GenCeption's released pipeline supports max_num_seqs=1")
        self.model_key = model_key
        self.checkpoint = CHECKPOINTS[model_key]
        self.max_batch_size = 1
        self.runtime = runtime or runtime_from_environment(self.checkpoint.variant)
        self.runtime.setup()
        self._last_adapter_metadata: dict[str, Any] = {}
        self._last_batch_adapter_metadata: list[dict[str, Any]] = []

    @staticmethod
    def _load_video(paths: Sequence[str | Path]) -> np.ndarray:
        if len(paths) not in {1, 2}:
            raise ValueError("RPX VQA rows must contain one or two images")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        target = images[-1]
        if len(images) == 1:
            return target[None]
        reference = np.asarray(
            Image.fromarray(images[0]).resize(
                (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
        # Preserve a long reference phase followed by a long target phase when
        # the official preprocessor maps the clip to its fixed 81-frame grid.
        return np.concatenate(
            [np.repeat(reference[None], 40, axis=0), np.repeat(target[None], 41, axis=0)]
        )

    def _predict_one(
        self,
        image_paths: Sequence[str | Path],
        prompt: str,
        output_kind: str,
    ) -> tuple[str, dict[str, Any]]:
        if output_kind != "bbox_native_genception_refvos":
            raise ValueError(f"unsupported GenCeption output kind: {output_kind}")
        video = self._load_video(image_paths)
        scores = self.runtime.predict_mask_scores(video, prompt)
        mask = scores[-1] >= self.runtime.segmentation_threshold
        ys, xs = np.where(mask)
        metadata = {
            "adapter": "official_genception_refvos_bbox",
            "variant": self.checkpoint.variant,
            "source_revision": UPSTREAM_REVISION,
            "image_count": len(image_paths),
            "image_order": "reference_then_target" if len(image_paths) == 2 else "target_only",
            "single_model_call": True,
            "single_scored_model_call": True,
            "referring_expression": prompt,
            "threshold": self.runtime.segmentation_threshold,
        }
        if xs.size == 0:
            return json.dumps({"error": "empty RefVOS mask"}), metadata
        height, width = mask.shape
        bbox = [
            float(xs.min()) * 1000.0 / max(width - 1, 1),
            float(ys.min()) * 1000.0 / max(height - 1, 1),
            float(xs.max()) * 1000.0 / max(width - 1, 1),
            float(ys.max()) * 1000.0 / max(height - 1, 1),
        ]
        return json.dumps({"label": "object", "bbox": bbox}, separators=(",", ":")), metadata

    def predict(
        self,
        image_paths: str | Path | Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        del max_tokens
        paths = [image_paths] if isinstance(image_paths, (str, Path)) else list(image_paths)
        value, self._last_adapter_metadata = self._predict_one(paths, prompt, output_kind)
        return value

    def predict_batch(
        self, requests: Sequence[tuple[Sequence[str | Path], str, int, str]]
    ) -> list[str]:
        values = []
        metadata = []
        for paths, prompt, _max_tokens, output_kind in requests:
            value, row_metadata = self._predict_one(paths, prompt, output_kind)
            values.append(value)
            metadata.append(row_metadata)
        self._last_batch_adapter_metadata = metadata
        return values

    def prediction_metadata(self) -> dict[str, Any]:
        return dict(self._last_adapter_metadata)

    def batch_prediction_metadata(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._last_batch_adapter_metadata]
