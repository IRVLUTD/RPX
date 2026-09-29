"""GenCeption image-depth adapter using the official video checkpoint."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from genception_runtime import GenCeptionRuntime, runtime_from_environment


class GenCeptionImageDepth:
    native_alignment = "ls_affine"
    native_precision = "bf16"

    def __init__(
        self,
        variant: str,
        device: str = "cuda",
        batch_size: int = 1,
        *,
        runtime: GenCeptionRuntime | None = None,
        **_: object,
    ) -> None:
        del device
        if batch_size != 1:
            raise ValueError("GenCeption's official pipeline supports batch_size=1")
        self.variant = variant
        self.model_id = f"google-deepmind/GenCeption-{variant}"
        self.batch_size = 1
        self.runtime = runtime or runtime_from_environment(variant)

    def _one(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected H x W x 3 RGB image, got {image.shape}")
        # A still frame is a one-frame video under the released pipeline.
        return self.runtime.predict_depth(image[None])[0]

    def __call__(
        self, rgb: np.ndarray | Sequence[np.ndarray]
    ) -> np.ndarray | list[np.ndarray]:
        if isinstance(rgb, (list, tuple)):
            return [self._one(image) for image in rgb]
        return self._one(rgb)


def build_1_3b(**kwargs: object) -> GenCeptionImageDepth:
    return GenCeptionImageDepth("1.3b", **kwargs)


def build_14b(**kwargs: object) -> GenCeptionImageDepth:
    return GenCeptionImageDepth("14b", **kwargs)
