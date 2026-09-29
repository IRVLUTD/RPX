"""Shared official GenCeption adapter for RPX video depth."""

from __future__ import annotations

import numpy as np
from genception_runtime import GenCeptionRuntime, runtime_from_environment

from ._video_adapter_base import VideoDepthAdapterBase


class GenCeptionVideoDepth(VideoDepthAdapterBase):
    DISPLAY_NAME = "GenCeption"
    OUTPUT_KIND = "relative"

    def __init__(
        self,
        device: str = "cuda",
        *,
        variant: str,
        runtime: GenCeptionRuntime | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, **kwargs)
        self.variant = variant
        self.runtime = runtime or runtime_from_environment(variant)
        self.name = f"GenCeption {variant.upper()}"

    def setup(self) -> None:
        if not self._loaded:
            self.runtime.setup()
            self._loaded = True

    def _predict_clip(self, sample) -> np.ndarray:
        return self.runtime.predict_depth(np.asarray(sample.rgb_seq))
