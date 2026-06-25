"""Shared base for true-video depth adapters.

Every D1-V adapter in this package follows the same six-step contract:

1. Subclass :class:`BenchmarkModel`, set ``task = TaskType.VIDEO_DEPTH``.
2. Set ``depth_output_kind`` — ``"metric"`` if the model emits metres
   directly, ``"relative"`` if affine-invariant (runner applies per-clip
   ``(s, t)`` alignment before metrics).
3. Set ``name`` to the canonical roster display name.
4. Override ``setup()`` to lazy-load weights. Idempotent on repeat
   calls (return early when ``self._loaded``).
5. Override ``predict(batch)`` — for each :class:`VideoSample` in the
   batch, return one :class:`VideoDepthPrediction` whose
   ``depth_map_seq`` is ``(T, H, W) float32`` in metres (or up-to-scale
   for relative models — the runner aligns before scoring).
6. Expose a module-level ``build(device, **kwargs) -> BenchmarkModel``
   factory the team CLI discovers.

This base class implements (1) and (3) generically; subclasses only
need (2), (4), (5), and the factory.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rpx_benchmark.api import (
    BenchmarkModel,
    TaskType,
    VideoDepthPrediction,
    VideoSample,
)
from rpx_benchmark.exceptions import AdapterError


class VideoDepthAdapterBase(BenchmarkModel):
    """Base class enforcing the D1-V adapter contract.

    Concrete subclasses set ``DISPLAY_NAME`` and ``OUTPUT_KIND`` class
    attributes, then override ``setup`` and ``_predict_clip`` (the
    per-clip forward pass). :meth:`predict` handles the batch loop and
    enforces the output shape so every adapter behaves identically to
    the runner downstream.
    """

    task = TaskType.VIDEO_DEPTH

    #: Display name surfaced in result.json and the cell log.
    DISPLAY_NAME: str = ""

    #: ``"metric"`` (default) or ``"relative"``. Relative models go
    #: through per-clip ``(s, t)`` alignment before metrics.
    OUTPUT_KIND: str = "metric"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ != "VideoDepthAdapterBase":
            if not cls.DISPLAY_NAME:
                raise TypeError(
                    f"{cls.__name__}: DISPLAY_NAME must be set on every "
                    "VideoDepthAdapterBase subclass."
                )
            cls.name = cls.DISPLAY_NAME
            cls.depth_output_kind = cls.OUTPUT_KIND

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self._loaded = False
        self._kwargs = dict(kwargs)

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        """Per-clip forward pass. Override in subclasses.

        Must return ``(T, H, W) float32`` depth in metres (or
        up-to-scale for relative models). Shape is enforced by
        :meth:`predict` after the call returns.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._predict_clip must be implemented by the subclass."
        )

    def predict(
        self,
        batch: Sequence[VideoSample],
    ) -> list[VideoDepthPrediction]:
        if not self._loaded:
            raise RuntimeError(
                f"{type(self).__name__}.setup() must be called before predict()."
            )
        out: list[VideoDepthPrediction] = []
        for sample in batch:
            depth_seq = self._predict_clip(sample)
            depth_seq = np.asarray(depth_seq, dtype=np.float32)
            expected_thw = np.asarray(sample.rgb_seq).shape[:3]
            if depth_seq.shape != expected_thw:
                raise AdapterError(
                    f"{type(self).__name__}: predict returned depth_seq "
                    f"shape {depth_seq.shape}; expected (T, H, W) = {expected_thw}",
                    hint="Resize / squeeze the per-frame depth maps to "
                    "match the input RGB resolution before returning.",
                )
            out.append(VideoDepthPrediction(depth_map_seq=depth_seq))
        return out


__all__ = ["VideoDepthAdapterBase"]
