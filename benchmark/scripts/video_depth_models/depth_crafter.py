"""DepthCrafter — sliding-window video-depth via diffusion (Tencent).

Real adapter that wraps the official ``tencent/DepthCrafter`` HF
release. Serves two purposes:

1. Fills one of the canonical Video Depth roster entries with a true
   video model (not a frame-as-video baseline), so the paper Table 4
   has at least one row that genuinely exercises temporal context.
2. Acts as the **reference template** for the other 8 true-video
   adapters (RollingDepth, ChronoDepth, MonST3R, VGGT-Ω, etc.). The
   pattern is: subclass :class:`BenchmarkModel`, set
   ``task = TaskType.VIDEO_DEPTH`` and ``depth_output_kind``, override
   ``setup`` to lazy-load weights, override ``predict`` to consume the
   clip and emit a :class:`VideoDepthPrediction` per sample.

DepthCrafter outputs **affine-invariant** depth — the runner picks
this up via ``depth_output_kind = "relative"`` and applies per-clip
``(s, t)`` alignment before metrics.

Install
-------

::

    pip install diffusers transformers accelerate
    # Weights are pulled lazily from HF (tencent/DepthCrafter,
    # ~5 GB; cached after first run).

Usage
-----

::

    PYTHONPATH=. python scripts/run_video_depth.py \\
        --model depth-crafter --split easy
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


class DepthCrafterAdapter(BenchmarkModel):
    """Wraps the official ``tencent/DepthCrafter`` HF release.

    Parameters
    ----------
    device
        Torch device, ``"cuda"`` (default) or ``"cpu"``. CPU will work
        but inference will be impractically slow (~minutes per frame).
    num_inference_steps
        Diffusion solver steps. DepthCrafter's paper uses 25; smaller
        values trade quality for speed. Defaults to 25.
    guidance_scale
        Classifier-free guidance scale. Paper uses 1.2.
    window_size
        Per-window frame count. DepthCrafter processes the clip in
        sliding 110-frame windows by default; overlapping windows are
        blended for temporal consistency. Reduce to fit smaller VRAM.
    overlap
        Frames of overlap between consecutive windows. Paper uses 25.
    """

    task = TaskType.VIDEO_DEPTH
    name = "DepthCrafter"
    #: DepthCrafter emits affine-invariant (relative) depth; the runner
    #: applies per-clip (s, t) alignment automatically.
    depth_output_kind = "relative"

    def __init__(
        self,
        device: str = "cuda",
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.2,
        window_size: int = 110,
        overlap: int = 25,
    ) -> None:
        self.device = device
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.window_size = int(window_size)
        self.overlap = int(overlap)
        self._pipe = None

    def setup(self) -> None:
        """Load the DepthCrafter pipeline.

        **TODO (team)**: the exact load incantation depends on the
        upstream release the team installs — verify against the
        official DepthCrafter README before wiring this in. Once the
        load call is correct, replace the ``NotImplementedError``
        below with it. Everything else on this adapter
        (``depth_output_kind="relative"`` for per-clip alignment,
        ``predict`` returning a stacked ``(T, H, W) float32`` tensor,
        window/overlap kwargs) follows the standard D1-V contract and
        shouldn't need changes.
        """
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401  — verify torch is available
            import diffusers  # noqa: F401  — verify diffusers is available
        except ImportError as e:
            raise ImportError(
                "DepthCrafterAdapter needs `diffusers` and `torch`. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e

        raise NotImplementedError(
            "DepthCrafter setup() not yet wired — see the TODO in the "
            "docstring. Pick one of the two load paths and replace this "
            "raise with the real call. The other adapter machinery "
            "(depth_output_kind='relative', predict, window/overlap "
            "kwargs) is ready."
        )

    def predict(
        self,
        batch: Sequence[VideoSample],
    ) -> list[VideoDepthPrediction]:
        if self._pipe is None:
            raise RuntimeError(
                "DepthCrafterAdapter.setup() must be called before predict()."
            )
        out: list[VideoDepthPrediction] = []
        for sample in batch:
            rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)  # (T, H, W, 3)
            # DepthCrafter expects a list of PIL images or an (T, H, W, 3)
            # uint8 array in [0, 255]. The official pipeline returns an
            # (T, H, W) float tensor.
            result = self._pipe(
                rgb_seq,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                window_size=self.window_size,
                overlap=self.overlap,
            )
            # The pipeline returns a tuple-like object whose ``frames``
            # attribute holds the depth sequence. Some releases return
            # (depth_seq,) directly; handle both.
            depth_seq = (
                result.frames if hasattr(result, "frames") else result[0]
            )
            depth_seq = np.asarray(depth_seq, dtype=np.float32)
            if depth_seq.ndim == 4:
                # Some pipelines emit (T, 1, H, W); squeeze the channel.
                depth_seq = depth_seq.squeeze(1)
            if depth_seq.shape != rgb_seq.shape[:3]:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"DepthCrafter returned depth_seq shape {depth_seq.shape}; "
                    f"expected (T, H, W) = {rgb_seq.shape[:3]}",
                )
            out.append(VideoDepthPrediction(depth_map_seq=depth_seq))
        return out


def build(device: str = "cuda", **kwargs) -> BenchmarkModel:
    """Factory the runner discovers via ``--model depth-crafter``."""
    return DepthCrafterAdapter(device=device, **kwargs)


__all__ = ["DepthCrafterAdapter", "build"]
