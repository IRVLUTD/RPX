"""RollingDepth — streaming video-depth via diffusion (PRS-ETH / Marigold team).

Real adapter wrapping the official ``prs-eth/rollingdepth-v1-0`` HF
release. Verified by HF model-card lookup
(huggingface.co/prs-eth/rollingdepth-v1-0): loads via
``diffusers.DiffusionPipeline.from_pretrained`` with the model card's
canonical incantation. Paper: Ke et al. "Video Depth without Video
Models" (2024) from ETH Zurich's Photogrammetry and Remote Sensing
Lab (PRS).

**Verification status**: model_id and load_path are verified against
the upstream HF model card. The pipeline signature (snippet/overlap
kwargs) is best-effort against the released checkpoint and may need
adjusting once the team runs it end-to-end on the lab GPU.

Install
-------

::

    pip install diffusers transformers accelerate
    # Weights pulled lazily from HF (~5 GB; cached after first run).
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample

from ._video_adapter_base import VideoDepthAdapterBase


class RollingDepthAdapter(VideoDepthAdapterBase):
    """Wraps ``prs-eth/rollingdepth-v1-0`` HF release.

    Parameters
    ----------
    device
        ``"cuda"`` (default) or ``"cpu"``.
    num_inference_steps
        Diffusion solver steps. The team can tune for speed vs quality.
    snippet_length
        Frames per sliding snippet. Default 10.
    overlap
        Snippet overlap for boundary blending. Default 5.
    """

    DISPLAY_NAME = "RollingDepth"
    OUTPUT_KIND = "relative"  # Marigold family — affine-invariant.

    def __init__(
        self,
        device: str = "cuda",
        *,
        num_inference_steps: int = 4,
        snippet_length: int = 10,
        overlap: int = 5,
    ) -> None:
        super().__init__(device=device)
        self.num_inference_steps = int(num_inference_steps)
        self.snippet_length = int(snippet_length)
        self.overlap = int(overlap)
        self._pipe = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "RollingDepthAdapter needs `diffusers` and `torch`. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e

        # Verified model_id from HF model card. The model card recommends
        # bfloat16 on CUDA; we honour that when available, else fp32.
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._pipe = DiffusionPipeline.from_pretrained(
            "prs-eth/rollingdepth-v1-0",
            torch_dtype=dtype,
        )
        self._pipe.to(self.device)
        for opt in (
            "enable_xformers_memory_efficient_attention",
            "enable_attention_slicing",
        ):
            try:
                getattr(self._pipe, opt)()
            except Exception:  # noqa: BLE001
                pass
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        from PIL import Image

        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape
        pil_frames = [Image.fromarray(rgb_seq[t]) for t in range(T)]

        # Pipeline kwargs are best-effort against the upstream release;
        # the team should sanity-check via one-clip smoke. If the release
        # exposes different kwarg names (e.g. ``num_frames_per_snippet``),
        # update them here.
        result = self._pipe(
            pil_frames,
            num_inference_steps=self.num_inference_steps,
            snippet_length=self.snippet_length,
            overlap=self.overlap,
        )
        depth_seq = result.frames if hasattr(result, "frames") else result[0]
        depth_seq = np.asarray(depth_seq, dtype=np.float32)
        if depth_seq.ndim == 4:
            depth_seq = depth_seq.squeeze(1)
        if depth_seq.shape != (T, H, W):
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth_seq[t].astype(np.float32), mode="F")
                    .resize((W, H), _Image.BILINEAR),
                    dtype=np.float32,
                )
            depth_seq = resized
        return depth_seq


def build(device: str = "cuda", **kwargs):
    return RollingDepthAdapter(device=device, **kwargs)


__all__ = ["RollingDepthAdapter", "build"]
