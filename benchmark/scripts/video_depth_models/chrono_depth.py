"""ChronoDepth — sliding-window video-depth via diffusion (Shao et al.).

Real adapter wrapping the official ``jhshao/ChronoDepth`` HF release.
Verified by HF model-card lookup (huggingface.co/jhshao/ChronoDepth):
loads via ``diffusers.DiffusionPipeline.from_pretrained``. Paper:
Shao et al. "Learning Temporally Consistent Video Depth from Video
Diffusion Priors" (2024).

**Verification status**: model_id and load_path verified. The pipeline
forward signature is best-effort; team should smoke on one clip.

Install
-------

::

    pip install diffusers transformers accelerate
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample

from ._video_adapter_base import VideoDepthAdapterBase


class ChronoDepthAdapter(VideoDepthAdapterBase):
    """Wraps ``jhshao/ChronoDepth`` HF release."""

    DISPLAY_NAME = "ChronoDepth"
    OUTPUT_KIND = "relative"  # Affine-invariant per the paper.

    def __init__(self, device: str = "cuda", *, num_inference_steps: int = 10) -> None:
        super().__init__(device=device)
        self.num_inference_steps = int(num_inference_steps)
        self._pipe = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "ChronoDepthAdapter needs `diffusers` and `torch`. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        # Verified model_id from HF model card.
        self._pipe = DiffusionPipeline.from_pretrained(
            "jhshao/ChronoDepth",
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

        result = self._pipe(
            pil_frames,
            num_inference_steps=self.num_inference_steps,
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
    return ChronoDepthAdapter(device=device, **kwargs)


__all__ = ["ChronoDepthAdapter", "build"]
