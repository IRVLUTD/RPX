"""ChronoDepth — sliding-window video-depth via diffusion (Shao et al.).

Real adapter wrapping the official ``jhshao/ChronoDepth-v1`` UNet and
Stable Video Diffusion base model. Paper:
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

    def __init__(self, device: str = "cuda", *, num_inference_steps: int = 5) -> None:
        super().__init__(device=device)
        self.num_inference_steps = int(num_inference_steps)
        self._pipe = None

    def setup(self) -> None:
        """Load via the upstream ``chronodepth`` package.

        The bare ``DiffusionPipeline.from_pretrained`` path fails with
        ``module diffusers has no attribute ChronoDepthPipeline`` —
        ChronoDepth's config declares a custom pipeline class. The
        upstream github repo provides it.
        """
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ChronoDepthAdapter needs `torch`. Install with: pip install torch"
            ) from e

        try:
            from chronodepth import ChronoDepthPipeline
            from chronodepth.unet_chronodepth import (
                DiffusersUNetSpatioTemporalConditionModelChronodepth,
            )
        except ImportError as e:
            raise ImportError(
                "ChronoDepthAdapter needs the upstream `chronodepth` "
                "package. Install with:\n"
                "    git clone https://github.com/jiahao-shao1/ChronoDepth\n"
                "    add that checkout to PYTHONPATH"
            ) from e

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        unet = DiffusersUNetSpatioTemporalConditionModelChronodepth.from_pretrained(
            "jhshao/ChronoDepth-v1",
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        self._pipe = ChronoDepthPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=dtype,
            variant="fp16" if self.device.startswith("cuda") else None,
        )
        self._pipe.n_tokens = 10
        self._pipe.chunk_size = 5
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
        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape
        # The official pipeline requires spatial dimensions divisible by 64.
        # Resize only for model input; the prediction is restored below.
        proc_h = max(64, round(H / 64) * 64)
        proc_w = max(64, round(W / 64) * 64)
        if (proc_h, proc_w) != (H, W):
            from PIL import Image

            rgb_input = np.stack(
                [
                    np.asarray(Image.fromarray(frame).resize((proc_w, proc_h), Image.BILINEAR))
                    for frame in rgb_seq
                ]
            )
        else:
            rgb_input = rgb_seq

        result = self._pipe(
            rgb_input.astype(np.float32) / 255.0,
            height=proc_h,
            width=proc_w,
            num_inference_steps=self.num_inference_steps,
            decode_chunk_size=min(8, T),
            noise_aug_strength=0.0,
            infer_mode="ours",
            sigma_epsilon=-4.0,
            show_progress_bar=False,
        )
        depth_seq = result.frames if hasattr(result, "frames") else result[0]
        depth_seq = np.asarray(depth_seq, dtype=np.float32)
        while depth_seq.ndim > 3:
            axis = next((i for i, n in enumerate(depth_seq.shape) if n == 1), None)
            if axis is None:
                break
            depth_seq = np.squeeze(depth_seq, axis=axis)
        if depth_seq.shape != (T, H, W):
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth_seq[t].astype(np.float32), mode="F").resize(
                        (W, H), _Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
            depth_seq = resized
        return depth_seq


def build(device: str = "cuda", **kwargs):
    return ChronoDepthAdapter(device=device, **kwargs)


__all__ = ["ChronoDepthAdapter", "build"]
