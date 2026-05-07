"""Lotus / Lotus-2 adapter — diffusion-based monocular depth.

Loads ``jingheya/Lotus-2`` (per the team's tracker) via diffusers'
generic auto-pipeline. Lotus repurposes Stable Diffusion's denoising
prior for depth by treating depth as a single-step "noise → depth"
prediction, so default inference is just 1 step (orders of magnitude
faster than Marigold's 50-step diffusion).

Output is **affine-invariant disparity / depth** in [0, 1]. Declares
``native_alignment="ls_affine"`` so the runner aligns per-frame.

Tracker reference: MonocularMetricDepth — relative head.

Install
-------
    pip install diffusers accelerate torch pillow

Notes
-----
The Lotus checkpoints have shipped under several names over the project's
lifetime (``jingheya/Lotus-2``, ``jingheya/lotus-depth-d-v2-0``,
``jingheya/lotus-depth-g-v2-1-disparity``). We pin to ``jingheya/Lotus-2``
per Feynman's verified-2026 list. If the auto-pipeline can't load it
(checkpoint format change), the adapter raises an actionable error.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class Lotus:
    """Diffusion depth: rgb → disparity (H×W float32, ls_affine-aligned)."""

    DEFAULT_MODEL_ID = "jingheya/Lotus-2"

    native_alignment: str = "ls_affine"
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        num_inference_steps: int = 1,        # Lotus is single-step by design
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image, DiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "Lotus needs `diffusers`. Install with: "
                "pip install diffusers accelerate torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.num_inference_steps = int(num_inference_steps)
        self._torch = torch

        torch_dtype = (getattr(torch, dtype) if isinstance(dtype, str)
                       else (torch.float16 if self.native_precision == "fp16"
                             else torch.float32))
        # Use the generic DiffusionPipeline so checkpoint-specific config
        # (custom_pipeline / scheduler) is honoured automatically.
        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, trust_remote_code=True,
            ).to(device)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError
            raise AdapterError(
                f"Lotus pipeline load failed for {model_id!r}: {e}",
                hint="The Lotus checkpoint name has changed across releases; "
                     "verify the current id at https://huggingface.co/jingheya "
                     "and pass `model_id=...` explicitly.",
            ) from e
        if hasattr(self._pipe, "set_progress_bar_config"):
            self._pipe.set_progress_bar_config(disable=True)

    @property
    def torch_module(self):
        return getattr(self._pipe, "unet", None)

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        from PIL import Image

        is_batch = isinstance(rgb, (list, tuple))
        rgbs = list(rgb) if is_batch else [rgb]
        for r in rgbs:
            r = np.asarray(r)
            if r.ndim != 3 or r.shape[2] != 3:
                from rpx_benchmark.exceptions import AdapterError
                raise AdapterError(
                    f"expected H×W×3 RGB uint8, got shape {r.shape}",
                )

        pil_imgs = [Image.fromarray(np.asarray(r, dtype=np.uint8)) for r in rgbs]
        depths: list[np.ndarray] = []
        with self._torch.inference_mode():
            # Lotus pipelines vary in their public call signature. The
            # consistent path: one image at a time + result.images[0] as
            # a PIL Image whose pixel values encode depth.
            for r, img in zip(rgbs, pil_imgs):
                out = self._pipe(img, num_inference_steps=self.num_inference_steps)
                d = self._extract_depth(out, r.shape[:2])
                depths.append(d.astype(np.float32))
        return depths if is_batch else depths[0]

    @staticmethod
    def _extract_depth(out, target_hw):
        """Pull a (H, W) float depth out of whatever the Lotus pipeline returned.

        Different Lotus snapshots return different shapes:
        ``out.images[0]`` (PIL), ``out.prediction`` (numpy), ``out["depth"]``
        (tensor). Try them in order.
        """
        from PIL import Image
        # Tensor / numpy paths
        for attr in ("prediction", "depth"):
            v = getattr(out, attr, None)
            if v is None and hasattr(out, "__getitem__"):
                try:
                    v = out[attr]
                except Exception:
                    v = None
            if v is not None:
                arr = np.asarray(v if not hasattr(v, "cpu") else v.cpu().numpy(),
                                 dtype=np.float32)
                if arr.ndim == 4 and arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = arr[0]
                if arr.shape != tuple(target_hw):
                    arr = _resize_bilinear(arr, target_hw)
                return arr
        # PIL path — convert intensity to float
        imgs = getattr(out, "images", None)
        if imgs:
            d = np.asarray(imgs[0].convert("F"), dtype=np.float32)
            if d.shape != tuple(target_hw):
                d = _resize_bilinear(d, target_hw)
            return d
        from rpx_benchmark.exceptions import AdapterError
        raise AdapterError(
            "Lotus pipeline returned an unrecognised shape — couldn't "
            "extract depth.",
            hint="Inspect the pipeline's return type; expected one of "
                 "`images[0]` (PIL), `prediction` (tensor/np), `depth` (tensor).",
        )


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
