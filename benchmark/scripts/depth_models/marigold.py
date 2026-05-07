"""Marigold adapter — diffusion-based monocular depth (ETH).

Two checkpoints share this adapter via ``model_id``:

* ``prs-eth/marigold-depth-v1-1`` — full diffusion, default
  ``ensemble_size=10`` for the published-best number; we run at
  ``ensemble_size=1`` for the raw / fast headline.
* ``prs-eth/marigold-depth-lcm-v1-0`` — latent-consistency-distilled
  fast variant, single-step inference (no ensemble).

Marigold output is **affine-invariant disparity** in [0, 1]. We
declare ``native_alignment="ls_affine"`` so the runner fits a
per-frame ``a·pred + b → GT`` before metric computation.

Tracker reference: MonocularMetricDepth — relative head.

Install
-------
    pip install diffusers accelerate torch pillow

Usage
-----
    from depth_models.marigold import Marigold
    import rpx_benchmark as rpx

    m = Marigold(device="cuda", batch_size=1, ensemble_size=1)
    model = rpx.make_numpy_depth_model(m, name="Marigold-v1.1")
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class Marigold:
    """Diffusion depth: rgb (H×W×3 uint8) → disparity (H×W float32, [0,1]).

    Per-frame ls_affine alignment in the runner converts to GT scale.
    """

    DEFAULT_MODEL_ID = "prs-eth/marigold-depth-v1-1"

    #: Up-to-scale (affine-invariant disparity); needs least-squares-affine
    #: alignment to GT before any metric is comparable to metric models.
    native_alignment: str = "ls_affine"

    #: Marigold + Marigold-LCM both run cleanly in fp16.
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        ensemble_size: int = 1,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from diffusers import MarigoldDepthPipeline
        except ImportError as e:
            raise ImportError(
                "Marigold needs `diffusers`. Install with: "
                "pip install diffusers accelerate torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.ensemble_size = int(ensemble_size)
        self._torch = torch

        torch_dtype = (getattr(torch, dtype) if isinstance(dtype, str)
                       else (torch.float16 if self.native_precision == "fp16"
                             else torch.float32))
        self._pipe = MarigoldDepthPipeline.from_pretrained(
            model_id, variant="fp16" if torch_dtype == torch.float16 else None,
            torch_dtype=torch_dtype,
        ).to(device)
        # Slim down: don't generate the visualisation map every call.
        self._pipe.set_progress_bar_config(disable=True)

    @property
    def torch_module(self):
        # Diffusers pipelines expose unet + vae + text_encoder; the unet is
        # the dominant compute and the right one to profile.
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
                    hint="Adapter contract: each input must be a (H, W, 3) "
                         "uint8 numpy array.",
                )

        pil_imgs = [Image.fromarray(np.asarray(r, dtype=np.uint8)) for r in rgbs]
        # Marigold's pipeline accepts a list of images; ensemble_size>1 means
        # multiple denoising passes per image and is memory-intensive.
        with self._torch.inference_mode():
            out = self._pipe(
                pil_imgs,
                ensemble_size=self.ensemble_size,
                num_inference_steps=4 if "lcm" in self.model_id else 50,
                output_type="np",
            )
        # ``out.prediction`` is shape [B, H, W, 1] in [0, 1].
        preds = np.asarray(out.prediction, dtype=np.float32)
        if preds.ndim == 4 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)

        depths: list[np.ndarray] = []
        for r, d in zip(rgbs, preds):
            target_hw = np.asarray(r).shape[:2]
            if d.shape != target_hw:
                d = _resize_bilinear(d, target_hw)
            depths.append(d.astype(np.float32))
        return depths if is_batch else depths[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
