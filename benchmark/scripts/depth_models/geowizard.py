"""GeoWizard adapter — diffusion-based joint depth + normal (ECCV 2024).

Loads ``lemonaddie/geowizard`` via the project's HF-hosted diffusers
pipeline. We only consume the depth head; the normal head is dropped
(GeoWizard is included in the depth-zoo for the depth axis only).

Output is **affine-invariant depth**. Declares
``native_alignment="ls_affine"``.

Tracker reference: MonocularMetricDepth — relative head.

Install
-------
    pip install diffusers accelerate torch pillow
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class GeoWizard:
    """Diffusion depth: rgb → depth (ls_affine-aligned)."""

    DEFAULT_MODEL_ID = "lemonaddie/geowizard"

    native_alignment: str = "ls_affine"
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        ensemble_size: int = 1,
        num_inference_steps: int = 10,
        domain: str = "indoor",  # "indoor" | "outdoor" | "object"
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "GeoWizard needs `diffusers`. Install with: "
                "pip install diffusers accelerate torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.ensemble_size = int(ensemble_size)
        self.num_inference_steps = int(num_inference_steps)
        self.domain = domain
        self._torch = torch

        torch_dtype = (
            getattr(torch, dtype)
            if isinstance(dtype, str)
            else (torch.float16 if self.native_precision == "fp16" else torch.float32)
        )
        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                custom_pipeline=model_id,
            ).to(device)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"GeoWizard pipeline load failed for {model_id!r}: {e}",
                hint=(
                    "`lemonaddie/geowizard` is an HF *Space*, not a model "
                    "repo loadable via DiffusionPipeline. To run GeoWizard "
                    "you need to clone the upstream + run their inference "
                    "script:\n"
                    "  git clone https://github.com/fuxiao0719/GeoWizard\n"
                    "  cd GeoWizard && pip install -r requirements.txt\n"
                    "  # weights download is handled by `run_infer.py` / `run_infer_v2.py`\n"
                    "Pass `model_id=<actual-hf-model-repo>` once an HF "
                    "model-repo distribution exists (currently the project "
                    "hosts weights inside the Space, not as a model card)."
                ),
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
            for r, img in zip(rgbs, pil_imgs, strict=False):
                out = self._pipe(
                    img,
                    denoising_steps=self.num_inference_steps,
                    ensemble_size=self.ensemble_size,
                    domain=self.domain,
                    show_progress_bar=False,
                )
                d = _extract_depth(out, r.shape[:2])
                depths.append(d.astype(np.float32))
        return depths if is_batch else depths[0]


def _extract_depth(out, target_hw):
    # Common shapes in GeoWizard's pipeline output
    for attr in ("depth_np", "depth", "prediction"):
        v = getattr(out, attr, None)
        if v is None and hasattr(out, "__getitem__"):
            try:
                v = out[attr]
            except Exception:
                v = None
        if v is not None:
            arr = np.asarray(v if not hasattr(v, "cpu") else v.cpu().numpy(), dtype=np.float32)
            while arr.ndim > 2:
                arr = arr.squeeze(0) if arr.shape[0] == 1 else arr[0]
            if arr.shape != tuple(target_hw):
                arr = _resize_bilinear(arr, target_hw)
            return arr
    from rpx_benchmark.exceptions import AdapterError

    raise AdapterError(
        "GeoWizard pipeline returned an unrecognised output shape.",
        hint="Expected one of `depth_np`, `depth`, or `prediction` on the "
        "pipeline output. Inspect with `print(out)`.",
    )


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
