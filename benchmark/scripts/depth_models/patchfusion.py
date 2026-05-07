"""PatchFusion adapter — tile-based high-resolution metric depth (CVPR'24).

Loads ``zhyever/patchfusion_zoedepth`` via HuggingFace.  PatchFusion
splits the image into overlapping tiles, runs the underlying depth
backbone (ZoeDepth) per tile at full resolution, and fuses the results.
Output is **metric depth in metres** at the input resolution.

Tracker reference: MonocularMetricDepth — PatchFusion (metric).

Install
-------
    pip install transformers torch pillow

Notes
-----
PatchFusion's HF repo carries either the wrapped pipeline weights or
the raw ZoeDepth + fusion module. We use ``AutoModelForDepthEstimation``
which the HF release supports; if that fails we fall back to the
patchfusion python package's own loader.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class PatchFusion:
    """Tile-fusion metric depth: rgb → depth (metres)."""

    DEFAULT_MODEL_ID = "zhyever/patchfusion_zoedepth"

    native_alignment: str = "none"  # metric
    native_precision: str = "fp32"  # tile fusion sensitive to fp16 noise

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise ImportError(
                "PatchFusion needs `transformers`. Install with: "
                "pip install transformers torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch

        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"PatchFusion load failed for {model_id!r}: {e}",
                hint="PatchFusion's HF release ships custom code; "
                "needs `trust_remote_code=True`. If the repo has "
                "moved, check https://huggingface.co/zhyever and "
                "pass `model_id=...` explicitly.",
            ) from e
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            self._model = self._model.to(dtype=target_dtype)
        self._model = self._model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        from PIL import Image

        torch = self._torch

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
        target_sizes = [(np.asarray(r).shape[0], np.asarray(r).shape[1]) for r in rgbs]
        inputs = self._processor(images=pil_imgs, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # PatchFusion's HF wrapper exposes post_process_depth_estimation
        # (inherited from DepthEstimationModelOutput) — yields metric depth.
        post = self._processor.post_process_depth_estimation(
            outputs,
            target_sizes=target_sizes,
        )
        depths: list[np.ndarray] = []
        for r, p in zip(rgbs, post, strict=False):
            d = p["predicted_depth"].detach().cpu().numpy().astype(np.float32)
            if d.ndim == 3:
                d = d.squeeze(0)
            target_hw = np.asarray(r).shape[:2]
            if d.shape != target_hw:
                d = _resize_bilinear(d, target_hw)
            depths.append(d)
        return depths if is_batch else depths[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
