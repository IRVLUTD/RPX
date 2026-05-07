"""Generic HuggingFace depth-estimation pipeline adapter.

Many monocular depth models on HuggingFace are wrapped uniformly via
``transformers.pipeline("depth-estimation")``: the pipeline returns
``{"predicted_depth": tensor [1, H, W]}`` per image and supports list
input for batched dispatch. ``HFDepthEstimationAdapter`` parametrises
the pattern so adding a new HF-pipeline-friendly model is a single line
in ``MODEL_REGISTRY`` — no per-model file needed.

This adapter handles every model in our zoo whose only customisation is
the checkpoint id and the alignment policy:

* DA-V2 Metric Indoor / Outdoor (metric, native_alignment="none")
* DA-V2 relative              (relative, native_alignment="ls_affine")
* DA V1                       (relative, native_alignment="ls_affine")
* MiDaS v3.1 (DPT-BEiT-L)     (relative, native_alignment="ls_affine")
* Distill-Any-Depth           (relative, native_alignment="ls_affine")

Models with bespoke post-processing (Depth Pro's FOV correction, ZoeDepth's
metric scaling, UniDepth V2's intrinsics estimation, diffusion ensembling)
keep their own per-model adapter. This one is just for the boring case.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class HFDepthEstimationAdapter:
    """Callable adapter wrapping ``transformers.pipeline("depth-estimation", model=...)``.

    Single-image: ``np.ndarray (H, W, 3) uint8 → np.ndarray (H, W) float32``.
    Batched: ``list[np.ndarray] → list[np.ndarray]``.

    Parameters
    ----------
    model_id : str
        HuggingFace checkpoint id (must have ``config.json`` for the
        auto-pipeline; some original repos ship state-dicts only and
        require the ``-hf`` suffix variant).
    device : str
        Torch device id ("cuda", "cpu", "cuda:0", ...).
    batch_size : int
        Pipeline batch size (controls dispatch shape; model semantics
        unchanged).
    native_alignment : str
        Reported via the attribute of the same name; the runner reads
        it to pick a default alignment for comprehensive metrics.
        ``"none"`` for metric models, ``"ls_affine"`` for up-to-scale
        / relative-depth models.
    dtype : str, optional
        Inference dtype ("float16" for fp16). Defaults to model's stored dtype.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        batch_size: int = 1,
        native_alignment: str = "none",
        native_precision: str = "fp32",
        dtype: Optional[str] = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "HFDepthEstimationAdapter needs `transformers`. Install with: "
                "pip install transformers torch timm pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.native_alignment = native_alignment
        self.native_precision = native_precision
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
            torch_dtype=dtype,
            batch_size=self.batch_size,
        )

    @property
    def torch_module(self):
        return getattr(self._pipe, "model", None)

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
                    "uint8 numpy array. Got an unexpected ndim or channel count.",
                )

        pil_imgs = [Image.fromarray(np.asarray(r, dtype=np.uint8)) for r in rgbs]
        out = self._pipe(pil_imgs)
        if not isinstance(out, list):
            out = [out]

        depths: list[np.ndarray] = []
        for r, o in zip(rgbs, out, strict=False):
            d = o["predicted_depth"].detach().cpu().numpy().astype(np.float32)
            if d.ndim == 3:
                d = d.squeeze(0)
            target_hw = np.asarray(r).shape[:2]
            if d.shape != target_hw:
                d = _resize_bilinear(d, target_hw)
            depths.append(d)
        return depths if is_batch else depths[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float array to (H, W). PIL-only; no OpenCV dep."""
    from PIL import Image

    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
