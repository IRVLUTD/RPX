"""DA3 Metric-L — Depth Anything 3, large metric variant (single-image mode).

Image Depth adapter wrapping the official ``depth-anything/DA3-LARGE``
HF release in single-image mode. Verified model_id and load path:
ByteDance Seed's ``depth_anything_3`` Python package
(``DepthAnything3.from_pretrained``).

The same checkpoint backs ``scripts/video_depth_models/da3_video.py``
(multi-view mode); this module is the per-frame entry point that
matches the canonical roster ``da3-metric-l`` key in
:data:`DEPTH_MODEL_CARDS`.

**Verification status**: model_id verified against HF model card. The
predict body is best-effort against the public API — team should
smoke before production.

Install
-------

::

    pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
_METRIC_FOCAL_DIVISOR = 300.0


class DA3Metric:
    """Callable adapter: rgb (H×W×3 uint8) → depth (H×W float32, metres).

    Matches the contract every other adapter in ``scripts/depth_models/``
    exposes so ``BatchedDepthBenchmarkModel`` can wrap it.
    """

    #: DA3 emits metric depth.
    native_alignment: str = "none"
    native_precision: str = "fp16"

    def __init__(
        self,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
        focal_length_px: Optional[float] = None,
    ) -> None:
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "DA3Metric needs the `depth_anything_3` package. "
                "Install with: pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3"
            ) from e
        self.device = device
        self.batch_size = int(batch_size)
        self.focal_length_px = float(focal_length_px or 615.0)
        self._model = DepthAnything3.from_pretrained(_MODEL_ID)
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        is_batch = isinstance(rgb, (list, tuple))
        rgbs = list(rgb) if is_batch else [rgb]
        for r in rgbs:
            r_arr = np.asarray(r)
            if r_arr.ndim != 3 or r_arr.shape[2] != 3:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"expected H×W×3 RGB uint8, got shape {r_arr.shape}",
                )
        rgbs = [np.asarray(r, dtype=np.uint8) for r in rgbs]

        # Pinned DA3 exposes the high-level inference() API. Calling
        # each image separately preserves monocular behaviour.
        outs = [self._model.inference([r]) for r in rgbs]

        depths: list[np.ndarray] = []
        for r, o in zip(rgbs, outs, strict=False):
            d = o.depth if hasattr(o, "depth") else o
            if isinstance(d, dict):
                d = d.get("depth")
                if d is None:
                    d = o.get("depth_map")
            elif isinstance(d, (list, tuple)):
                d = d[0]
            if d is None:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError("DA3 metric inference returned no depth field.")
            if hasattr(d, "detach"):
                d = d.detach().cpu().float().numpy()
            d = np.squeeze(np.asarray(d, dtype=np.float32))
            if d.ndim != 2:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"DA3 metric inference returned depth shape {d.shape}; expected 2-D."
                )

            # Official DA3METRIC conversion:
            # metric_depth = processed_focal_px * raw_depth / 300.
            processed_focal_px = self.focal_length_px * (
                d.shape[1] / float(r.shape[1])
            )
            d = d * (processed_focal_px / _METRIC_FOCAL_DIVISOR)
            target_hw = r.shape[:2]
            if d.shape != target_hw:
                from PIL import Image as _Image

                d = np.asarray(
                    _Image.fromarray(d, mode="F").resize(
                        (target_hw[1], target_hw[0]), _Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
            depths.append(d)
        return depths if is_batch else depths[0]
