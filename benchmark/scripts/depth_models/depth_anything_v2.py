"""Depth Anything V2 (Metric) adapter — both indoor and outdoor heads.

Two checkpoints share this adapter, parametrised by ``model_id``:

* ``depth-anything/Depth-Anything-V2-Metric-Hypersim-Large`` — indoor (Hypersim)
* ``depth-anything/Depth-Anything-V2-Metric-VKITTI-Large``    — outdoor (VKITTI)

Both publish metric depth in metres; the difference is supervised data
domain. Reporting both lets the paper show the within-method indoor /
outdoor split — important since RPX has both indoor tabletop scenes
(e.g., ``scene23.jo.4f``) and outdoor ones (e.g., ``scene55.TENNISCOURTS``).

Tracker reference: MonocularMetricDepth sheet, row "Depth Anything V2".

Install
-------
    pip install transformers torch timm pillow

Usage
-----
    from depth_models.depth_anything_v2 import DepthAnythingV2Metric
    import rpx_benchmark as rpx

    dav2 = DepthAnythingV2Metric(
        model_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
        device="cuda", batch_size=8,
    )
    model = rpx.make_numpy_depth_model(dav2, name="DA-V2-Metric-Hypersim-L")
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class DepthAnythingV2Metric:
    """Callable adapter: rgb (H×W×3 uint8) → depth (H×W float32, metres).

    Supports batched dispatch via the HF pipeline list contract.
    """

    # The '-hf' suffix variants are HF-transformers-compatible (config.json
    # present); the non-'-hf' repos are PyTorch state-dicts that need the
    # standalone depth_anything_v2 python package.
    INDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    OUTDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

    #: Both heads publish metric depth in metres.
    native_alignment: str = "none"

    #: DA-V2 supports fp16 inference cleanly (DPT backbone + ViT-L
    #: encoder, well-tested under autocast). Reading this attribute
    #: drives the runner's OperatingPoint precision tag for DRS.
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = INDOOR_MODEL_ID,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "DepthAnythingV2Metric needs `transformers`. Install with: "
                "pip install transformers torch timm pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
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
    from PIL import Image

    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
