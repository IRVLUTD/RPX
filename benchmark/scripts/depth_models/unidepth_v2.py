"""UniDepth V2 adapter (Piccinelli et al., https://arxiv.org/abs/2403.18913).

Loads ``lpiccinelli/unidepth-v2-vitl14`` via the ``unidepth`` Python
package (not pure HuggingFace transformers — UniDepth ships its own
loader with a custom forward that returns metric depth + intrinsics).
Output is metric depth in metres at the input resolution.

Tracker reference: MonocularMetricDepth sheet, row "UniDepth V2".

Install
-------
    pip install unidepth torch timm pillow huggingface-hub

(The ``unidepth`` package vendors the model code; the weights are
fetched from HuggingFace via ``hf_hub_download`` on first use.)

Usage
-----
    from depth_models.unidepth_v2 import UniDepthV2
    import rpx_benchmark as rpx

    ud = UniDepthV2(device="cuda", batch_size=8)
    model = rpx.make_numpy_depth_model(ud, name="UniDepth-V2-ViTL14")
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class UniDepthV2:
    """Callable adapter: rgb (H×W×3 uint8) → depth (H×W float32, metres).

    Supports batched dispatch by stacking RGBs into a single tensor and
    calling ``model.infer`` on the batch dimension.
    """

    DEFAULT_MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"

    #: UniDepth V2 publishes metric depth in metres directly.
    native_alignment: str = "none"

    #: UniDepth V2's ViT-L encoder runs cleanly in fp16 and is the
    #: paper's reported inference mode. Drives the OperatingPoint
    #: precision tag.
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
    ) -> None:
        try:
            import torch
            from unidepth.models import UniDepthV2 as _UniDepthV2Backbone
        except ImportError as e:
            raise ImportError(
                "UniDepthV2 needs the pinned upstream runtime dependencies. "
                "Run setup_depth_smoke_env.py --model unidepth-v2."
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = _UniDepthV2Backbone.from_pretrained(model_id)
        self._model = self._model.to(device).eval()
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            self._model = self._model.to(dtype=target_dtype)

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
            r = np.asarray(r)
            if r.ndim != 3 or r.shape[2] != 3:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"expected H×W×3 RGB uint8, got shape {r.shape}",
                    hint="Adapter contract: each input must be a (H, W, 3) "
                    "uint8 numpy array. Got an unexpected ndim or channel count.",
                )

        # UniDepth's `.infer(rgb)` expects [3, H, W] uint8 torch tensor (or
        # [B, 3, H, W] for batched). We stack and dispatch one forward.
        torch = self._torch
        tensors = [
            torch.from_numpy(np.asarray(r, dtype=np.uint8)).permute(2, 0, 1).contiguous()
            for r in rgbs
        ]
        # Same H, W across the batch (RPX is uniform 480×640) → stack works.
        x = torch.stack(tensors, dim=0).to(self.device)

        with torch.inference_mode():
            out = self._model.infer(x)
        # Output: dict with key "depth" of shape [B, 1, H, W] in metres.
        depth_t = out["depth"]
        if depth_t.dim() == 4:
            depth_t = depth_t.squeeze(1)  # [B, H, W]

        depths_np: list[np.ndarray] = []
        for i, r in enumerate(rgbs):
            d = depth_t[i].detach().cpu().numpy().astype(np.float32)
            target_hw = np.asarray(r).shape[:2]
            if d.shape != target_hw:
                d = _resize_bilinear(d, target_hw)
            depths_np.append(d)
        return depths_np if is_batch else depths_np[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
