"""MoGe / MoGe-2 adapter — Microsoft's joint depth + normal model.

Two checkpoints share this adapter via ``model_id``:

* ``Ruicheng/moge-2-vitl-normal`` — MoGe-2, **metric** depth.
* ``Ruicheng/moge-vitl``          — MoGe v1, **affine-invariant** depth
  (relative).

Different ``native_alignment`` per checkpoint (set by the registry
builders, not hardcoded on the class).

MoGe ships its own loader via the ``moge`` python package; we don't go
through HF transformers' depth-estimation pipeline because MoGe outputs
multiple geometry quantities (depth, points, mask, intrinsics) and the
HF wrapper would discard everything except depth in a way that loses
the FOV calibration the model emits.

Tracker reference: MonocularMetricDepth — MoGe-2 (metric), MoGe v1 (relative).

Install
-------
    pip install git+https://github.com/microsoft/MoGe.git torch pillow huggingface-hub
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class MoGe:
    """Geometry-aware depth: rgb → depth (metres if metric checkpoint)."""

    DEFAULT_MODEL_ID = "Ruicheng/moge-2-vitl-normal"  # MoGe-2 (metric)

    #: Set per-checkpoint by the registry builders.
    native_alignment: str = "none"  # default for MoGe-2 metric
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        native_alignment: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from moge.model.v2 import MoGeModel as _MoGeV2
        except ImportError:
            try:
                import torch
                from moge.model import MoGeModel as _MoGeV2  # v1 path
            except ImportError as e:
                raise ImportError(
                    "MoGe needs the `moge` package. Install with: "
                    "pip install git+https://github.com/microsoft/MoGe.git torch pillow huggingface-hub"
                ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        if native_alignment is not None:
            self.native_alignment = native_alignment

        self._model = _MoGeV2.from_pretrained(model_id).to(device).eval()
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
                )

        torch = self._torch
        # MoGe expects [3, H, W] uint8 → float in [0, 1] internally.
        tensors = [
            torch.from_numpy(np.asarray(r, dtype=np.uint8)).permute(2, 0, 1).contiguous().float()
            / 255.0
            for r in rgbs
        ]
        x = torch.stack(tensors, dim=0).to(self.device)

        with torch.inference_mode():
            output = self._model.infer(x)
        # MoGe.infer returns a dict: {"depth": [B, H, W], "points": [B, H, W, 3],
        #                              "intrinsics": [B, 3, 3], "mask": [B, H, W]}.
        depth_t = output["depth"]
        if depth_t.dim() == 4:
            depth_t = depth_t.squeeze(1)

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
