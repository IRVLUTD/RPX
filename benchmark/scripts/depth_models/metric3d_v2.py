"""Metric3D V2 adapter — universal metric depth (Yvan Yin et al., 2024).

Loads the ``metric_depth_vit_giant2`` weights via the project's torch.hub
entry. Output is **metric depth in metres**.

Install
-------
The ``metric3d`` python package isn't on PyPI; the upstream ships via
torch.hub. The adapter loads through:

    model = torch.hub.load("YvanYin/Metric3D", "metric3d_vit_giant2",
                            pretrain=True, trust_repo=True)

Make sure your environment has internet access on first load (torch.hub
clones the repo + downloads weights to ``~/.cache/torch/hub/``).
Subsequent calls are offline.

Tracker reference: MonocularMetricDepth — Metric3D V2 (metric).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class Metric3DV2:
    """Universal metric depth: rgb → depth (metres)."""

    DEFAULT_HUB_ENTRY = "metric3d_vit_giant2"
    # Pin the upstream source so local/server smoke runs execute identical
    # model code even if the repository's default branch moves.
    DEFAULT_HUB_REPO = (
        "YvanYin/Metric3D:eb5b6fac0dc155e4e52f576e304fbf11655ff339"
    )

    native_alignment: str = "none"  # metric
    native_precision: str = "fp32"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        hub_entry: str = DEFAULT_HUB_ENTRY,
        hub_repo: str = DEFAULT_HUB_REPO,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("Metric3D V2 needs torch.") from e
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self.hub_entry = hub_entry
        self.hub_repo = hub_repo

        try:
            model = torch.hub.load(
                hub_repo, hub_entry, pretrain=True, trust_repo=True, source="github"
            )
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"Metric3D V2 hub.load failed: {e}",
                hint="First-load needs internet access (torch.hub clones the "
                "repo + downloads weights to ~/.cache/torch/hub/). Once "
                "loaded, re-runs are offline. Verify the repo name with "
                "`torch.hub.list('YvanYin/Metric3D')`.",
            ) from e
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            model = model.to(dtype=target_dtype)
        self._model = model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
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

        # Metric3D's hub model expects [B, 3, H, W] float in [0, 1].
        tensors = [
            torch.from_numpy(np.asarray(r, dtype=np.uint8)).permute(2, 0, 1).contiguous().float()
            / 255.0
            for r in rgbs
        ]
        x = torch.stack(tensors, dim=0).to(self.device)

        with torch.inference_mode():
            try:
                pred = self._model.inference({"input": x})
            except (AttributeError, TypeError):
                # Fall back to plain forward
                pred = self._model(x)
        # ``pred`` is a dict in the hub flow, or a tensor in the fallback.
        if isinstance(pred, dict):
            depth_t = pred.get("prediction") or pred.get("depth")
        else:
            depth_t = pred
        if depth_t is None:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                "Metric3D V2 returned no depth tensor.",
                hint="Inspect the hub model's output dict keys with `torch.hub.help(repo, entry)`.",
            )
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
