"""Metric3D V2 — ViT Small / Large / Giant2 via torch.hub.

All variants use the same canonical-focal letterbox pipeline; only the
``entry`` argument to ``torch.hub.load`` changes.
"""

from __future__ import annotations

from ...adapters import BenchmarkableModel
from ..adapters.depth_metric3d import make_metric3d_v2_model


def metric3d_v2_vit_small(
    *,
    device: str = "cuda",
    fx_real: float = 605.0,
    **kwargs,
) -> BenchmarkableModel:
    return make_metric3d_v2_model(
        device=device,
        fx_real=fx_real,
        entry="metric3d_vit_small",
        name="metric3d_v2_vit_small",
        **kwargs,
    )


def metric3d_v2_vit_large(
    *,
    device: str = "cuda",
    fx_real: float = 605.0,
    **kwargs,
) -> BenchmarkableModel:
    return make_metric3d_v2_model(
        device=device,
        fx_real=fx_real,
        entry="metric3d_vit_large",
        name="metric3d_v2_vit_large",
        **kwargs,
    )


def metric3d_v2_vit_giant2(
    *,
    device: str = "cuda",
    fx_real: float = 605.0,
    **kwargs,
) -> BenchmarkableModel:
    return make_metric3d_v2_model(
        device=device,
        fx_real=fx_real,
        entry="metric3d_vit_giant2",
        name="metric3d_v2_vit_giant2",
        **kwargs,
    )
