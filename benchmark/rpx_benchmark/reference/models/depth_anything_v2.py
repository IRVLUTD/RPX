"""Depth Anything V2 — metric-indoor variants (Small / Base / Large).

Thin factories over :func:`rpx_benchmark.adapters.depth_hf.make_hf_depth_model`.
"""

from __future__ import annotations

from ...adapters import BenchmarkableModel
from ..adapters.depth_hf import make_hf_depth_model

_SMALL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_BASE = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
_LARGE = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def depth_anything_v2_metric_indoor_small(
    *, device: str = "cuda", **kwargs,
) -> BenchmarkableModel:
    return make_hf_depth_model(
        _SMALL, device=device,
        name="depth_anything_v2_metric_indoor_small",
        **kwargs,
    )


def depth_anything_v2_metric_indoor_base(
    *, device: str = "cuda", **kwargs,
) -> BenchmarkableModel:
    return make_hf_depth_model(
        _BASE, device=device,
        name="depth_anything_v2_metric_indoor_base",
        **kwargs,
    )


def depth_anything_v2_metric_indoor_large(
    *, device: str = "cuda", **kwargs,
) -> BenchmarkableModel:
    return make_hf_depth_model(
        _LARGE, device=device,
        name="depth_anything_v2_metric_indoor_large",
        **kwargs,
    )
