"""Depth Pro (Apple) — sharp monocular metric depth.

Loaded via the ``apple/DepthPro-hf`` transformers port.
"""

from __future__ import annotations

from ...adapters import BenchmarkableModel
from ..adapters.depth_hf import make_hf_depth_model


def depth_pro(*, device: str = "cuda", **kwargs) -> BenchmarkableModel:
    return make_hf_depth_model(
        "apple/DepthPro-hf",
        device=device,
        name="depth_pro",
        **kwargs,
    )
