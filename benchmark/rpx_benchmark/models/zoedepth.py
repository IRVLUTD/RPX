"""ZoeDepth — legacy metric depth baseline (archived upstream May 2025).

Kept as a robot-learning anchor because it's the RGB->3D workhorse
inside 3D Diffusion Policy (RSS'24) and FlowPolicy.
"""

from __future__ import annotations

from ..adapters import BenchmarkableModel
from ..adapters.depth_hf import make_hf_depth_model


def zoedepth_nyu(*, device: str = "cuda", **kwargs) -> BenchmarkableModel:
    return make_hf_depth_model(
        "Intel/zoedepth-nyu-kitti",
        device=device,
        name="zoedepth_nyu",
        **kwargs,
    )
