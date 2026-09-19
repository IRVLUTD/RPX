"""Protocol metadata for the RPX NVS model slate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InputTrack = Literal["rgb", "rgbd", "compact"]
ExecutionMode = Literal["feed_forward", "scene_optimization", "compression"]


@dataclass(frozen=True)
class NVSModelSpec:
    key: str
    display_name: str
    track: InputTrack
    execution_mode: ExecutionMode
    upstream: str
    uses_sensor_depth: bool
    uses_known_poses: bool
    gate_context_count: int = 2
    integration: Literal["native", "external_bridge"] = "external_bridge"


MODEL_SPECS: dict[str, NVSModelSpec] = {
    "dn_splatter": NVSModelSpec(
        "dn_splatter", "DN-Splatter", "rgbd", "scene_optimization",
        "https://github.com/maturk/dn-splatter", True, True,
    ),
    "splatam": NVSModelSpec(
        "splatam", "SplaTAM", "rgbd", "scene_optimization",
        "https://github.com/spla-tam/SplaTAM", True, True,
    ),
    "rtg_slam": NVSModelSpec(
        "rtg_slam", "RTG-SLAM", "rgbd", "scene_optimization",
        "https://github.com/MisEty/RTG-SLAM", True, True,
    ),
    "gaus_slam": NVSModelSpec(
        "gaus_slam", "GauS-SLAM", "rgbd", "scene_optimization",
        "https://github.com/gaus-slam/gaus-slam", True, True,
    ),
    "depthsplat": NVSModelSpec(
        "depthsplat", "DepthSplat", "rgb", "feed_forward",
        "https://github.com/cvg/depthsplat", False, True,
        integration="native",
    ),
    "mvsplat": NVSModelSpec(
        "mvsplat", "MVSplat", "rgb", "feed_forward",
        "https://github.com/donydchen/mvsplat", False, True,
    ),
    "pixelsplat": NVSModelSpec(
        "pixelsplat", "pixelSplat", "rgb", "feed_forward",
        "https://github.com/dcharatan/pixelsplat", False, True,
    ),
    "nopo_splat": NVSModelSpec(
        "nopo_splat", "NoPoSplat", "rgb", "feed_forward",
        "https://github.com/cvg/NoPoSplat", False, False,
    ),
    "lightgaussian": NVSModelSpec(
        "lightgaussian", "LightGaussian", "compact", "compression",
        "https://github.com/VITA-Group/LightGaussian", False, True,
    ),
    "compgs": NVSModelSpec(
        "compgs", "CompGS", "compact", "compression",
        "https://github.com/UCDvision/compact3d", False, True,
    ),
}

PAPER_MODEL_KEYS: tuple[str, ...] = tuple(MODEL_SPECS)


def models_by_track(track: InputTrack) -> tuple[str, ...]:
    return tuple(key for key, spec in MODEL_SPECS.items() if spec.track == track)
