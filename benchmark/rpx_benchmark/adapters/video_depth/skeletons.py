"""Skeleton adapter classes for every D1-V (video depth) model.

Same one-line-subclass pattern as the D1-F skeletons; see
:mod:`rpx_benchmark.adapters.depth.skeletons` for the conventions
and :mod:`rpx_benchmark.adapters.depth_scaffold` for the base class.
"""

from __future__ import annotations

from ..depth_scaffold import DepthAdapterSkeleton


class DA3VideoAdapter(DepthAdapterSkeleton):
    """DA3 in video mode — same weights as ``da3-metric-l`` but fed a clip."""

    MODEL_KEY = "da3-video"


class DepthCrafterAdapter(DepthAdapterSkeleton):
    """DepthCrafter (Tencent, sliding-window diffusion, relative)."""

    MODEL_KEY = "depth-crafter"


class VideoDAAdapter(DepthAdapterSkeleton):
    """Video Depth Anything."""

    MODEL_KEY = "video-da"


class ChronoDepthAdapter(DepthAdapterSkeleton):
    """ChronoDepth (sliding-window, relative)."""

    MODEL_KEY = "chrono-depth"


class RollingDepthAdapter(DepthAdapterSkeleton):
    """RollingDepth (streaming, relative)."""

    MODEL_KEY = "rolling-depth"


class D4RTAdapter(DepthAdapterSkeleton):
    """D4RT."""

    MODEL_KEY = "d4rt"


class GemDepthAdapter(DepthAdapterSkeleton):
    """GemDepth."""

    MODEL_KEY = "gem-depth"


class ViGeoAdapter(DepthAdapterSkeleton):
    """ViGeo."""

    MODEL_KEY = "vigeo"


class MonST3RAdapter(DepthAdapterSkeleton):
    """MonST3R (multi-view 4-frame chunks)."""

    MODEL_KEY = "monst3r"


class VGGTOmegaAdapter(DepthAdapterSkeleton):
    """VGGT-Ω (multi-view, large variant)."""

    MODEL_KEY = "vggt-omega"


__all__ = [
    "DA3VideoAdapter",
    "DepthCrafterAdapter",
    "VideoDAAdapter",
    "ChronoDepthAdapter",
    "RollingDepthAdapter",
    "D4RTAdapter",
    "GemDepthAdapter",
    "ViGeoAdapter",
    "MonST3RAdapter",
    "VGGTOmegaAdapter",
]
