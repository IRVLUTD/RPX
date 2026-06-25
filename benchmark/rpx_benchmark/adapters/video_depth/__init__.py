"""Video-depth adapters (paper task Video Depth: VIDEO_DEPTH).

Same skeleton pattern as :mod:`rpx_benchmark.adapters.depth`. The
Video Depth roster differs from Image Depth in iteration unit (per-clip) and in
the model family — sliding-window video models (DepthCrafter,
ChronoDepth, RollingDepth, Video DA), streaming models, and
multi-view 3D models (MonST3R, VGGT-Ω). DA3 appears in both
rosters: the same weights, different input context.

See :mod:`rpx_benchmark.adapters.depth_scaffold` for the
implementation pattern.
"""

from __future__ import annotations

from ..depth_scaffold import (
    DEPTH_MODEL_CARDS,
    DepthAdapterSkeleton,
    DepthModelCard,
    available_depth_adapters,
    lazy_import,
)
from . import skeletons as _skeletons  # noqa: F401 — registers subclasses

__all__ = [
    "DepthAdapterSkeleton",
    "DepthModelCard",
    "DEPTH_MODEL_CARDS",
    "available_depth_adapters",
    "lazy_import",
]
