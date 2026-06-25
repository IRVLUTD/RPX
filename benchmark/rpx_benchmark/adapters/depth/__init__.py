"""Depth-model adapters (paper task Image Depth: MONOCULAR_DEPTH).

This subpackage holds one adapter class per model in the Image Depth roster.
Adapters are skeletons today (see ``skeletons.py``) — they declare the
right ``task`` and ``depth_output_kind`` and raise
``NotImplementedError`` with the install hint when ``predict`` is
called. A contributor who installs a model fills in its forward call
without touching any other adapter.

Re-exports the canonical model-card table and the skeleton base class
so callers can do ``from rpx_benchmark.adapters.depth import
DepthAdapterSkeleton, DEPTH_MODEL_CARDS, available_depth_adapters``.
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
