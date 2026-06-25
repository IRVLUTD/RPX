"""Skeleton adapter classes for every D1-F (monocular depth) model.

Each class is a one-line subclass of :class:`DepthAdapterSkeleton`
that pins it to a particular ``MODEL_KEY``; the scaffold infers
``task``, ``depth_output_kind``, ``name``, and ``install_hint`` from
the central :data:`~rpx_benchmark.adapters.depth_scaffold.DEPTH_MODEL_CARDS`
table.

To implement a real adapter:

1. Create ``adapters/depth/<model>.py`` (e.g. ``da3.py``).
2. Define a class that subclasses :class:`DepthAdapterSkeleton`
   (or :class:`~rpx_benchmark.adapters.base.BenchmarkableModel` if you
   want the input/output adapter composition pattern), sets
   ``MODEL_KEY`` to the same key as the skeleton here, and overrides
   ``setup()`` and ``predict()``.
3. Remove the matching skeleton class from this file (or import the
   real implementation into ``__init__.py`` so it's the one users
   pick up by default).

The skeleton classes are intentionally one-liners — every other
attribute (task, output kind, install hint, paper reference) lives in
``DEPTH_MODEL_CARDS`` so there's one canonical source of truth for
the roster.
"""

from __future__ import annotations

from ..depth_scaffold import DepthAdapterSkeleton


class DA3MetricLAdapter(DepthAdapterSkeleton):
    """DA3 Metric-L (Depth Anything 3, metric Large)."""

    MODEL_KEY = "da3-metric-l"


class DAV2LargeAdapter(DepthAdapterSkeleton):
    """Depth Anything V2 — Large metric variant."""

    MODEL_KEY = "da-v2-large"


class DepthProAdapter(DepthAdapterSkeleton):
    """Apple Depth Pro."""

    MODEL_KEY = "depth-pro"


class UniDepthV2Adapter(DepthAdapterSkeleton):
    """UniDepth V2 (Piccinelli et al. 2024)."""

    MODEL_KEY = "unidepth-v2"


class Metric3DV2Adapter(DepthAdapterSkeleton):
    """Metric3D V2."""

    MODEL_KEY = "metric3d-v2"


class MoGe2ViTLAdapter(DepthAdapterSkeleton):
    """MoGe-2 ViT-L (Microsoft, relative)."""

    MODEL_KEY = "moge-2-vit-l"


class HyDenAdapter(DepthAdapterSkeleton):
    """HyDen (relative)."""

    MODEL_KEY = "hyden"


class Lotus2Adapter(DepthAdapterSkeleton):
    """Lotus-2 (He et al. 2024, relative diffusion-based)."""

    MODEL_KEY = "lotus-2"


class FE2EAdapter(DepthAdapterSkeleton):
    """FE2E (relative)."""

    MODEL_KEY = "fe2e"


class DepthLMAdapter(DepthAdapterSkeleton):
    """DepthLM (relative)."""

    MODEL_KEY = "depthlm"


__all__ = [
    "DA3MetricLAdapter",
    "DAV2LargeAdapter",
    "DepthProAdapter",
    "UniDepthV2Adapter",
    "Metric3DV2Adapter",
    "MoGe2ViTLAdapter",
    "HyDenAdapter",
    "Lotus2Adapter",
    "FE2EAdapter",
    "DepthLMAdapter",
]
