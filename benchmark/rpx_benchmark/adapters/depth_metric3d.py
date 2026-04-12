"""Deprecated shim: moved to ``rpx_benchmark.reference.adapters.depth_metric3d``."""

from __future__ import annotations

import warnings as _warnings

from ..reference.adapters.depth_metric3d import (  # noqa: F401
    Metric3DInputAdapter,
    Metric3DOutputAdapter,
    make_metric3d_v2_model,
)

_warnings.warn(
    "rpx_benchmark.adapters.depth_metric3d has moved to "
    "rpx_benchmark.reference.adapters.depth_metric3d; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Metric3DInputAdapter", "Metric3DOutputAdapter", "make_metric3d_v2_model"]
