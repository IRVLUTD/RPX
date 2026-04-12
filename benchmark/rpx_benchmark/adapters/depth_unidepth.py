"""Deprecated shim: moved to ``rpx_benchmark.reference.adapters.depth_unidepth``."""

from __future__ import annotations

import warnings as _warnings

from ..reference.adapters.depth_unidepth import (  # noqa: F401
    UniDepthInputAdapter,
    UniDepthOutputAdapter,
    make_unidepth_v2_model,
)

_warnings.warn(
    "rpx_benchmark.adapters.depth_unidepth has moved to "
    "rpx_benchmark.reference.adapters.depth_unidepth; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["UniDepthInputAdapter", "UniDepthOutputAdapter", "make_unidepth_v2_model"]
