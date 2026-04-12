"""Deprecated shim: moved to ``rpx_benchmark.reference.adapters.seg_hf``."""

from __future__ import annotations

import warnings as _warnings

from ..reference.adapters.seg_hf import (  # noqa: F401
    HFInstanceSegInputAdapter,
    HFInstanceSegOutputAdapter,
    make_hf_instance_seg_model,
)

_warnings.warn(
    "rpx_benchmark.adapters.seg_hf has moved to "
    "rpx_benchmark.reference.adapters.seg_hf; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HFInstanceSegInputAdapter",
    "HFInstanceSegOutputAdapter",
    "make_hf_instance_seg_model",
]
