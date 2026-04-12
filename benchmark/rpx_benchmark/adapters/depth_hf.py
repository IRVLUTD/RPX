"""Deprecated shim: depth_hf adapter moved to ``rpx_benchmark.reference.adapters.depth_hf``.

Re-exported names continue to work with a ``DeprecationWarning``. The
shim will be removed once the next minor release cuts; update imports
to the new path before then::

    # old (deprecated):
    from rpx_benchmark.adapters.depth_hf import make_hf_depth_model

    # new:
    from rpx_benchmark.reference.adapters.depth_hf import make_hf_depth_model
"""

from __future__ import annotations

import warnings as _warnings

from ..reference.adapters.depth_hf import (  # noqa: F401 — re-export
    HFDepthInputAdapter,
    HFDepthOutputAdapter,
    make_hf_depth_model,
)

_warnings.warn(
    "rpx_benchmark.adapters.depth_hf has moved to "
    "rpx_benchmark.reference.adapters.depth_hf; update your imports. "
    "The shim will be removed in the next minor release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HFDepthInputAdapter", "HFDepthOutputAdapter", "make_hf_depth_model"]
