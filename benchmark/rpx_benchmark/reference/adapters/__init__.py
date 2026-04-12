"""Reference input/output adapters for popular model families.

Each module here targets one model family (HuggingFace depth,
HuggingFace segmentation, UniDepth V2, Metric3D V2, ...). The
adapters follow the stable
:mod:`rpx_benchmark.adapters.base` protocols —
``InputAdapter`` / ``OutputAdapter`` / ``BenchmarkableModel`` — and
are usable as-is or as copy-from templates.
"""

from .depth_hf import (
    HFDepthInputAdapter,
    HFDepthOutputAdapter,
    make_hf_depth_model,
)
from .depth_metric3d import (
    Metric3DInputAdapter,
    Metric3DOutputAdapter,
    make_metric3d_v2_model,
)
from .depth_unidepth import (
    UniDepthInputAdapter,
    UniDepthOutputAdapter,
    make_unidepth_v2_model,
)
from .seg_hf import (
    HFInstanceSegInputAdapter,
    HFInstanceSegOutputAdapter,
    make_hf_instance_seg_model,
)

__all__ = [
    "HFDepthInputAdapter",
    "HFDepthOutputAdapter",
    "make_hf_depth_model",
    "Metric3DInputAdapter",
    "Metric3DOutputAdapter",
    "make_metric3d_v2_model",
    "UniDepthInputAdapter",
    "UniDepthOutputAdapter",
    "make_unidepth_v2_model",
    "HFInstanceSegInputAdapter",
    "HFInstanceSegOutputAdapter",
    "make_hf_instance_seg_model",
]
