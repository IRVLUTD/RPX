"""Pluggable metric registry for RPX benchmark tasks.

Adding a new metric to an existing task is a three-line change::

    from rpx_benchmark.metrics import MetricCalculator, register_metric
    from rpx_benchmark.api import TaskType

    @register_metric(TaskType.MONOCULAR_DEPTH)
    class DepthMedian(MetricCalculator):
        name = "depth_median"

        def compute(self, prediction, ground_truth):
            # return a dict of metric_name -> float
            return {"median_absrel": float(np.median(np.abs(...)))}

Any number of calculators can be registered per task; the runner
aggregates their outputs into the per-sample metric dict. To *remove*
a built-in metric, either instantiate a new :class:`MetricSuite` with
a specific calculator list or unregister via
:func:`unregister_metric`.

The built-in calculators for every task we ship live in sibling
modules (``metrics/depth.py``, ``metrics/segmentation.py``, etc.). They
register themselves at import time; importing the :mod:`rpx_benchmark`
package triggers those registrations so the registry is always
populated.

Design notes
------------

- **One class per metric family** (not per metric key). A family
  shares preprocessing cost (e.g. masking valid depth pixels) across
  several related outputs like AbsRel + RMSE + δ<1.25.
- **Families are stateless** between frames — instantiate once, reuse.
  If a family needs per-run state (running variance, etc.) it should
  hold it internally and return the aggregate via a ``finalize()``
  hook the runner will call at the end. (Not implemented yet; add
  when needed.)
- **Metric output keys must be numeric** so the aggregator can average
  them. Non-numeric metadata belongs in the sample record, not the
  metric record.
"""

# Built-in calculator imports register themselves at import time via
# @register_metric decorators. Import-order matters: registry has to
# exist first (imported above), then the calculator modules.
from . import depth as _depth  # noqa: F401 — triggers registration
from . import depth_robotics as _depth_robotics  # noqa: F401
from . import detection as _detection  # noqa: F401
from . import grounding as _grounding  # noqa: F401
from . import keypoints as _keypoints  # noqa: F401
from . import nvs as _nvs  # noqa: F401
from . import pose as _pose  # noqa: F401
from . import segmentation as _segmentation  # noqa: F401
from . import sparse_depth as _sparse_depth  # noqa: F401
from . import tracking as _tracking  # noqa: F401
from . import video_depth as _video_depth  # noqa: F401  # registers VIDEO_DEPTH
from .registry import (
    MetricCalculator,
    MetricSuite,
    available_metrics,
    clear_registry,
    compute_metrics,
    get_calculators,
    register_metric,
    unregister_metric,
)

__all__ = [
    "MetricCalculator",
    "MetricSuite",
    "available_metrics",
    "clear_registry",
    "compute_metrics",
    "get_calculators",
    "register_metric",
    "unregister_metric",
]
