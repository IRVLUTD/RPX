"""Task-level entrypoints for RPX benchmark pipelines.

Importing this package triggers every task module below, which in
turn self-register their :class:`TaskSpec` with
:mod:`rpx_benchmark.tasks.registry`. After import,
:func:`available_tasks` lists everything the CLI can run.

Task coverage status:

- Runnable end-to-end: monocular_depth, object_segmentation,
  object_detection + open_vocab_detection, visual_grounding,
  relative_camera_pose, keypoint_matching, sparse_depth,
  novel_view_synthesis, object_tracking (per-frame MOTA+IDF1).
- Scene-level HOTA for object_tracking lands once the M3
  temporal-metric generalisation is in.
"""

from .monocular_depth import (
    MonocularDepthRunConfig,
    TASK_SPEC as MONOCULAR_DEPTH_SPEC,
    run_monocular_depth,
)
from .segmentation import (
    SegmentationRunConfig,
    TASK_SPEC as SEGMENTATION_SPEC,
    run_segmentation,
)
from .detection import (
    ObjectDetectionRunConfig,
    OPEN_VOCAB_TASK_SPEC,
    TASK_SPEC as OBJECT_DETECTION_SPEC,
    run_object_detection,
    run_open_vocab_detection,
)
from .visual_grounding import (
    TASK_SPEC as VISUAL_GROUNDING_SPEC,
    VisualGroundingRunConfig,
    run_visual_grounding,
)
from .relative_pose import (
    RelativePoseRunConfig,
    TASK_SPEC as RELATIVE_POSE_SPEC,
    run_relative_pose,
)
from .keypoint_matching import (
    KeypointMatchingRunConfig,
    TASK_SPEC as KEYPOINT_MATCHING_SPEC,
    run_keypoint_matching,
)
from .sparse_depth import (
    SparseDepthRunConfig,
    TASK_SPEC as SPARSE_DEPTH_SPEC,
    run_sparse_depth,
)
from .novel_view_synthesis import (
    NovelViewSynthesisRunConfig,
    TASK_SPEC as NOVEL_VIEW_SYNTHESIS_SPEC,
    run_novel_view_synthesis,
)
from .tracking import (
    ObjectTrackingRunConfig,
    TASK_SPEC as OBJECT_TRACKING_SPEC,
    run_object_tracking,
)
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import (
    TaskRunResult,
    TaskSpec,
    available_tasks,
    get_task_spec,
    iter_task_specs,
    register_task,
    unregister_task,
)

__all__ = [
    # Task runners — configs
    "MonocularDepthRunConfig",
    "SegmentationRunConfig",
    "ObjectDetectionRunConfig",
    "VisualGroundingRunConfig",
    "RelativePoseRunConfig",
    "KeypointMatchingRunConfig",
    "SparseDepthRunConfig",
    "NovelViewSynthesisRunConfig",
    "ObjectTrackingRunConfig",
    # Task runners — entry-points
    "run_monocular_depth",
    "run_segmentation",
    "run_object_detection",
    "run_open_vocab_detection",
    "run_visual_grounding",
    "run_relative_pose",
    "run_keypoint_matching",
    "run_sparse_depth",
    "run_novel_view_synthesis",
    "run_object_tracking",
    # Task specs (exposed so callers can introspect)
    "MONOCULAR_DEPTH_SPEC",
    "SEGMENTATION_SPEC",
    "OBJECT_DETECTION_SPEC",
    "OPEN_VOCAB_TASK_SPEC",
    "VISUAL_GROUNDING_SPEC",
    "RELATIVE_POSE_SPEC",
    "KEYPOINT_MATCHING_SPEC",
    "SPARSE_DEPTH_SPEC",
    "NOVEL_VIEW_SYNTHESIS_SPEC",
    "OBJECT_TRACKING_SPEC",
    # Shared helpers
    "TaskRunConfig",
    "PipelineResult",
    "run_pipeline",
    # Registry
    "TaskSpec",
    "TaskRunResult",
    "register_task",
    "unregister_task",
    "get_task_spec",
    "available_tasks",
    "iter_task_specs",
]
