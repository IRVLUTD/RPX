"""Task runners + registry.

Importing this package triggers every task module, which self-register
their :class:`TaskSpec` with the registry below. After import,
:func:`available_tasks` lists everything runnable.
"""

from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .detection import (
    OPEN_VOCAB_TASK_SPEC,
    ObjectDetectionRunConfig,
    run_object_detection,
    run_open_vocab_detection,
)
from .detection import (
    TASK_SPEC as OBJECT_DETECTION_SPEC,
)
from .keypoint_matching import (
    TASK_SPEC as KEYPOINT_MATCHING_SPEC,
)
from .keypoint_matching import (
    KeypointMatchingRunConfig,
    run_keypoint_matching,
)
from .monocular_depth import (
    TASK_SPEC as MONOCULAR_DEPTH_SPEC,
)
from .monocular_depth import (
    MonocularDepthRunConfig,
    run_monocular_depth,
)
from .registry import (
    TaskRunResult,
    TaskSpec,
    available_tasks,
    get_task_spec,
    iter_task_specs,
    register_task,
    unregister_task,
)
from .relative_pose import (
    TASK_SPEC as RELATIVE_POSE_SPEC,
)
from .relative_pose import (
    RelativePoseRunConfig,
    run_relative_pose,
)
from .segmentation import (
    TASK_SPEC as SEGMENTATION_SPEC,
)
from .segmentation import (
    SegmentationRunConfig,
    run_segmentation,
)
from .sparse_depth import (
    TASK_SPEC as SPARSE_DEPTH_SPEC,
)
from .sparse_depth import (
    SparseDepthRunConfig,
    run_sparse_depth,
)
from .tracking import (
    TASK_SPEC as OBJECT_TRACKING_SPEC,
)
from .tracking import (
    ObjectTrackingRunConfig,
    run_object_tracking,
)
from .visual_grounding import (
    TASK_SPEC as VISUAL_GROUNDING_SPEC,
)
from .visual_grounding import (
    VisualGroundingRunConfig,
    run_visual_grounding,
)

__all__ = [
    # Configs
    "MonocularDepthRunConfig",
    "SegmentationRunConfig",
    "ObjectDetectionRunConfig",
    "VisualGroundingRunConfig",
    "RelativePoseRunConfig",
    "KeypointMatchingRunConfig",
    "SparseDepthRunConfig",
    "ObjectTrackingRunConfig",
    # Runners
    "run_monocular_depth",
    "run_segmentation",
    "run_object_detection",
    "run_open_vocab_detection",
    "run_visual_grounding",
    "run_relative_pose",
    "run_keypoint_matching",
    "run_sparse_depth",
    "run_object_tracking",
    # Specs
    "MONOCULAR_DEPTH_SPEC",
    "SEGMENTATION_SPEC",
    "OBJECT_DETECTION_SPEC",
    "OPEN_VOCAB_TASK_SPEC",
    "VISUAL_GROUNDING_SPEC",
    "RELATIVE_POSE_SPEC",
    "KEYPOINT_MATCHING_SPEC",
    "SPARSE_DEPTH_SPEC",
    "OBJECT_TRACKING_SPEC",
    # Shared
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
