"""RPX — choose and rank perception models for robot learning.

``rpx_benchmark`` is the reference toolkit for the RPX benchmark: a
unified real-world RGB-D evaluation suite for the models actually
deployed inside robot learning stacks. Bring your model (any
HuggingFace checkpoint, numpy callable, or custom torch stack) and
the toolkit handles dataset download, splits, metrics, reports, and
ESD-weighted deployment-readiness scoring.

See the top-level README and the online documentation for getting
started, the adapter framework, and the extension guides.
"""

from .api import (
    Difficulty,
    ESD_WEIGHTS,
    Phase,
    TaskType,
    Sample,
    DepthGroundTruth,
    DetectionGroundTruth,
    SegmentationGroundTruth,
    DepthPrediction,
    DetectionPrediction,
    SegmentationPrediction,
    BenchmarkModel,
    Tracklet,
    TrackletGroundTruth,
    TrackletPrediction,
    VisualGroundingGroundTruth,
    VisualGroundingPrediction,
    RelativePoseGroundTruth,
    RelativePosePrediction,
    SparseDepthGroundTruth,
    SparseDepthPrediction,
    NovelViewSynthesisGroundTruth,
    NovelViewSynthesisPrediction,
    KeypointCorrespondenceGroundTruth,
    KeypointCorrespondencePrediction,
)
from .loader import RPXDataset
from .evaluators import MetricSuite, BenchmarkResult
from .runner import BenchmarkRunner
from .deployment import (
    DeploymentReadinessReport,
    ESDResult,
    StackGeometricCoherenceResult,
    StateTransitionRobustnessResult,
    TemporalStabilityResult,
    WeightedPhaseScore,
    compute_esd,
    compute_sgc,
    compute_str,
    compute_temporal_stability_depth,
    compute_temporal_stability_seg,
    compute_weighted_phase_score,
)
from .profiler import EfficiencyMetadata, profile_model, count_parameters
from . import hub
from .hub import (
    DEFAULT_REPO_ID,
    TASK_MODALITIES,
    download_split,
    fetch_manifest,
    load,
    mount,
)
from . import adapters, exceptions, logging_utils, metrics, models  # noqa: F401 — triggers side-effect registrations
from .exceptions import (
    AdapterError,
    ConfigError,
    DatasetError,
    DownloadError,
    ManifestError,
    MetricError,
    ModelError,
    RPXError,
)
from .banner import show_banner
from .logging_utils import configure_logging, get_logger
from .adapters import (
    BenchmarkableModel,
    InputAdapter,
    OutputAdapter,
    PreparedInput,
    make_numpy_depth_model,
    make_numpy_detection_model,
    make_numpy_grounding_model,
    make_numpy_keypoint_model,
    make_numpy_mask_model,
    make_numpy_nvs_model,
    make_numpy_pose_model,
    make_numpy_sparse_depth_model,
)
from .adapters.depth_hf import make_hf_depth_model
from .adapters.seg_hf import make_hf_instance_seg_model
from .models.registry import available_models, get_factory, resolve
from .tasks.monocular_depth import MonocularDepthRunConfig, run_monocular_depth
from .tasks.segmentation import SegmentationRunConfig, run_segmentation
from .tasks.detection import (
    ObjectDetectionRunConfig,
    run_object_detection,
    run_open_vocab_detection,
)
from .tasks.visual_grounding import VisualGroundingRunConfig, run_visual_grounding
from .tasks.relative_pose import RelativePoseRunConfig, run_relative_pose
from .tasks.keypoint_matching import KeypointMatchingRunConfig, run_keypoint_matching
from .tasks.sparse_depth import SparseDepthRunConfig, run_sparse_depth
from .tasks.novel_view_synthesis import (
    NovelViewSynthesisRunConfig,
    run_novel_view_synthesis,
)
from .reports import format_markdown_summary, write_json

__all__ = [
    # Enums
    "TaskType",
    "Phase",
    "Difficulty",
    "ESD_WEIGHTS",
    # Data contracts
    "Sample",
    "DepthGroundTruth",
    "DetectionGroundTruth",
    "SegmentationGroundTruth",
    "DepthPrediction",
    "DetectionPrediction",
    "SegmentationPrediction",
    "Tracklet",
    "TrackletGroundTruth",
    "TrackletPrediction",
    "VisualGroundingGroundTruth",
    "VisualGroundingPrediction",
    "RelativePoseGroundTruth",
    "RelativePosePrediction",
    "SparseDepthGroundTruth",
    "SparseDepthPrediction",
    "NovelViewSynthesisGroundTruth",
    "NovelViewSynthesisPrediction",
    "KeypointCorrespondenceGroundTruth",
    "KeypointCorrespondencePrediction",
    # Model base
    "BenchmarkModel",
    # Core harness
    "RPXDataset",
    "MetricSuite",
    "BenchmarkResult",
    "BenchmarkRunner",
    # Deployment-readiness
    "DeploymentReadinessReport",
    "ESDResult",
    "StackGeometricCoherenceResult",
    "StateTransitionRobustnessResult",
    "TemporalStabilityResult",
    "WeightedPhaseScore",
    "compute_esd",
    "compute_sgc",
    "compute_str",
    "compute_temporal_stability_depth",
    "compute_temporal_stability_seg",
    "compute_weighted_phase_score",
    # Profiler
    "EfficiencyMetadata",
    "profile_model",
    "count_parameters",
    # Hub
    "hub",
    "DEFAULT_REPO_ID",
    "TASK_MODALITIES",
    "download_split",
    "fetch_manifest",
    "load",
    "mount",
    # Adapters + model registry
    "adapters",
    "BenchmarkableModel",
    "InputAdapter",
    "OutputAdapter",
    "PreparedInput",
    "make_numpy_depth_model",
    "make_numpy_detection_model",
    "make_numpy_grounding_model",
    "make_numpy_keypoint_model",
    "make_numpy_mask_model",
    "make_numpy_nvs_model",
    "make_numpy_pose_model",
    "make_numpy_sparse_depth_model",
    "make_hf_depth_model",
    "make_hf_instance_seg_model",
    "available_models",
    "get_factory",
    "resolve",
    # Task runners + reporting
    "MonocularDepthRunConfig",
    "run_monocular_depth",
    "SegmentationRunConfig",
    "run_segmentation",
    "ObjectDetectionRunConfig",
    "run_object_detection",
    "run_open_vocab_detection",
    "VisualGroundingRunConfig",
    "run_visual_grounding",
    "RelativePoseRunConfig",
    "run_relative_pose",
    "KeypointMatchingRunConfig",
    "run_keypoint_matching",
    "SparseDepthRunConfig",
    "run_sparse_depth",
    "NovelViewSynthesisRunConfig",
    "run_novel_view_synthesis",
    "format_markdown_summary",
    "write_json",
    # Exceptions
    "exceptions",
    "RPXError",
    "ConfigError",
    "DatasetError",
    "ManifestError",
    "DownloadError",
    "ModelError",
    "AdapterError",
    "MetricError",
    # Logging
    "logging_utils",
    "get_logger",
    "configure_logging",
    # Terminal banner
    "show_banner",
    # Metric plugin system
    "metrics",
]
