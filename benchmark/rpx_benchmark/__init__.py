"""RPX — choose and rank perception models for robot learning.

``rpx_benchmark`` is the RPX benchmark toolkit: **dataset loaders,
task-specific metrics, hardware-agnostic profiling, and an adapter
protocol for plugging your model in**. The toolkit deliberately
ships no models — bring your own.

Bring your model as a :class:`BenchmarkableModel` (easiest: wrap a
numpy callable via ``make_numpy_<task>_model(fn)``; for full control
implement your own :class:`InputAdapter` / :class:`OutputAdapter`
pair). Hand it to the relevant ``run_<task>(config)`` entry point.

See the top-level README and the online documentation for the full
tour.
"""

from . import exceptions, logging_utils, metrics  # noqa: F401 — side-effect registrations
from . import hub
from .adapters import (
    BatchedDepthBenchmarkModel,
    BatchedRelativePoseBenchmarkModel,
    BatchedSegmentationBenchmarkModel,
    BatchedTaskBenchmarkModel,
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
    make_numpy_tracking_model,
)
from .api import (
    ESD_WEIGHTS,
    BenchmarkModel,
    DepthGroundTruth,
    DepthPrediction,
    DetectionGroundTruth,
    DetectionPrediction,
    Difficulty,
    KeypointCorrespondenceGroundTruth,
    KeypointCorrespondencePrediction,
    NovelViewSynthesisGroundTruth,
    NovelViewSynthesisPrediction,
    Phase,
    RelativePoseGroundTruth,
    RelativePosePrediction,
    Sample,
    SegmentationGroundTruth,
    SegmentationPrediction,
    SparseDepthGroundTruth,
    SparseDepthPrediction,
    TaskType,
    Tracklet,
    TrackletGroundTruth,
    TrackletPrediction,
    VisualGroundingGroundTruth,
    VisualGroundingPrediction,
)
from .deployment import (
    DEFAULT_ERS_BUDGETS,
    DEFAULT_ERS_WEIGHTS,
    DeploymentReadinessReport,
    EmbodiedReadinessScore,
    ESDResult,
    StackGeometricCoherenceResult,
    StateTransitionRobustnessResult,
    TemporalStabilityResult,
    WeightedPhaseScore,
    compute_embodied_readiness,
    compute_esd,
    compute_sgc,
    compute_str,
    compute_temporal_stability_depth,
    compute_temporal_stability_seg,
    compute_weighted_phase_score,
    DeploymentReadinessResult,
    OperatingPoint,
    compute_drs,
    compute_sweep_drs,
)
from .determinism import RPX_SEED, deterministic, seed_all
from .evaluators import BenchmarkResult, MetricSuite
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
from .hub import (
    DEFAULT_REPO_ID,
    TASK_MODALITIES,
    download_split,
    fetch_manifest,
    load,
    mount,
)
from .loader import RPXDataset
from .logging_utils import configure_logging, get_logger
from .profiler import (
    EfficiencyMetadata,
    GPUSpec,
    LatencyProfiler,
    MemoryProfiler,
    REFERENCE_GPUS,
    RooflineBound,
    SystemCard,
    count_parameters,
    estimate_memory_traffic_gb,
    profile_model,
)
from .reports import format_markdown_summary, write_json
from .runner import BenchmarkRunner
from .tasks.detection import (
    ObjectDetectionRunConfig,
    run_object_detection,
    run_open_vocab_detection,
)
from .tasks.keypoint_matching import KeypointMatchingRunConfig, run_keypoint_matching
from .tasks.monocular_depth import MonocularDepthRunConfig, run_monocular_depth
from .tasks.novel_view_synthesis import (
    NovelViewSynthesisRunConfig,
    run_novel_view_synthesis,
)
from .tasks.relative_pose import RelativePoseRunConfig, run_relative_pose
from .tasks.segmentation import SegmentationRunConfig, run_segmentation
from .tasks.sparse_depth import SparseDepthRunConfig, run_sparse_depth
from .tasks.tracking import ObjectTrackingRunConfig, run_object_tracking
from .tasks.visual_grounding import VisualGroundingRunConfig, run_visual_grounding

# Optional subpackages that require extras. We import them best-effort
# so `import rpx_benchmark` works without pydantic or datasets
# installed; users who want strict validation or the HF datasets
# bridge install the corresponding extras and then use `rpx.schemas`
# / `rpx.data` directly.
try:
    from . import schemas  # noqa: F401
except ImportError:
    pass
try:
    from . import data  # noqa: F401
except ImportError:
    pass

__all__ = [
    # Enums
    "TaskType", "Phase", "Difficulty", "ESD_WEIGHTS",
    # Data contracts
    "Sample",
    "DepthGroundTruth", "DepthPrediction",
    "DetectionGroundTruth", "DetectionPrediction",
    "SegmentationGroundTruth", "SegmentationPrediction",
    "Tracklet", "TrackletGroundTruth", "TrackletPrediction",
    "VisualGroundingGroundTruth", "VisualGroundingPrediction",
    "RelativePoseGroundTruth", "RelativePosePrediction",
    "SparseDepthGroundTruth", "SparseDepthPrediction",
    "NovelViewSynthesisGroundTruth", "NovelViewSynthesisPrediction",
    "KeypointCorrespondenceGroundTruth", "KeypointCorrespondencePrediction",
    # Model base
    "BenchmarkModel",
    # Core harness
    "RPXDataset", "MetricSuite", "BenchmarkResult", "BenchmarkRunner",
    # Deployment-readiness
    "DeploymentReadinessReport",
    "EmbodiedReadinessScore",
    "DEFAULT_ERS_WEIGHTS", "DEFAULT_ERS_BUDGETS",
    "ESDResult", "StackGeometricCoherenceResult",
    "StateTransitionRobustnessResult", "TemporalStabilityResult",
    "WeightedPhaseScore",
    "compute_embodied_readiness",
    "compute_esd", "compute_sgc", "compute_str",
    "compute_temporal_stability_depth", "compute_temporal_stability_seg",
    "compute_weighted_phase_score",
    # Profiler
    "EfficiencyMetadata", "LatencyProfiler", "MemoryProfiler",
    "profile_model", "count_parameters",
    # Hub
    "hub", "DEFAULT_REPO_ID", "TASK_MODALITIES",
    "download_split", "fetch_manifest", "load", "mount",
    # Adapters (framework)
    "BenchmarkableModel", "InputAdapter", "OutputAdapter", "PreparedInput",
    "make_numpy_depth_model", "make_numpy_detection_model",
    "make_numpy_grounding_model", "make_numpy_keypoint_model",
    "make_numpy_mask_model", "make_numpy_nvs_model",
    "make_numpy_pose_model", "make_numpy_sparse_depth_model",
    "make_numpy_tracking_model",
    # Batched-dispatch wrappers (true GPU batching, not stock per-sample loop)
    "BatchedTaskBenchmarkModel",
    "BatchedDepthBenchmarkModel",
    "BatchedSegmentationBenchmarkModel",
    "BatchedRelativePoseBenchmarkModel",
    # Task runners
    "MonocularDepthRunConfig", "run_monocular_depth",
    "SegmentationRunConfig", "run_segmentation",
    "ObjectDetectionRunConfig", "run_object_detection", "run_open_vocab_detection",
    "VisualGroundingRunConfig", "run_visual_grounding",
    "RelativePoseRunConfig", "run_relative_pose",
    "KeypointMatchingRunConfig", "run_keypoint_matching",
    "SparseDepthRunConfig", "run_sparse_depth",
    "NovelViewSynthesisRunConfig", "run_novel_view_synthesis",
    "ObjectTrackingRunConfig", "run_object_tracking",
    "format_markdown_summary", "write_json",
    # Exceptions
    "exceptions",
    "RPXError", "ConfigError", "DatasetError", "ManifestError",
    "DownloadError", "ModelError", "AdapterError", "MetricError",
    # Logging
    "logging_utils", "get_logger", "configure_logging",
    # Determinism
    "seed_all", "deterministic", "RPX_SEED",
    # Plugin systems
    "metrics",
]
