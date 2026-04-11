"""Core task enums, data contracts, and the base model interface.

This module is the stable public surface users bind against when they
plug a model into the benchmark. Everything here is either an
``Enum``, a plain ``@dataclass`` prediction/ground-truth container, or
the :class:`BenchmarkModel` abstract base that defines what a model
looks like to the runner.

The three pluggable systems that sit on top of these types are:

- :mod:`rpx_benchmark.adapters` — turns an arbitrary model into a
  :class:`BenchmarkModel`-shaped object via the
  ``InputAdapter / model / OutputAdapter`` contract.
- :mod:`rpx_benchmark.metrics` — task → calculator plugin registry.
- :mod:`rpx_benchmark.tasks.registry` — task → runner plugin registry.

Stability
---------

Enums and dataclasses in this module are append-only: adding new
tasks or new fields is fine; renaming or removing them is a breaking
change that requires a major version bump.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Sequence

import numpy as np


class TaskType(str, Enum):
    """Enumeration of every task the benchmark toolkit recognises.

    Members are plain strings so they serialise cleanly to JSON and
    can be used as dict keys for logging / table rows.

    Members
    -------
    MONOCULAR_DEPTH
        Dense metric depth from a single RGB frame.
    OBJECT_DETECTION
        Closed-vocabulary detection with category labels.
    OBJECT_SEGMENTATION
        Instance segmentation masks with per-pixel instance IDs.
    OBJECT_TRACKING
        Multi-object tracking with persistent track IDs.
    RELATIVE_CAMERA_POSE
        6-DoF pose of frame B relative to frame A.
    OPEN_VOCAB_DETECTION
        Detection conditioned on a free-text vocabulary.
    VISUAL_GROUNDING
        Referring expression → bounding box on the image.
    SPARSE_DEPTH
        Depth values at a sparse set of image locations only.
    NOVEL_VIEW_SYNTHESIS
        RGB synthesis from a held-out target pose.
    KEYPOINT_MATCHING
        Dense/sparse correspondences between two images.

    Examples
    --------
    >>> from rpx_benchmark.api import TaskType
    >>> TaskType.MONOCULAR_DEPTH.value
    'monocular_depth'
    >>> TaskType("monocular_depth") is TaskType.MONOCULAR_DEPTH
    True
    """

    MONOCULAR_DEPTH = "monocular_depth"
    OBJECT_DETECTION = "object_detection"
    OBJECT_SEGMENTATION = "object_segmentation"
    OBJECT_TRACKING = "object_tracking"
    RELATIVE_CAMERA_POSE = "relative_camera_pose"
    OPEN_VOCAB_DETECTION = "open_vocab_detection"
    VISUAL_GROUNDING = "visual_grounding"
    SPARSE_DEPTH = "sparse_depth"
    NOVEL_VIEW_SYNTHESIS = "novel_view_synthesis"
    KEYPOINT_MATCHING = "keypoint_matching"


class Phase(str, Enum):
    """Capture phases of the three-phase RPX reconfiguration protocol.

    Every scene is recorded in three phases so the benchmark can
    attribute performance changes to scene state rather than to
    lighting / viewpoint / camera identity.

    Members
    -------
    CLUTTER
        Initial dense object arrangement; significant inter-object
        occlusion.
    INTERACTION
        Human operator grasps and moves objects. Introduces
        hand-object contact and transient occlusion.
    CLEAN
        Same objects re-organised sparsely. Serves as a within-scene
        control for the other two phases.
    """

    CLUTTER = "clutter"
    INTERACTION = "interaction"
    CLEAN = "clean"


class Difficulty(str, Enum):
    """Effort-Stratified Difficulty (ESD) split label.

    ESD splits are derived per ``(scene, phase)`` from the
    annotation-effort signal described in paper §4. See
    :mod:`rpx_benchmark.deployment` for the scoring details.

    Members
    -------
    EASY
        Few annotation iterations, low occlusion, stable visibility.
    MEDIUM
    HARD
        Many annotation iterations, dense occlusion, high depth-
        invalid fraction, high jerk.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


#: Weights used for the ESD-weighted phase score
#: :math:`S_p = 0.25 E + 0.35 M + 0.40 H`.
ESD_WEIGHTS: Dict[Difficulty, float] = {
    Difficulty.EASY: 0.25,
    Difficulty.MEDIUM: 0.35,
    Difficulty.HARD: 0.40,
}


@dataclass
class DepthGroundTruth:
    depth_map: np.ndarray  # H x W float32 meters


@dataclass
class DetectionGroundTruth:
    boxes: np.ndarray  # N x 4 float (x1, y1, x2, y2)
    labels: Sequence[str]  # class names aligned with boxes


@dataclass
class SegmentationGroundTruth:
    mask: np.ndarray  # H x W int (class ids)


@dataclass
class Tracklet:
    track_id: str
    boxes: np.ndarray  # T x 4 float (x1, y1, x2, y2)
    scores: np.ndarray | None = None  # Optional T float


@dataclass
class TrackletGroundTruth:
    tracks: Sequence[Tracklet]


@dataclass
class VisualGroundingGroundTruth:
    text: str
    boxes: np.ndarray  # N x 4 float
    labels: Sequence[str] | None = None


@dataclass
class RelativePoseGroundTruth:
    rotation: np.ndarray  # 3 x 3 or 4-array quaternion
    translation: np.ndarray  # 3-vector meters


@dataclass
class SparseDepthGroundTruth:
    coordinates: np.ndarray  # N x 2 (x, y) pixel coords
    depths: np.ndarray  # N float depths (m)


@dataclass
class NovelViewSynthesisGroundTruth:
    rgb: np.ndarray  # H x W x 3 uint8
    camera_pose: Dict[str, Any] | None = None


@dataclass
class KeypointCorrespondenceGroundTruth:
    points0: np.ndarray  # N x 2 float
    points1: np.ndarray  # N x 2 float
    visibility: np.ndarray | None = None  # optional mask


@dataclass
class Sample:
    """One input unit handed by :class:`RPXDataset` to a model.

    Samples are produced by the loader and consumed by
    ``BenchmarkModel.predict``. Every field is deliberately simple
    (numpy arrays, enums, plain dicts) so models and adapters don't
    need to know anything about the on-disk dataset format.

    Parameters
    ----------
    id : str
        Unique identifier of the form ``{scene}_{phase}_{frame}``.
        Used for joining per-sample metrics back to scenes / phases.
    rgb : np.ndarray
        H × W × 3 uint8 RGB image in row-major order.
    ground_truth : Any
        Task-specific GroundTruth dataclass (e.g.
        :class:`DepthGroundTruth`). The concrete type is determined
        by :attr:`RPXDataset.task`.
    metadata : dict, optional
        Free-form metadata the loader can attach; conventionally
        holds fisheye images, secondary RGB frames for pair tasks,
        and any label paths that do not fit into the ground-truth
        dataclass. Consumers should treat unknown keys as opaque.
    phase : Phase, optional
        Capture phase the frame belongs to. Required for ESD-weighted
        phase scoring.
    difficulty : Difficulty, optional
        ESD difficulty label of the ``(scene, phase)`` this sample
        belongs to.
    camera_pose : np.ndarray, optional
        4 × 4 float64 SE(3) matrix (camera → world) sourced from the
        T265 tracker. Used for the temporal-stability metric.
    """

    id: str
    rgb: np.ndarray
    ground_truth: Any
    metadata: Dict[str, Any] | None = None
    phase: Phase | None = None
    difficulty: Difficulty | None = None
    camera_pose: np.ndarray | None = None


@dataclass
class DepthPrediction:
    depth_map: np.ndarray  # H x W float32


@dataclass
class DetectionPrediction:
    boxes: np.ndarray  # N x 4 float
    scores: np.ndarray  # N float
    labels: Sequence[str]


@dataclass
class SegmentationPrediction:
    mask: np.ndarray  # H x W int (class ids)


@dataclass
class TrackletPrediction:
    tracks: Sequence[Tracklet]


@dataclass
class VisualGroundingPrediction:
    boxes: np.ndarray  # N x 4 float
    scores: np.ndarray  # N float
    labels: Sequence[str] | None = None


@dataclass
class RelativePosePrediction:
    rotation: np.ndarray  # 3 x 3 or quaternion
    translation: np.ndarray  # 3-vector meters


@dataclass
class SparseDepthPrediction:
    coordinates: np.ndarray  # N x 2
    depths: np.ndarray  # N


@dataclass
class NovelViewSynthesisPrediction:
    rgb: np.ndarray  # H x W x 3 uint8


@dataclass
class KeypointCorrespondencePrediction:
    points0: np.ndarray  # N x 2
    points1: np.ndarray  # N x 2
    scores: np.ndarray | None = None


class BenchmarkModel(ABC):
    """Abstract base class every RPX-compatible model must implement.

    In practice, most users should **not** subclass this directly —
    instead compose a :class:`rpx_benchmark.adapters.BenchmarkableModel`
    from an input adapter, a model callable, and an output adapter.
    ``BenchmarkableModel`` already implements :meth:`predict` and
    :meth:`setup` correctly for you.

    Subclass only when you need complete control over how samples
    are routed to your model (e.g. true minibatching across GPU
    devices).

    Attributes
    ----------
    task : TaskType
        The task this model solves. Must be set by subclasses (either
        at class level or in ``__init__``). The runner checks that
        ``model.task == dataset.task`` before running.

    Examples
    --------
    Minimal subclass::

        class MyDepth(BenchmarkModel):
            task = TaskType.MONOCULAR_DEPTH

            def setup(self):
                self.net = load_my_checkpoint()

            def predict(self, batch):
                return [
                    DepthPrediction(depth_map=self.net(s.rgb))
                    for s in batch
                ]

    Composed via :class:`BenchmarkableModel`::

        bm = rpx.BenchmarkableModel(
            task=TaskType.MONOCULAR_DEPTH,
            input_adapter=MyInputAdapter(),
            model=my_nn_module,
            output_adapter=MyOutputAdapter(),
            name="my_model",
        )
    """

    task: TaskType

    @abstractmethod
    def setup(self) -> None:
        """Load checkpoints, warm CUDA, and do any other one-time init.

        The runner calls this exactly once before iterating the
        dataset, unless ``BenchmarkRunner(call_setup=False)`` was
        passed — in which case the caller is responsible.
        """

    @abstractmethod
    def predict(self, batch: Sequence[Sample]) -> Sequence[Any]:
        """Run inference on a batch of samples.

        Parameters
        ----------
        batch : sequence of Sample
            One or more samples. Length equals ``dataset.batch_size``
            except possibly for the final tail batch.

        Returns
        -------
        sequence
            One task-specific Prediction dataclass per input sample,
            in the same order. The prediction dataclass must match
            what :class:`MetricSuite` expects for this task.

        Raises
        ------
        ModelError
            (By convention) when a sample cannot be processed. The
            runner surfaces it as a clean error rather than a stack
            trace.
        """


def _check_shape(arr: np.ndarray, dims: int, name: str) -> None:
    """Raise :class:`ModelError` if ``arr`` does not have exactly ``dims`` axes."""
    if arr.ndim != dims:
        from .exceptions import ModelError
        raise ModelError(
            f"{name} must have {dims} dims; got {arr.ndim}",
            hint=f"Check the shape your model returns for {name!r}.",
        )


def validate_prediction(task: TaskType, prediction: Any, sample: Sample | None = None) -> None:
    """Validate a Prediction dataclass's shape and type for a given task.

    Parameters
    ----------
    task : TaskType
        Task the runner is evaluating.
    prediction : Any
        Prediction dataclass the model just returned.
    sample : Sample, optional
        The sample the prediction was produced for; used for shape
        cross-checks (e.g. segmentation mask vs RGB size).

    Raises
    ------
    ModelError
        If the prediction is the wrong type or the wrong shape for the
        task.
    """
    from .exceptions import ModelError

    def _type_error(expected: str) -> "ModelError":
        return ModelError(
            f"{task.value} models must return {expected}, got "
            f"{type(prediction).__name__}",
            hint=f"Check your model's predict() return type for {task.value}.",
        )

    if task == TaskType.MONOCULAR_DEPTH:
        if not isinstance(prediction, DepthPrediction):
            raise _type_error("DepthPrediction")
        _check_shape(prediction.depth_map, 2, "Depth map")
        return

    if task == TaskType.OBJECT_DETECTION:
        if not isinstance(prediction, DetectionPrediction):
            raise _type_error("DetectionPrediction")
        boxes = prediction.boxes
        _check_shape(boxes, 2, "Detection boxes")
        if boxes.shape[1] != 4:
            raise ModelError(
                f"Detection boxes must have shape [N, 4]; got {boxes.shape}",
            )
        if len(prediction.scores) != len(boxes) or len(prediction.labels) != len(boxes):
            raise ModelError(
                "DetectionPrediction boxes, scores, and labels must all "
                "have the same length.",
                details={
                    "n_boxes": len(boxes),
                    "n_scores": len(prediction.scores),
                    "n_labels": len(prediction.labels),
                },
            )
        return

    if task == TaskType.OBJECT_SEGMENTATION:
        if not isinstance(prediction, SegmentationPrediction):
            raise _type_error("SegmentationPrediction")
        _check_shape(prediction.mask, 2, "Segmentation mask")
        if sample and prediction.mask.shape != sample.rgb.shape[:2]:
            raise ModelError(
                f"Segmentation mask shape {prediction.mask.shape} must "
                f"match the RGB spatial size {sample.rgb.shape[:2]}",
                hint="The OutputAdapter should resize masks to the sample's "
                     "original H x W before returning.",
            )
        return

    if task == TaskType.OBJECT_TRACKING:
        if not isinstance(prediction, TrackletPrediction):
            raise _type_error("TrackletPrediction")
        return

    if task == TaskType.VISUAL_GROUNDING:
        if not isinstance(prediction, VisualGroundingPrediction):
            raise _type_error("VisualGroundingPrediction")
        return

    if task == TaskType.RELATIVE_CAMERA_POSE:
        if not isinstance(prediction, RelativePosePrediction):
            raise _type_error("RelativePosePrediction")
        return

    if task == TaskType.SPARSE_DEPTH:
        if not isinstance(prediction, SparseDepthPrediction):
            raise _type_error("SparseDepthPrediction")
        return

    if task == TaskType.NOVEL_VIEW_SYNTHESIS:
        if not isinstance(prediction, NovelViewSynthesisPrediction):
            raise _type_error("NovelViewSynthesisPrediction")
        return

    if task == TaskType.KEYPOINT_MATCHING:
        if not isinstance(prediction, KeypointCorrespondencePrediction):
            raise _type_error("KeypointCorrespondencePrediction")
        return

    raise ModelError(f"Unsupported task: {task}")
