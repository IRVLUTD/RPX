"""Core types for the RPX adapter framework.

::

    Sample ─► InputAdapter.prepare ─► PreparedInput(payload, context)
                                           │
                                           ▼
                                       model(payload)
                                           │
                                           ▼
    Sample, context, model_output ─► OutputAdapter.finalize ─► Prediction

Users extending RPX for their own model only need to supply the *model*
and pick a matching pair of adapters. The adapters ship with the
library for common families (HuggingFace transformers, UniDepth,
Metric3D, raw numpy callables).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..api import (
    BenchmarkModel,
    DepthPrediction,
    DetectionPrediction,
    KeypointCorrespondencePrediction,
    NovelViewSynthesisPrediction,
    RelativePosePrediction,
    Sample,
    SegmentationPrediction,
    SparseDepthPrediction,
    TaskType,
    Tracklet,
    TrackletPrediction,
    VideoDepthPrediction,
    VideoSample,
    VisualGroundingPrediction,
)
from ..exceptions import AdapterError

# --------------------------------------------------------------------------- #
# Payload container
# --------------------------------------------------------------------------- #


@dataclass
class PreparedInput:
    """Everything a model needs for one sample, plus context for post-processing.

    ``payload`` is whatever the model's forward call accepts. If it is a
    ``dict``, the default invoker calls ``model(**payload)``; otherwise
    ``model(payload)``.

    ``context`` is a free-form dict the output adapter receives back. Use
    it to stash things like target image size, original intrinsics, or
    any preprocessing metadata the postprocessing step needs.
    """

    payload: Any
    context: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Protocol types
# --------------------------------------------------------------------------- #


@runtime_checkable
class InputAdapter(Protocol):
    """Sample → model-ready payload."""

    def setup(self) -> None:  # pragma: no cover - optional hook
        """Optional one-time setup (e.g., build a processor on first use)."""

    def prepare(self, sample: Sample) -> PreparedInput: ...


@runtime_checkable
class OutputAdapter(Protocol):
    """Model output → RPX prediction object."""

    def setup(self) -> None:  # pragma: no cover - optional hook
        """Optional one-time setup."""

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> Any: ...


ModelInvoker = Callable[[Any, Any], Any]
"""Given (model, payload) return the raw model output.

The default implementation handles the two common cases: dict payloads
become ``model(**payload)``, anything else becomes ``model(payload)``.
Override when the model needs non-standard invocation (e.g., a method
other than ``__call__``).
"""


def default_invoker(model: Any, payload: Any) -> Any:
    """Call ``model`` with ``payload``. Uses torch no_grad if available."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return _dispatch(model, payload)

    import torch

    with torch.no_grad():
        return _dispatch(model, payload)


def _dispatch(model: Any, payload: Any) -> Any:
    if isinstance(payload, dict):
        return model(**payload)
    return model(payload)


# --------------------------------------------------------------------------- #
# BenchmarkableModel
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkableModel(BenchmarkModel):
    """Compose an input adapter, a model, and an output adapter.

    This is the canonical way to plug a model into the RPX benchmark
    harness. The ``BenchmarkRunner`` only ever sees the
    :class:`BenchmarkModel` contract (``setup``, ``predict``); all the
    model-family-specific logic lives in the adapters so the harness
    stays task-agnostic.

    Example — wrap a plain numpy depth callable::

        import numpy as np
        import rpx_benchmark as rpx

        def my_depth(rgb: np.ndarray) -> np.ndarray:
            return ...           # H x W float32, metres

        bm = rpx.make_numpy_depth_model(my_depth, name="my_model")
        # Hand `bm` to a task runner or the BenchmarkRunner directly.

    Example — wrap a plain numpy callable::

        def my_depth(rgb: np.ndarray) -> np.ndarray:
            return some_depth_in_metres

        bm = make_numpy_depth_model(my_depth)
    """

    task: TaskType
    input_adapter: InputAdapter
    model: Any
    output_adapter: OutputAdapter
    invoker: ModelInvoker = field(default=default_invoker)
    name: str = "benchmarkable_model"
    setup_hook: Optional[Callable[[], None]] = None

    def setup(self) -> None:
        if hasattr(self.input_adapter, "setup"):
            self.input_adapter.setup()
        if hasattr(self.output_adapter, "setup"):
            self.output_adapter.setup()
        if self.setup_hook is not None:
            self.setup_hook()

    def predict(self, batch: Sequence[Sample]) -> Sequence[Any]:
        preds = []
        for sample in batch:
            prepared = self.input_adapter.prepare(sample)
            model_out = self.invoker(self.model, prepared.payload)
            pred = self.output_adapter.finalize(model_out, prepared.context, sample)
            preds.append(pred)
        return preds


# --------------------------------------------------------------------------- #
# Convenience factory: numpy-in / numpy-out depth model
# --------------------------------------------------------------------------- #


class _NumpyDepthInput:
    """Hand the model the raw ``(H, W, 3) uint8`` array untouched."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        return PreparedInput(payload=rgb, context={"target_hw": rgb.shape[:2]})


class _NumpyDepthOutput:
    """Wrap a raw ``(H_pred, W_pred) float`` array into ``DepthPrediction``.

    If the model returns a shape different from ``target_hw``, the result
    is bilinearly resized (via PIL) so ``BenchmarkRunner`` can compare
    pixel-for-pixel against the GT.
    """

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> DepthPrediction:
        depth = np.asarray(model_output, dtype=np.float32)
        if depth.ndim != 2:
            raise AdapterError(
                f"numpy depth model must return a 2-D array; got shape {depth.shape}",
                hint="Your depth callable must return a (H, W) float array. "
                "A common mistake is returning (1, H, W) or (H, W, 1).",
            )
        target_hw = context.get("target_hw", depth.shape)
        if depth.shape != target_hw:
            from PIL import Image

            pil = Image.fromarray(depth, mode="F")
            pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            depth = np.asarray(pil, dtype=np.float32)
        return DepthPrediction(depth_map=depth)


# --------------------------------------------------------------------------- #
# Convenience factory: numpy-in / numpy-out video-depth model
# --------------------------------------------------------------------------- #


class _NumpyVideoDepthInput:
    """Hand the model the raw ``(T, H, W, 3) uint8`` clip untouched."""

    def prepare(self, sample: VideoSample) -> PreparedInput:
        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        return PreparedInput(
            payload=rgb_seq,
            context={"target_thw": rgb_seq.shape[:3]},
        )


class _NumpyVideoDepthOutput:
    """Wrap a raw ``(T, H_pred, W_pred) float`` array into
    :class:`VideoDepthPrediction`.

    If the model returns a shape different from ``target_thw``, the
    per-frame depth maps are bilinearly resized (via PIL) so the
    runner can compare pixel-for-pixel against the GT clip. T must
    match — a model that drops frames is a contract violation, not
    something this output adapter is allowed to paper over.
    """

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: VideoSample,
    ) -> VideoDepthPrediction:
        depth_seq = np.asarray(model_output, dtype=np.float32)
        if depth_seq.ndim == 4 and depth_seq.shape[1] == 1:
            # (T, 1, H, W) → (T, H, W); a common model output shape.
            depth_seq = depth_seq.squeeze(1)
        if depth_seq.ndim != 3:
            raise AdapterError(
                f"numpy video-depth model must return a 3-D (T, H, W) "
                f"array; got shape {depth_seq.shape}",
                hint="Your video-depth callable must return (T, H, W) "
                "float32. A common mistake is returning (T, 1, H, W) "
                "without squeezing the channel dim.",
            )
        target_thw = context.get("target_thw", depth_seq.shape)
        target_t, target_h, target_w = target_thw
        if depth_seq.shape[0] != target_t:
            raise AdapterError(
                f"numpy video-depth model returned {depth_seq.shape[0]} "
                f"frames; clip has {target_t}. The model must emit one "
                "depth map per input frame — runner / metric calculators "
                "assume aligned T.",
            )
        if depth_seq.shape[1:] != (target_h, target_w):
            from PIL import Image

            resized = np.empty(
                (target_t, target_h, target_w), dtype=np.float32
            )
            for t in range(target_t):
                pil = Image.fromarray(depth_seq[t], mode="F")
                pil = pil.resize((target_w, target_h), Image.BILINEAR)
                resized[t] = np.asarray(pil, dtype=np.float32)
            depth_seq = resized
        return VideoDepthPrediction(depth_map_seq=depth_seq)


def make_numpy_video_depth_model(
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    name: str = "numpy_video_depth_model",
    depth_output_kind: str = "metric",
) -> BenchmarkableModel:
    """Wrap a plain numpy video-depth callable as a
    :class:`BenchmarkableModel`.

    The callable must accept a ``(T, H, W, 3) uint8`` RGB clip and
    return a ``(T, H', W') float`` per-frame depth sequence (metres
    if metric; up-to-scale if relative — the runner applies per-clip
    ``(s, t)`` alignment for relative models). If ``(H', W') !=
    (H, W)``, the output is bilinearly resized per frame to match.
    ``T`` must match the input clip length.

    Parameters
    ----------
    fn : callable
        Signature: ``fn(rgb_seq_uint8) -> depth_seq_float``.
    name : str
        Display name used in logs and reports.
    depth_output_kind : str
        ``"metric"`` (default) — model emits metres; runner skips
        alignment. ``"relative"`` — runner applies per-clip
        ``(s, t)`` scale-and-shift alignment before metrics.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_clip_depth(rgb_seq):
    ...     T, H, W, _ = rgb_seq.shape
    ...     return np.full((T, H, W), 2.0, dtype=np.float32)
    >>> bm = rpx.make_numpy_video_depth_model(my_clip_depth, name="mine")
    >>> bm.task is rpx.TaskType.VIDEO_DEPTH
    True
    """
    model = BenchmarkableModel(
        task=TaskType.VIDEO_DEPTH,
        input_adapter=_NumpyVideoDepthInput(),
        model=fn,
        output_adapter=_NumpyVideoDepthOutput(),
        invoker=lambda model, payload: model(payload),
        name=name,
    )
    # Propagate depth_output_kind to the runner's alignment dispatch.
    model.depth_output_kind = depth_output_kind
    return model


def make_numpy_depth_model(
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    name: str = "numpy_depth_model",
    depth_output_kind: str = "metric",
    native_alignment: str | None = None,
) -> BenchmarkableModel:
    """Wrap a plain numpy depth callable as a :class:`BenchmarkableModel`.

    The callable must accept a ``(H, W, 3) uint8`` RGB image and return a
    ``(H', W') float`` depth map. If ``depth_output_kind="metric"`` the
    values are metres. Relative outputs remain raw until the runner fits
    one pooled alignment per ``(scene, phase)`` cell. If ``(H', W') !=
    (H, W)``, the output is bilinearly resized to match the ground truth.

    Parameters
    ----------
    fn : callable
        The depth function. Signature: ``fn(rgb_uint8) -> depth_float``.
    name : str
        Display name used in logs and reports.
    depth_output_kind : str
        ``"metric"`` (default) or ``"relative"``.
    native_alignment : str, optional
        Alignment used for relative output. Defaults to ``"ls_affine"``
        for relative models and ``"none"`` for metric models.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_depth(rgb):
    ...     return np.full(rgb.shape[:2], 2.0, dtype=np.float32)
    >>> bm = rpx.make_numpy_depth_model(my_depth, name="mine")
    >>> bm.task is rpx.TaskType.MONOCULAR_DEPTH
    True
    """
    if depth_output_kind not in {"metric", "relative"}:
        raise AdapterError(
            f"depth_output_kind must be 'metric' or 'relative', got {depth_output_kind!r}",
        )
    model = BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=_NumpyDepthInput(),
        model=fn,
        output_adapter=_NumpyDepthOutput(),
        invoker=lambda model, payload: model(payload),
        name=name,
    )
    model.depth_output_kind = depth_output_kind
    model.native_alignment = native_alignment or (
        "ls_affine" if depth_output_kind == "relative" else "none"
    )
    return model


# --------------------------------------------------------------------------- #
# Convenience factory: numpy-in / int-mask-out segmentation model
# --------------------------------------------------------------------------- #


class _NumpyMaskInput:
    """Hand the model the raw ``(H, W, 3) uint8`` array untouched."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        return PreparedInput(payload=rgb, context={"target_hw": rgb.shape[:2]})


class _NumpyMaskOutput:
    """Wrap a raw ``(H_pred, W_pred) int`` array into ``SegmentationPrediction``.

    If the model returns a shape different from ``target_hw``, the result
    is nearest-neighbour resized (masks must keep integer IDs — bilinear
    would introduce invalid fractional values).
    """

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> SegmentationPrediction:
        mask = np.asarray(model_output)
        if mask.ndim != 2:
            raise AdapterError(
                f"numpy mask model must return a 2-D array; got shape {mask.shape}",
                hint="Your mask callable must return a (H, W) int array. "
                "A common mistake is returning (H, W, C) per-class logits.",
            )
        mask = mask.astype(np.int32, copy=False)
        target_hw = context.get("target_hw", mask.shape)
        if mask.shape != target_hw:
            from PIL import Image

            pil = Image.fromarray(mask.astype(np.int32), mode="I")
            pil = pil.resize((target_hw[1], target_hw[0]), Image.NEAREST)
            mask = np.asarray(pil, dtype=np.int32)
        return SegmentationPrediction(mask=mask)


def make_numpy_mask_model(
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    name: str = "numpy_mask_model",
) -> BenchmarkableModel:
    """Wrap a plain numpy instance-mask callable as a :class:`BenchmarkableModel`.

    The callable must accept a ``(H, W, 3) uint8`` RGB image and return
    a ``(H', W') int`` instance mask where pixel values are instance
    IDs (``0`` is conventionally background). If ``(H', W') != (H, W)``
    the output is nearest-neighbour resized to match the GT mask so
    integer IDs are preserved.

    Parameters
    ----------
    fn : callable
        ``fn(rgb_uint8) -> mask_int``.
    name : str

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_seg(rgb):
    ...     mask = np.zeros(rgb.shape[:2], dtype=np.int32)
    ...     mask[rgb.sum(-1) > 384] = 1  # trivial brightness threshold
    ...     return mask
    >>> bm = rpx.make_numpy_mask_model(my_seg, name="mine")
    >>> bm.task is rpx.TaskType.OBJECT_SEGMENTATION
    True
    """
    return BenchmarkableModel(
        task=TaskType.OBJECT_SEGMENTATION,
        input_adapter=_NumpyMaskInput(),
        model=fn,
        output_adapter=_NumpyMaskOutput(),
        invoker=lambda model, payload: model(payload),
        name=name,
    )


# --------------------------------------------------------------------------- #
# Shared helpers for numpy fast paths
# --------------------------------------------------------------------------- #


def _passthrough_invoker(model: Any, payload: Any) -> Any:
    """Invoker for numpy fast paths — just call the function on its payload."""
    return model(payload)


class _NumpyRgbInput:
    """Every numpy task's input adapter: just hand the RGB ndarray through.

    Extra tokens (second RGB, target pose, ...) travel in
    ``sample.metadata`` and are picked out by the per-task output
    adapter, so a single input adapter serves every task.
    """

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        context = {"target_hw": rgb.shape[:2], "sample": sample}
        return PreparedInput(payload=rgb, context=context)


def _coerce_dict_result(
    result: Any,
    *,
    required_keys: Sequence[str],
    task_name: str,
) -> Dict[str, Any]:
    """Normalise a function's return value into a dict with the expected keys.

    Accepts either a dict or an ``(a, b, ...)`` tuple aligned with
    ``required_keys``. Raises :class:`AdapterError` when neither
    shape matches.
    """
    if isinstance(result, dict):
        missing = [k for k in required_keys if k not in result]
        if missing:
            raise AdapterError(
                f"{task_name} model result is missing required keys: {missing}",
                hint=f"Return a dict with keys {list(required_keys)}.",
            )
        return {k: result[k] for k in required_keys}
    if isinstance(result, tuple) and len(result) == len(required_keys):
        return dict(zip(required_keys, result, strict=False))
    raise AdapterError(
        f"{task_name} model must return a dict with keys "
        f"{list(required_keys)} or a {len(required_keys)}-tuple in that order; "
        f"got {type(result).__name__}",
    )


# --------------------------------------------------------------------------- #
# Object detection — fn(rgb) -> {boxes, scores, labels}
# --------------------------------------------------------------------------- #


class _NumpyDetectionOutput:
    """Wrap a detection callable's output into a :class:`DetectionPrediction`."""

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> DetectionPrediction:
        fields = _coerce_dict_result(
            model_output,
            required_keys=("boxes", "scores", "labels"),
            task_name="detection",
        )
        boxes = np.asarray(fields["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(fields["scores"], dtype=np.float32).reshape(-1)
        labels = list(fields["labels"])
        if not (len(boxes) == len(scores) == len(labels)):
            raise AdapterError(
                f"detection model returned mismatched lengths: "
                f"boxes={len(boxes)} scores={len(scores)} labels={len(labels)}",
            )
        return DetectionPrediction(boxes=boxes, scores=scores, labels=labels)


def make_numpy_detection_model(
    fn: Callable[[np.ndarray], Any],
    *,
    name: str = "numpy_detection_model",
    task: TaskType = TaskType.OBJECT_DETECTION,
) -> BenchmarkableModel:
    """Wrap a plain numpy detection callable as a :class:`BenchmarkableModel`.

    The callable must accept a ``(H, W, 3) uint8`` RGB image and
    return either a dict with keys ``"boxes"`` / ``"scores"`` /
    ``"labels"`` or a ``(boxes, scores, labels)`` tuple in that
    order. Boxes are pixel coordinates in ``(x1, y1, x2, y2)``
    format, scores are floats in ``[0, 1]``, labels are strings.

    Parameters
    ----------
    fn : callable
        ``fn(rgb_uint8) -> dict | tuple``.
    name : str
        Display name for reports.
    task : TaskType
        Use :attr:`TaskType.OBJECT_DETECTION` for closed-vocabulary
        detection or :attr:`TaskType.OPEN_VOCAB_DETECTION` for
        open-vocab (the Prediction contract is the same).

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_det(rgb):
    ...     return {
    ...         "boxes": np.array([[10, 10, 30, 30]], dtype=np.float32),
    ...         "scores": np.array([0.9], dtype=np.float32),
    ...         "labels": ["cup"],
    ...     }
    >>> bm = rpx.make_numpy_detection_model(my_det)
    >>> bm.task is rpx.TaskType.OBJECT_DETECTION
    True
    """
    return BenchmarkableModel(
        task=task,
        input_adapter=_NumpyRgbInput(),
        model=fn,
        output_adapter=_NumpyDetectionOutput(),
        invoker=_passthrough_invoker,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Visual grounding — fn(rgb, text) -> {boxes, scores}
# --------------------------------------------------------------------------- #


class _NumpyGroundingInput:
    """Grounding input: raw RGB + the referring expression from the GT."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        text = getattr(sample.ground_truth, "text", "")
        return PreparedInput(
            payload={"rgb": rgb, "text": text},
            context={"target_hw": rgb.shape[:2]},
        )


class _NumpyGroundingOutput:
    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> VisualGroundingPrediction:
        fields = _coerce_dict_result(
            model_output,
            required_keys=("boxes", "scores"),
            task_name="grounding",
        )
        boxes = np.asarray(fields["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(fields["scores"], dtype=np.float32).reshape(-1)
        labels = fields.get("labels") if isinstance(fields, dict) else None
        return VisualGroundingPrediction(
            boxes=boxes,
            scores=scores,
            labels=labels,
        )


def _grounding_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    """Grounding callable signature: ``fn(rgb, text)``."""
    return model(payload["rgb"], payload["text"])


def make_numpy_grounding_model(
    fn: Callable[[np.ndarray, str], Any],
    *,
    name: str = "numpy_grounding_model",
) -> BenchmarkableModel:
    """Wrap a visual-grounding callable as a :class:`BenchmarkableModel`.

    The callable takes ``(rgb_uint8, text)`` and returns either a
    dict with keys ``"boxes"`` / ``"scores"`` or a tuple
    ``(boxes, scores)``. Boxes are ``(x1, y1, x2, y2)`` pixel
    coordinates; scores are floats. The referring expression
    ``text`` is plucked from ``sample.ground_truth.text`` by the
    adapter so the callable never sees the GT boxes.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_ground(rgb, text):
    ...     return (
    ...         np.array([[10, 10, 30, 30]], dtype=np.float32),
    ...         np.array([0.8], dtype=np.float32),
    ...     )
    >>> bm = rpx.make_numpy_grounding_model(my_ground)
    """
    return BenchmarkableModel(
        task=TaskType.VISUAL_GROUNDING,
        input_adapter=_NumpyGroundingInput(),
        model=fn,
        output_adapter=_NumpyGroundingOutput(),
        invoker=_grounding_invoker,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Relative camera pose — fn(rgb_a, rgb_b) -> {rotation, translation}
# --------------------------------------------------------------------------- #


class _NumpyPoseInput:
    """Pose input: extract rgb_a + rgb_b from the sample metadata side-channel."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb_a = np.asarray(sample.rgb, dtype=np.uint8)
        rgb_b = None
        if sample.metadata is not None:
            rgb_b = sample.metadata.get("rgb_b")
        if rgb_b is None:
            raise AdapterError(
                "Relative-pose numpy adapter requires a second RGB frame "
                "(`rgb_b`) loaded by the dataset.",
                hint="Check that your manifest entries include `rgb_b` "
                "alongside `rgb` so the loader stashes it in sample.metadata.",
            )
        return PreparedInput(
            payload={"rgb_a": rgb_a, "rgb_b": np.asarray(rgb_b, dtype=np.uint8)},
            context={"target_hw": rgb_a.shape[:2]},
        )


class _NumpyPoseOutput:
    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> RelativePosePrediction:
        fields = _coerce_dict_result(
            model_output,
            required_keys=("rotation", "translation"),
            task_name="relative_pose",
        )
        return RelativePosePrediction(
            rotation=np.asarray(fields["rotation"], dtype=np.float64),
            translation=np.asarray(fields["translation"], dtype=np.float64),
        )


def _pose_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    return model(payload["rgb_a"], payload["rgb_b"])


def make_numpy_pose_model(
    fn: Callable[[np.ndarray, np.ndarray], Any],
    *,
    name: str = "numpy_pose_model",
) -> BenchmarkableModel:
    """Wrap a relative-camera-pose callable as a :class:`BenchmarkableModel`.

    The callable takes ``(rgb_a, rgb_b)`` and returns either a dict
    with keys ``"rotation"`` (3×3 rotation matrix or 4-element
    quaternion) and ``"translation"`` (3-vector, metres) or a
    ``(rotation, translation)`` tuple.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_pose(rgb_a, rgb_b):
    ...     return {"rotation": np.eye(3), "translation": np.zeros(3)}
    >>> bm = rpx.make_numpy_pose_model(my_pose)
    """
    return BenchmarkableModel(
        task=TaskType.RELATIVE_CAMERA_POSE,
        input_adapter=_NumpyPoseInput(),
        model=fn,
        output_adapter=_NumpyPoseOutput(),
        invoker=_pose_invoker,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Sparse depth — fn(rgb, coords) -> depths
# --------------------------------------------------------------------------- #


class _NumpySparseDepthInput:
    """Sparse-depth input: rgb + the GT sparse pixel coordinates."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        gt = sample.ground_truth
        coords = getattr(gt, "coordinates", None)
        if coords is None:
            raise AdapterError(
                "Sparse-depth ground truth missing `coordinates`.",
            )
        return PreparedInput(
            payload={"rgb": rgb, "coordinates": np.asarray(coords, dtype=np.float32)},
            context={"target_hw": rgb.shape[:2]},
        )


class _NumpySparseDepthOutput:
    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> SparseDepthPrediction:
        # The callable is responsible for returning depths at the
        # sparse coords it was given; we pass its coords back as-is.
        coords = np.asarray(sample.ground_truth.coordinates, dtype=np.float32)
        depths = np.asarray(model_output, dtype=np.float32).reshape(-1)
        if len(depths) != len(coords):
            raise AdapterError(
                f"sparse-depth model returned {len(depths)} depths for "
                f"{len(coords)} queried coordinates.",
            )
        return SparseDepthPrediction(coordinates=coords, depths=depths)


def _sparse_depth_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    return model(payload["rgb"], payload["coordinates"])


def make_numpy_sparse_depth_model(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    name: str = "numpy_sparse_depth_model",
) -> BenchmarkableModel:
    """Wrap a sparse-depth callable as a :class:`BenchmarkableModel`.

    The callable takes ``(rgb_uint8, coords)`` where ``coords`` is a
    ``(N, 2)`` float32 array of pixel coordinates and returns an
    ``(N,)`` float32 array of depths in metres at those exact
    coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_sd(rgb, coords):
    ...     return np.full(len(coords), 2.0, dtype=np.float32)
    >>> bm = rpx.make_numpy_sparse_depth_model(my_sd)
    """
    return BenchmarkableModel(
        task=TaskType.SPARSE_DEPTH,
        input_adapter=_NumpySparseDepthInput(),
        model=fn,
        output_adapter=_NumpySparseDepthOutput(),
        invoker=_sparse_depth_invoker,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Novel view synthesis — fn(rgb, target_pose) -> rgb
# --------------------------------------------------------------------------- #


class _NumpyNVSInput:
    """NVS input: source RGB + target camera pose (4x4)."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb_src = np.asarray(sample.rgb, dtype=np.uint8)
        target_pose = getattr(sample.ground_truth, "camera_pose", None)
        if target_pose is None:
            raise AdapterError(
                "NVS ground truth missing `camera_pose` (target pose).",
                hint="Use a manifest that sets `target_pose` so the loader "
                "populates ground_truth.camera_pose.",
            )
        return PreparedInput(
            payload={"rgb": rgb_src, "target_pose": np.asarray(target_pose)},
            context={"target_hw": rgb_src.shape[:2]},
        )


class _NumpyNVSOutput:
    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> NovelViewSynthesisPrediction:
        rgb = np.asarray(model_output)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise AdapterError(
                f"NVS model must return a (H, W, 3) RGB array; got shape {rgb.shape}",
            )
        if rgb.dtype != np.uint8:
            rgb = rgb.clip(0, 255).astype(np.uint8)
        target_hw = context.get("target_hw")
        if target_hw is not None and rgb.shape[:2] != target_hw:
            from PIL import Image

            pil = Image.fromarray(rgb)
            pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            rgb = np.asarray(pil, dtype=np.uint8)
        return NovelViewSynthesisPrediction(rgb=rgb)


def _nvs_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    return model(payload["rgb"], payload["target_pose"])


def make_numpy_nvs_model(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    name: str = "numpy_nvs_model",
) -> BenchmarkableModel:
    """Wrap a novel-view-synthesis callable as a :class:`BenchmarkableModel`.

    The callable takes ``(rgb_uint8, target_pose)`` where the target
    pose is a 4×4 SE(3) camera-to-world matrix (float64). It
    returns an RGB image for the target viewpoint. Non-uint8 output
    is clipped and cast; shape mismatches are bilinearly resized.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_nvs(rgb, target_pose):
    ...     return rgb  # identity baseline
    >>> bm = rpx.make_numpy_nvs_model(my_nvs)
    """
    return BenchmarkableModel(
        task=TaskType.NOVEL_VIEW_SYNTHESIS,
        input_adapter=_NumpyNVSInput(),
        model=fn,
        output_adapter=_NumpyNVSOutput(),
        invoker=_nvs_invoker,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Keypoint matching — fn(rgb_a, rgb_b) -> {points0, points1}
# --------------------------------------------------------------------------- #


class _NumpyKeypointInput:
    """Keypoint matching input: rgb_a + rgb_b from sample metadata."""

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb_a = np.asarray(sample.rgb, dtype=np.uint8)
        rgb_b = None
        if sample.metadata is not None:
            rgb_b = sample.metadata.get("rgb_b")
        if rgb_b is None:
            raise AdapterError(
                "Keypoint matching numpy adapter requires a second RGB "
                "frame (`rgb_b`) loaded by the dataset.",
                hint="Check that your manifest entries include `rgb_b` alongside `rgb`.",
            )
        return PreparedInput(
            payload={"rgb_a": rgb_a, "rgb_b": np.asarray(rgb_b, dtype=np.uint8)},
            context={"target_hw": rgb_a.shape[:2]},
        )


class _NumpyKeypointOutput:
    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> KeypointCorrespondencePrediction:
        if isinstance(model_output, dict):
            p0 = np.asarray(model_output.get("points0"), dtype=np.float32)
            p1 = np.asarray(model_output.get("points1"), dtype=np.float32)
            scores = model_output.get("scores")
        elif isinstance(model_output, tuple) and len(model_output) in (2, 3):
            p0 = np.asarray(model_output[0], dtype=np.float32)
            p1 = np.asarray(model_output[1], dtype=np.float32)
            scores = model_output[2] if len(model_output) == 3 else None
        else:
            raise AdapterError(
                "keypoint matching model must return a dict with keys "
                "{points0, points1[, scores]} or a 2/3-tuple.",
            )
        scores_arr = (
            np.asarray(scores, dtype=np.float32).reshape(-1) if scores is not None else None
        )
        return KeypointCorrespondencePrediction(
            points0=p0.reshape(-1, 2),
            points1=p1.reshape(-1, 2),
            scores=scores_arr,
        )


def _keypoint_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    return model(payload["rgb_a"], payload["rgb_b"])


def make_numpy_keypoint_model(
    fn: Callable[[np.ndarray, np.ndarray], Any],
    *,
    name: str = "numpy_keypoint_model",
) -> BenchmarkableModel:
    """Wrap a keypoint-matching callable as a :class:`BenchmarkableModel`.

    The callable takes ``(rgb_a, rgb_b)`` and returns either a dict
    with keys ``"points0"``, ``"points1"`` and optional ``"scores"``
    or a 2/3-tuple in the same order. Points are ``(N, 2)`` pixel
    coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_matcher(rgb_a, rgb_b):
    ...     pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
    ...     return pts, pts
    >>> bm = rpx.make_numpy_keypoint_model(my_matcher)
    """
    return BenchmarkableModel(
        task=TaskType.KEYPOINT_MATCHING,
        input_adapter=_NumpyKeypointInput(),
        model=fn,
        output_adapter=_NumpyKeypointOutput(),
        invoker=_keypoint_invoker,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Object tracking — fn(rgb) -> [{track_id, boxes, scores?}, ...]
# --------------------------------------------------------------------------- #


class _NumpyTrackingInput:
    """Tracking input: a single RGB frame per step.

    Per-frame tracking protocol: the model is handed one frame and
    must return the *currently active* tracks with a persistent
    ``track_id`` per track. The surrounding benchmark pipeline is
    responsible for accumulating tracks into a full :class:`Tracklet`
    set at scene boundaries.
    """

    def prepare(self, sample: Sample) -> PreparedInput:
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        return PreparedInput(
            payload={"rgb": rgb},
            context={"target_hw": rgb.shape[:2], "sample_id": sample.id},
        )


class _NumpyTrackingOutput:
    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> TrackletPrediction:
        tracks = _coerce_tracking_output(model_output)
        return TrackletPrediction(tracks=tracks)


def _coerce_tracking_output(model_output: Any) -> list[Tracklet]:
    """Accept a variety of tracking return shapes; normalise to Tracklets.

    Supported shapes:
        * ``TrackletPrediction(tracks=[...])`` — returned verbatim.
        * ``[Tracklet, ...]`` — returned verbatim.
        * ``[{"track_id": str, "boxes": (T, 4), "scores"?: (T,)}, ...]``
        * ``{"tracks": [as above]}`` — unwrapped.
    """
    if isinstance(model_output, TrackletPrediction):
        return list(model_output.tracks)
    if isinstance(model_output, dict) and "tracks" in model_output:
        model_output = model_output["tracks"]
    if not isinstance(model_output, Sequence):
        raise AdapterError(
            "Tracking model must return a sequence of tracks or a TrackletPrediction.",
            hint="Return either [Tracklet, ...] or "
            "[{'track_id': str, 'boxes': (T,4) array, ...}, ...].",
        )
    tracks: list[Tracklet] = []
    for item in model_output:
        if isinstance(item, Tracklet):
            tracks.append(item)
            continue
        if not isinstance(item, dict):
            raise AdapterError(
                f"Tracking output entries must be Tracklet or dict, got {type(item).__name__}.",
            )
        try:
            boxes = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4)
        except (KeyError, ValueError) as e:
            raise AdapterError(
                f"Tracklet dict is missing / malformed 'boxes': {e}",
                hint="Boxes must be a (T, 4) array of [x1,y1,x2,y2] floats.",
            ) from e
        scores_raw = item.get("scores")
        scores = (
            np.asarray(scores_raw, dtype=np.float32).reshape(-1) if scores_raw is not None else None
        )
        tracks.append(
            Tracklet(
                track_id=str(item.get("track_id", f"t{len(tracks)}")),
                boxes=boxes,
                scores=scores,
            )
        )
    return tracks


def _tracking_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    return model(payload["rgb"])


def make_numpy_tracking_model(
    fn: Callable[[np.ndarray], Any],
    *,
    name: str = "numpy_tracking_model",
) -> BenchmarkableModel:
    """Wrap a per-frame tracking callable as a :class:`BenchmarkableModel`.

    The callable takes a single ``rgb`` ``H×W×3 uint8`` frame and
    returns the active tracks. Accepted return shapes (see
    :func:`_coerce_tracking_output`):

    - ``TrackletPrediction``
    - list of :class:`Tracklet`
    - list of ``{"track_id": str, "boxes": (T, 4), "scores"?: (T,)}``
    - ``{"tracks": [as above]}``

    Examples
    --------
    >>> import numpy as np
    >>> import rpx_benchmark as rpx
    >>> def my_tracker(rgb):
    ...     return [{
    ...         "track_id": "obj_0",
    ...         "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
    ...     }]
    >>> bm = rpx.make_numpy_tracking_model(my_tracker)
    """
    return BenchmarkableModel(
        task=TaskType.OBJECT_TRACKING,
        input_adapter=_NumpyTrackingInput(),
        model=fn,
        output_adapter=_NumpyTrackingOutput(),
        invoker=_tracking_invoker,
        name=name,
    )
