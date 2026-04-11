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
    Sample,
    SegmentationPrediction,
    TaskType,
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

    Example — wrap a HuggingFace depth model::

        from rpx_benchmark.adapters.depth_hf import make_hf_depth_model
        bm = make_hf_depth_model(
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            device="cuda",
        )
        # Hand `bm` to the benchmark runner or the task entrypoint.

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


def make_numpy_depth_model(
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    name: str = "numpy_depth_model",
) -> BenchmarkableModel:
    """Wrap a plain numpy depth callable as a :class:`BenchmarkableModel`.

    The callable must accept a ``(H, W, 3) uint8`` RGB image and return a
    ``(H', W') float`` metric depth map (in metres). If ``(H', W') !=
    (H, W)``, the output is bilinearly resized to match the ground truth.

    Parameters
    ----------
    fn : callable
        The depth function. Signature: ``fn(rgb_uint8) -> depth_float``.
    name : str
        Display name used in logs and reports.

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
    return BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=_NumpyDepthInput(),
        model=fn,
        output_adapter=_NumpyDepthOutput(),
        invoker=lambda model, payload: model(payload),
        name=name,
    )


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
