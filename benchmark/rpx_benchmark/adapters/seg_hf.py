"""HuggingFace Transformers adapters for instance / panoptic segmentation.

Works with any checkpoint whose image processor exposes one of
``post_process_instance_segmentation`` or
``post_process_panoptic_segmentation`` — Mask2Former, OneFormer,
MaskFormer, DETR-for-panoptic.

The output contract is a single ``(H, W) int32`` mask whose pixel
values are consistent instance IDs. For Mask2Former-family models
that means we flatten the per-segment outputs returned by the
processor and paint each segment's pixels with a unique integer.

The processor-signature detection follows the same pattern as
:mod:`rpx_benchmark.adapters.depth_hf`: we introspect at setup time
and pick the right postprocess method.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..api import Sample, SegmentationPrediction, TaskType
from ..exceptions import AdapterError
from .base import BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput


# --------------------------------------------------------------------------- #
# Input adapter
# --------------------------------------------------------------------------- #

@dataclass
class HFInstanceSegInputAdapter(InputAdapter):
    """Turn an RPX :class:`Sample` into a HuggingFace batch of 1 for segmentation.

    Parameters
    ----------
    processor : Any
        The ``AutoImageProcessor`` loaded via
        ``AutoImageProcessor.from_pretrained(checkpoint)``.
    device : str
        PyTorch device string. Pixel values are moved to this device
        on ``prepare``.
    """

    processor: Any
    device: str = "cuda"

    def setup(self) -> None:
        pass

    def prepare(self, sample: Sample) -> PreparedInput:
        from PIL import Image
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = inputs.to(self.device)
        return PreparedInput(
            payload=dict(inputs),
            context={"target_hw": (rgb.shape[0], rgb.shape[1])},
        )


# --------------------------------------------------------------------------- #
# Output adapter
# --------------------------------------------------------------------------- #

@dataclass
class HFInstanceSegOutputAdapter(OutputAdapter):
    """Map HF segmentation outputs to an integer instance mask.

    The processor's ``post_process_instance_segmentation`` or
    ``post_process_panoptic_segmentation`` method is used; which one
    is picked depends on which one the processor exposes (detected at
    :meth:`setup`).

    The result is a single ``(H, W) int32`` mask where each instance
    gets a unique integer (``0`` is background). Instance IDs are
    assigned in the order the processor returns them.

    Parameters
    ----------
    processor : Any
        The image processor that produced the model inputs. Its
        ``post_process_*_segmentation`` method must be called on the
        model outputs.
    threshold : float
        Score threshold below which segments are dropped. Defaults
        to 0.5, matching Mask2Former's default eval config.
    """

    processor: Any
    threshold: float = 0.5
    _post_method: Optional[Callable] = field(default=None, init=False, repr=False)
    _post_kind: str = field(default="", init=False, repr=False)

    def setup(self) -> None:
        candidates = (
            ("instance", "post_process_instance_segmentation"),
            ("panoptic", "post_process_panoptic_segmentation"),
            ("semantic", "post_process_semantic_segmentation"),
        )
        for kind, method_name in candidates:
            method = getattr(self.processor, method_name, None)
            if callable(method):
                self._post_method = method
                self._post_kind = kind
                return
        raise AdapterError(
            f"Processor {type(self.processor).__name__} exposes no "
            "post_process_*_segmentation method.",
            hint="Use a Mask2Former/OneFormer/MaskFormer/SegFormer processor.",
        )

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> SegmentationPrediction:
        if self._post_method is None:
            self.setup()
        h, w = context["target_hw"]

        kwargs: Dict[str, Any] = {}
        sig_params = set(inspect.signature(self._post_method).parameters.keys())
        if "target_sizes" in sig_params:
            kwargs["target_sizes"] = [(h, w)]
        if "threshold" in sig_params:
            kwargs["threshold"] = self.threshold
        if "return_coco_annotation" in sig_params:
            kwargs["return_coco_annotation"] = False

        try:
            result = self._post_method(model_output, **kwargs)
        except Exception as e:
            raise AdapterError(
                f"{self._post_method.__name__} failed: {e}",
            ) from e

        mask = _hf_seg_result_to_instance_mask(result, kind=self._post_kind, h=h, w=w)
        return SegmentationPrediction(mask=mask)


def _hf_seg_result_to_instance_mask(
    result: Any,
    *,
    kind: str,
    h: int,
    w: int,
) -> np.ndarray:
    """Collapse a HF post-process output into a ``(H, W) int32`` instance mask.

    HF processors return different shapes for different task kinds:

    - **instance**: ``[{"segmentation": (H, W) Tensor, "segments_info":
      [{"id": i, "label_id": c, "score": s}, ...]}]``
    - **panoptic**: same shape as instance.
    - **semantic**: ``[(H, W) Tensor]`` — per-pixel class IDs. We
      treat each class ID as its own "instance" for RPX reporting.
    """
    import torch

    if kind in ("instance", "panoptic"):
        item = result[0]
        seg_t = item.get("segmentation")
        if seg_t is None:
            return np.zeros((h, w), dtype=np.int32)
        seg = seg_t.detach().cpu().numpy().astype(np.int32)
        if seg.shape != (h, w):
            from PIL import Image
            seg = np.asarray(
                Image.fromarray(seg, mode="I").resize((w, h), Image.NEAREST),
                dtype=np.int32,
            )
        # HF's output uses 0 for no-object and per-segment ids otherwise.
        # That already matches our contract.
        return seg

    # semantic
    item = result[0]
    if isinstance(item, torch.Tensor):
        seg = item.detach().cpu().numpy().astype(np.int32)
    else:
        seg = np.asarray(item, dtype=np.int32)
    if seg.shape != (h, w):
        from PIL import Image
        seg = np.asarray(
            Image.fromarray(seg, mode="I").resize((w, h), Image.NEAREST),
            dtype=np.int32,
        )
    return seg


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def make_hf_instance_seg_model(
    checkpoint: str,
    *,
    device: str = "cuda",
    threshold: float = 0.5,
    name: Optional[str] = None,
    model_class_hint: Optional[str] = None,
) -> BenchmarkableModel:
    """One-line factory for a HuggingFace segmentation checkpoint.

    Parameters
    ----------
    checkpoint : str
        HuggingFace Hub id
        (e.g. ``"facebook/mask2former-swin-tiny-coco-instance"``).
    device : str
    threshold : float
        Score threshold for instance acceptance (passed to the
        processor's post-process if it accepts the kwarg).
    name : str, optional
        Display name. Defaults to ``checkpoint``.
    model_class_hint : str, optional
        One of ``"instance"``, ``"universal"``, ``"semantic"``. Most
        users should leave this as ``None`` and rely on
        ``AutoModelForUniversalSegmentation`` (the super-class used
        by Mask2Former / OneFormer). Only set this if the auto class
        does not dispatch correctly for your checkpoint.

    Raises
    ------
    AdapterError
        If the processor exposes no post-process method we can use.
    ImportError
        If ``torch`` or ``transformers`` are not installed.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "make_hf_instance_seg_model needs torch + transformers. "
            "Install with: pip install 'rpx-benchmark[depth-hf]'"
        ) from e

    processor = AutoImageProcessor.from_pretrained(checkpoint)

    model_cls_name = {
        None: "AutoModelForUniversalSegmentation",
        "universal": "AutoModelForUniversalSegmentation",
        "instance": "AutoModelForInstanceSegmentation",
        "semantic": "AutoModelForSemanticSegmentation",
    }[model_class_hint]
    import transformers
    try:
        model_cls = getattr(transformers, model_cls_name)
    except AttributeError as e:
        raise AdapterError(
            f"transformers has no class named {model_cls_name!r}.",
        ) from e
    model = model_cls.from_pretrained(checkpoint).to(device).eval()

    return BenchmarkableModel(
        task=TaskType.OBJECT_SEGMENTATION,
        input_adapter=HFInstanceSegInputAdapter(processor=processor, device=device),
        model=model,
        output_adapter=HFInstanceSegOutputAdapter(
            processor=processor, threshold=threshold,
        ),
        name=name or checkpoint,
    )
