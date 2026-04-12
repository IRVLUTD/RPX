"""HuggingFace Transformers adapters for monocular metric depth.

Works with any ``AutoModelForDepthEstimation`` checkpoint that exposes
``post_process_depth_estimation`` on its image processor — including
Depth Anything V2 (metric), Depth Pro, ZoeDepth, Video Depth Anything
(per-frame), and PromptDA.

Three entry points:

    * :class:`HFDepthInputAdapter` — PIL -> processor -> tensors on device
    * :class:`HFDepthOutputAdapter` — model output -> resized metric depth
    * :func:`make_hf_depth_model` — single-call factory returning a
      ready-to-run :class:`BenchmarkableModel`

``make_hf_depth_model("my-org/my-depth-ckpt")`` is the "bring your own
HuggingFace depth model" fast path — no subclassing required.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from ...api import DepthPrediction, Sample, TaskType
from ...adapters.base import BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput


@dataclass
class HFDepthInputAdapter(InputAdapter):
    """Converts an RPX Sample into a HuggingFace model input batch of 1."""

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
            payload=dict(inputs),  # forwarded as **kwargs to model
            context={"target_hw": (rgb.shape[0], rgb.shape[1])},
        )


@dataclass
class HFDepthOutputAdapter(OutputAdapter):
    """Runs ``post_process_depth_estimation`` and wraps in ``DepthPrediction``.

    Different transformers depth processors take different kwargs:

    - DA-V2 / Depth Pro: ``target_sizes`` only.
    - ZoeDepth: needs ``source_sizes`` (and optionally ``do_remove_padding``)
      to unpad its internal left-right mirror augmentation.
    - PromptDA: takes both plus an optional ``outputs`` key.

    We introspect the post-process signature once and forward whatever
    kwargs it accepts. Both ``target_sizes`` and ``source_sizes`` are set
    to the caller's original ``(H, W)`` since the adapter contract
    promises we always return depth at the input RGB's resolution.
    """

    processor: Any
    _post_kwargs_names: frozenset = field(default_factory=frozenset, init=False, repr=False)

    def setup(self) -> None:
        try:
            sig = inspect.signature(self.processor.post_process_depth_estimation)
            self._post_kwargs_names = frozenset(sig.parameters.keys())
        except (TypeError, ValueError):
            self._post_kwargs_names = frozenset({"target_sizes"})

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> DepthPrediction:
        if not self._post_kwargs_names:
            # Defensive: setup() wasn't called; introspect lazily.
            self.setup()

        h, w = context["target_hw"]
        kwargs: Dict[str, Any] = {}
        if "target_sizes" in self._post_kwargs_names:
            kwargs["target_sizes"] = [(h, w)]
        if "source_sizes" in self._post_kwargs_names:
            kwargs["source_sizes"] = [(h, w)]

        post = self.processor.post_process_depth_estimation(model_output, **kwargs)
        depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        return DepthPrediction(depth_map=depth)


def make_hf_depth_model(
    checkpoint: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    name: str | None = None,
) -> BenchmarkableModel:
    """One-line factory for any HuggingFace depth-estimation checkpoint.

    Parameters
    ----------
    checkpoint : str
        HuggingFace Hub path, e.g.
        ``"depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"``.
    device : str
        Device string passed to ``.to(device)``.
    dtype : str, optional
        One of ``"float16"`` / ``"bfloat16"`` / ``"float32"``. If provided,
        the model is cast to the matching ``torch`` dtype.
    name : str, optional
        Display name; defaults to the checkpoint id.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:  # pragma: no cover - guarded at install time
        raise ImportError(
            "make_hf_depth_model needs torch + transformers. "
            "Install with: pip install 'rpx-benchmark[depth-hf]'"
        ) from e

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint)

    if dtype is not None:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        model = model.to(dtype=torch_dtype)

    model = model.to(device).eval()

    return BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=HFDepthInputAdapter(processor=processor, device=device),
        model=model,
        output_adapter=HFDepthOutputAdapter(processor=processor),
        name=name or checkpoint,
    )
