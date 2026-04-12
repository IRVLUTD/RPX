# Adding a New Model Adapter

A model adapter is a **factory** that returns a
[`BenchmarkableModel`][rpx_benchmark.adapters.base.BenchmarkableModel]
when called. Registering it under a short name makes it available
via `rpx bench <task> --model <your_name>`.

## Case 1 — Another HuggingFace depth checkpoint

The shortest possible factory:

```python
# rpx_benchmark/models/my_depth.py
from ..reference.adapters.depth_hf import make_hf_depth_model


def my_depth_large(*, device="cuda", **kwargs):
    return make_hf_depth_model(
        "my-org/my-depth-large",
        device=device,
        name="my_depth_large",
        **kwargs,
    )
```

Register it:

```python
# rpx_benchmark/models/registry.py — one line in MODEL_REGISTRY
MODEL_REGISTRY["my_depth_large"] = ("my_depth", "my_depth_large")
```

That's it. The CLI now accepts `rpx bench monocular_depth --model
my_depth_large`, and `rpx models` lists the new entry.

## Case 2 — Another HuggingFace segmentation checkpoint

```python
from ..reference.adapters.seg_hf import make_hf_instance_seg_model


def my_segmentor(*, device="cuda", **kwargs):
    return make_hf_instance_seg_model(
        "my-org/my-mask2former",
        device=device,
        name="my_segmentor",
        **kwargs,
    )
```

Works with any checkpoint whose processor exposes one of
`post_process_instance_segmentation`,
`post_process_panoptic_segmentation`, or
`post_process_semantic_segmentation`. The output adapter detects
which one at setup time.

## Case 3 — A non-HuggingFace model family

Write your own input/output adapter pair plus a factory that glues
them to the actual model class. Two shipped examples:

- **[`adapters/depth_unidepth.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/adapters/depth_unidepth.py)**
  — UniDepth V2. Uses a custom invoker because UniDepth exposes
  `.infer(rgb, camera=...)` instead of `__call__`.
- **[`adapters/depth_metric3d.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/adapters/depth_metric3d.py)**
  — Metric3D V2. Uses `torch.hub.load` plus letterbox preprocessing
  and canonical-focal rescale on the output side.

Skeleton:

```python
from typing import Any, Dict, Optional
from rpx_benchmark.adapters import (
    BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput,
)
from rpx_benchmark.api import DepthPrediction, Sample, TaskType


class MyInputAdapter(InputAdapter):
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
    def setup(self) -> None: ...
    def prepare(self, sample: Sample) -> PreparedInput:
        # your preprocessing here
        return PreparedInput(payload=..., context={"target_hw": sample.rgb.shape[:2]})


class MyOutputAdapter(OutputAdapter):
    def setup(self) -> None: ...
    def finalize(self, model_output: Any, context: Dict[str, Any],
                 sample: Sample) -> DepthPrediction:
        # your postprocessing here
        return DepthPrediction(depth_map=...)


def my_invoker(model, payload):
    """Override if your model needs a non-standard call signature."""
    import torch
    with torch.no_grad():
        return model.forward_custom(payload["x"])


def make_my_model(*, device="cuda", name: Optional[str] = None):
    # Load whatever actually holds the weights
    model = _load_my_weights().to(device).eval()
    return BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=MyInputAdapter(device=device),
        model=model,
        output_adapter=MyOutputAdapter(),
        invoker=my_invoker,            # omit for the default
        name=name or "my_model",
    )
```

## Deferred models

If you want the model listed in `rpx models` but not yet runnable
(e.g. you're waiting on an upstream fix), stub a factory that raises
`NotImplementedError` and add the name to `DEFERRED_MODELS`:

```python
# rpx_benchmark/models/_deferred.py
def my_future_model(*, device="cuda", **_):
    return _deferred(
        "my_future_model",
        "Waiting on upstream support for X.",
    )
```

```python
# rpx_benchmark/models/registry.py
MODEL_REGISTRY["my_future_model"] = ("_deferred", "my_future_model")
DEFERRED_MODELS = frozenset({
    ...,
    "my_future_model",
})
```

`rpx models` will print the deferred entries in a separate section
with a short reason.

## Testing your adapter

Two layers of tests pay off:

1. **Fake processor / model test** — exercise the
   `Input → model → Output` contract without touching real weights.
   See
   [`tests/test_adapters.py::test_hf_output_adapter_forwards_only_accepted_kwargs`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/tests/test_adapters.py)
   for the pattern.
2. **Real checkpoint smoke test** (gated, for CI hosts with torch /
   GPU available) — run one batch of synthetic RGB through the real
   adapter. Validates signature assumptions, output shapes, metric
   range.

Both patterns are demonstrated by the shipped adapters' tests.

## Full API

See [`rpx_benchmark.adapters`](../api/adapters.md) for the protocol
definitions and [`rpx_benchmark.models`](../api/models.md) for the
registry API.
