# Bring Your Own Model

The toolkit is built so **the only variable is the model**. Datasets,
splits, metrics, reports, and deployment-readiness scoring are
fixed. Users touch only the model — input and output adapters are
already shipped for the two common families.

## Adapter framework at a glance

```text
Sample ───► InputAdapter.prepare ───► PreparedInput(payload, context)
                                              │
                                              ▼
                                       model(payload)
                                              │
                                              ▼
Sample, context, model_output ───► OutputAdapter.finalize ───► Prediction
```

A [`BenchmarkableModel`][rpx_benchmark.adapters.base.BenchmarkableModel]
composes three things: an input adapter, a model callable (anything
that responds to `(payload)` or `(**payload)`), and an output
adapter. All three together satisfy the
[`BenchmarkModel`][rpx_benchmark.api.BenchmarkModel] ABC that the
runner consumes.

## Three fast paths

=== "Zero code — any HF checkpoint"

    ```bash
    rpx bench monocular_depth \
        --hf-checkpoint my-org/my-depth-model \
        --split hard
    ```

    Works with any checkpoint loadable via
    `transformers.AutoModelForDepthEstimation`. The shipped
    [`HFDepthInputAdapter`][rpx_benchmark.adapters.depth_hf.HFDepthInputAdapter] /
    [`HFDepthOutputAdapter`][rpx_benchmark.adapters.depth_hf.HFDepthOutputAdapter]
    pair handles preprocessing, postprocessing, and resize-back-to-GT.

    The output adapter **introspects the processor signature** at
    setup time so it correctly handles checkpoints with different
    post-process kwargs (DA-v2 takes `target_sizes` only; ZoeDepth
    also requires `source_sizes`; PromptDA takes an optional
    `prompt_depth`).

=== "Plain numpy callable"

    ```python
    import numpy as np
    import rpx_benchmark as rpx

    def my_depth(rgb: np.ndarray) -> np.ndarray:
        """rgb: H x W x 3 uint8 → returns H x W float32 in metres."""
        ...
        return depth_map

    bm = rpx.make_numpy_depth_model(my_depth, name="my_numpy_depth")

    cfg = rpx.MonocularDepthRunConfig(model=bm, split="hard", device="cpu")
    result, report, paths = rpx.run_monocular_depth(cfg)

    print(result.aggregated)                     # absrel, rmse, delta1..3
    print(report.weighted_phase_score.to_dict()) # phase-stratified scores
    ```

    For segmentation the symmetric helper is
    [`rpx.make_numpy_mask_model(fn)`][rpx_benchmark.adapters.base.make_numpy_mask_model]
    where `fn(rgb) → int32 mask`.

=== "Custom torch / transformers stack"

    ```python
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import rpx_benchmark as rpx
    from rpx_benchmark.adapters.depth_hf import (
        HFDepthInputAdapter, HFDepthOutputAdapter,
    )

    processor = AutoImageProcessor.from_pretrained("my-org/my-model")
    model = (
        AutoModelForDepthEstimation
        .from_pretrained("my-org/my-model")
        .to("cuda").eval()
    )

    bm = rpx.BenchmarkableModel(
        task=rpx.TaskType.MONOCULAR_DEPTH,
        input_adapter=HFDepthInputAdapter(processor=processor, device="cuda"),
        model=model,
        output_adapter=HFDepthOutputAdapter(processor=processor),
        name="my_model",
    )

    rpx.run_monocular_depth(
        rpx.MonocularDepthRunConfig(model=bm, split="hard")
    )
    ```

    For non-HuggingFace model families, clone
    `rpx_benchmark/adapters/depth_unidepth.py` (UniDepth pattern) or
    `rpx_benchmark/adapters/depth_metric3d.py` (torch.hub + letterbox
    pattern).

## Writing your own input / output adapter

The protocols are minimal:

```python
from typing import Any, Dict
from rpx_benchmark.adapters import (
    BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput,
)
from rpx_benchmark.api import DepthPrediction, Sample, TaskType

class MyInputAdapter(InputAdapter):
    def setup(self) -> None: ...           # optional one-time init
    def prepare(self, sample: Sample) -> PreparedInput:
        # Return whatever your model wants plus any context you need
        # later for post-processing.
        return PreparedInput(
            payload={"pixel_values": some_tensor},
            context={"target_hw": sample.rgb.shape[:2]},
        )

class MyOutputAdapter(OutputAdapter):
    def setup(self) -> None: ...
    def finalize(self, model_output: Any, context: Dict[str, Any],
                 sample: Sample) -> DepthPrediction:
        # Run your post-processing and return a task Prediction dataclass.
        return DepthPrediction(depth_map=...)

bm = BenchmarkableModel(
    task=TaskType.MONOCULAR_DEPTH,
    input_adapter=MyInputAdapter(),
    model=my_model_object,                 # any callable or nn.Module
    output_adapter=MyOutputAdapter(),
    name="my_custom_model",
)
```

The default invoker calls `model(**payload)` when the payload is a
dict, otherwise `model(payload)`. Override with `invoker=` if your
model needs a different calling convention (see how
`make_unidepth_v2_model` uses a custom invoker to call `.infer(...)`).
