# Bring Your Own Model

The only variable in RPX is **your model**. The toolkit supplies:

- the dataset (via `datasets.load_dataset` or
  `huggingface_hub.snapshot_download`),
- per-task dataloaders yielding `Sample` batches,
- per-task metric calculators,
- hardware-agnostic latency and memory profiling,
- ESD-weighted phase scoring and deployment-readiness reports.

You supply a `BenchmarkModel` (usually composed via an `InputAdapter` +
model callable + `OutputAdapter`). This page shows the four paths
we support, from zero-code to fully custom.

## Adapter framework at a glance

```text
  dataset row ──► RPXDataset (or load_hf)
                           │
                           ▼
                   rpx_benchmark.Sample   (id, rgb, ground_truth, phase, …)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       │
 InputAdapter.prepare(sample)                  │
       │                                       │
       ▼                                       │
 PreparedInput(payload, context)               │
       │                                       │
       ▼                                       │
 model(**payload)  or  model(payload)          │
       │                                       │
       ▼                                       │
 OutputAdapter.finalize(output, context, sample)
       │                                       │
       ▼                                       ▼
 Prediction ─────► MetricCalculator (per-task registry)
```

`BenchmarkableModel` composes the three pieces into the
`BenchmarkModel` ABC that the runner consumes.

## Load data once, run any path below

```python
from rpx_benchmark.data import load_hf

ds = load_hf("monocular_depth", split="hard")
```

Every code sample on this page consumes this `ds` — see
[Load the Dataset](load-the-dataset.md) for the full data-loading
surface (bulk snapshot, streaming, pinned revisions).

## Three paths

=== "1. Plain numpy callable"

    One `make_numpy_*_model` per task — point it at your function and
    you're benchmarkable. No protocol implementation required:

    ```python
    import numpy as np
    import rpx_benchmark as rpx

    def my_depth(rgb: np.ndarray) -> np.ndarray:
        """rgb: H×W×3 uint8 → H×W float32 (metres)."""
        ...

    bm  = rpx.make_numpy_depth_model(my_depth, name="my_depth")
    cfg = rpx.MonocularDepthRunConfig(model=bm, split="hard", device="cpu")
    result, report, paths = rpx.run_monocular_depth(cfg)
    ```

    Ten factories cover every task:
    `make_numpy_depth_model`, `make_numpy_mask_model`,
    `make_numpy_detection_model`, `make_numpy_grounding_model`,
    `make_numpy_pose_model`, `make_numpy_sparse_depth_model`,
    `make_numpy_nvs_model`, `make_numpy_keypoint_model`, and
    `make_numpy_tracking_model`.

=== "2. Custom adapter stack"

    When you need fine-grained control over preprocessing, batching,
    or output post-processing, implement the two protocols
    directly.

    The protocols are minimal:

    ```python
    from rpx_benchmark.adapters import (
        BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput,
    )
    from rpx_benchmark.api import DepthPrediction, Sample, TaskType


    class MyInputAdapter(InputAdapter):
        def setup(self) -> None: ...             # optional one-time init

        def prepare(self, sample: Sample) -> PreparedInput:
            # Return whatever your model wants plus any context you
            # need later for post-processing.
            return PreparedInput(
                payload={"pixel_values": some_tensor},
                context={"target_hw": sample.rgb.shape[:2]},
            )


    class MyOutputAdapter(OutputAdapter):
        def setup(self) -> None: ...

        def finalize(self, model_output, context, sample) -> DepthPrediction:
            # Task-specific post-processing; return one of the task's
            # Prediction dataclasses (DepthPrediction in this case).
            return DepthPrediction(depth_map=...)


    bm = BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=MyInputAdapter(),
        model=my_model_object,                   # any callable or nn.Module
        output_adapter=MyOutputAdapter(),
        name="my_custom_model",
    )
    ```

    The default invoker calls `model(**payload)` when the payload is
    a dict, otherwise `model(payload)`. Pass `invoker=` to override
    (e.g. for a model that exposes `.infer(...)` instead of `.forward(...)`).

=== "3. API / cloud model"

    Wrap a remote inference service in the same adapter pattern —
    `InputAdapter.prepare` serialises the sample, `OutputAdapter.finalize`
    parses the response. Build-in profiler reports the round-trip
    latency; `EfficiencyMetadata(model_type="api")` suppresses
    params / FLOPs / peak-memory (displayed as `N/A (API)`).

    ```python
    import base64
    import io

    import numpy as np
    from PIL import Image

    import rpx_benchmark as rpx
    from rpx_benchmark.adapters import (
        BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput,
    )
    from rpx_benchmark.api import DepthPrediction, Sample, TaskType


    class CloudDepthInput(InputAdapter):
        def setup(self) -> None: ...

        def prepare(self, sample: Sample) -> PreparedInput:
            buf = io.BytesIO()
            Image.fromarray(sample.rgb).save(buf, format="PNG")
            return PreparedInput(
                payload={"image_b64": base64.b64encode(buf.getvalue()).decode()},
                context={"target_hw": sample.rgb.shape[:2]},
            )


    class CloudDepthOutput(OutputAdapter):
        def setup(self) -> None: ...

        def finalize(self, response, context, sample) -> DepthPrediction:
            # response is whatever your client returns.
            depth = np.asarray(response["depth_meters"], dtype=np.float32)
            return DepthPrediction(depth_map=depth)


    def my_cloud_model(image_b64: str) -> dict:
        # Your HTTPS call — replace with requests / httpx / SDK.
        return {"depth_meters": [[...]]}


    bm = BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=CloudDepthInput(),
        model=my_cloud_model,
        output_adapter=CloudDepthOutput(),
        name="my_cloud_depth",
    )

    # Mark this as an API model so efficiency columns read "N/A (API)".
    from rpx_benchmark.profiler import EfficiencyMetadata
    cfg = rpx.MonocularDepthRunConfig(
        model=bm, split="hard", device="cpu",
        efficiency=EfficiencyMetadata(model_type="api"),
    )
    result, report, paths = rpx.run_monocular_depth(cfg)
    ```

    Cost / rate-limit handling is outside the toolkit's concern —
    put retries and throttling in the model callable itself so the
    adapter stays stateless and testable.

## After your run

Every path produces the same three outputs:

1. `BenchmarkResult` — per-sample + aggregated metrics from the
   task's registered calculators.
2. `DeploymentReadinessReport` — ESD-weighted phase score, Temporal
   Stability, Stack Geometric Coherence (where the task registers
   hooks), and `EfficiencyMetadata` (params, FLOPs, latency
   percentiles, peak CPU/CUDA/MPS memory).
3. A `{json, markdown}` dict pointing at written report files.

Hand the `BenchmarkResult` / `DeploymentReadinessReport` to your own
aggregation code, or keep the written files as the canonical record.
