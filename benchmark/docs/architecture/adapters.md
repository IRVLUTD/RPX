# Adapters

The adapter framework separates three concerns:

1. **InputAdapter** — turns an RPX [`Sample`][rpx_benchmark.api.Sample]
   into whatever the model wants as input.
2. **Model** — whatever callable you are benchmarking.
3. **OutputAdapter** — turns the model's raw output into a
   task-specific Prediction dataclass at the resolution the ground
   truth expects.

```text
Sample ───► InputAdapter.prepare ───► PreparedInput(payload, context)
                                              │
                                              ▼
                                       model(payload)
                                              │
                                              ▼
Sample, context, model_output ───► OutputAdapter.finalize ───► Prediction
```

All three are composed into a single
[`BenchmarkableModel`][rpx_benchmark.adapters.base.BenchmarkableModel]
that satisfies the
[`BenchmarkModel`][rpx_benchmark.api.BenchmarkModel] ABC the runner
iterates over.

## Why three pieces?

- **The model becomes interchangeable.** Swapping a model family
  (e.g. HF depth → UniDepth → Metric3D) only changes the adapter
  pair, not the runner, the metric computation, the reports, or the
  CLI.
- **Preprocessing is reusable.** A single
  [`HFDepthInputAdapter`][rpx_benchmark.adapters.depth_hf.HFDepthInputAdapter]
  serves every HuggingFace depth checkpoint — DA-v2, Depth Pro,
  ZoeDepth, PromptDA, Video Depth Anything.
- **Postprocessing is introspectable.**
  [`HFDepthOutputAdapter`][rpx_benchmark.adapters.depth_hf.HFDepthOutputAdapter]
  detects which `post_process_depth_estimation` kwargs the processor
  accepts so different checkpoints' signature quirks are handled
  without custom code.

## PreparedInput

```python
@dataclass
class PreparedInput:
    payload: Any           # what gets passed to the model
    context: Dict[str, Any] = field(default_factory=dict)
```

- **payload**: if it's a dict the default invoker calls
  `model(**payload)`; otherwise `model(payload)`. Override by passing
  `invoker=...` to `BenchmarkableModel`.
- **context**: free-form dict that the output adapter receives back.
  Use it to stash things like target image size, original intrinsics,
  letterbox scale, or any preprocessing metadata that the
  postprocessing step needs to invert.

## Default invoker

```python
def default_invoker(model, payload):
    with torch.no_grad():
        return model(**payload) if isinstance(payload, dict) else model(payload)
```

Override with `BenchmarkableModel(..., invoker=my_invoker)` when your
model needs a different calling convention. UniDepth V2 uses this to
call `.infer(...)` instead of `__call__`.

## Shipped adapters

| File | Families |
|---|---|
| `rpx_benchmark/adapters/base.py` | `make_numpy_depth_model`, `make_numpy_mask_model` — wrap any numpy callable |
| `rpx_benchmark/adapters/depth_hf.py` | Any HuggingFace `AutoModelForDepthEstimation` checkpoint (DA-v2, Depth Pro, ZoeDepth, PromptDA, Video DA) |
| `rpx_benchmark/adapters/depth_unidepth.py` | UniDepth V2 (custom `.infer()` invoker) |
| `rpx_benchmark/adapters/depth_metric3d.py` | Metric3D V2 via `torch.hub`, letterbox canonical-focal |
| `rpx_benchmark/adapters/seg_hf.py` | Any HuggingFace Mask2Former / OneFormer / MaskFormer / SegFormer / DETR-panoptic checkpoint |

## Writing your own

```python
from rpx_benchmark.adapters import (
    BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput,
)
from rpx_benchmark.api import DepthPrediction, Sample, TaskType

class MyInputAdapter(InputAdapter):
    def setup(self) -> None: ...                       # optional
    def prepare(self, sample: Sample) -> PreparedInput:
        return PreparedInput(
            payload={"pixel_values": some_tensor},
            context={"target_hw": sample.rgb.shape[:2]},
        )

class MyOutputAdapter(OutputAdapter):
    def setup(self) -> None: ...                       # optional
    def finalize(self, model_output, context, sample) -> DepthPrediction:
        return DepthPrediction(depth_map=...)

bm = BenchmarkableModel(
    task=TaskType.MONOCULAR_DEPTH,
    input_adapter=MyInputAdapter(),
    model=my_model_object,
    output_adapter=MyOutputAdapter(),
    name="my_custom_model",
)
```
