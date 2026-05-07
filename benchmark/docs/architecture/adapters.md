# Adapter framework

The adapter framework is how a user's model plugs into the runner.
It consists of two protocols and one container class.

```text
Sample ──► InputAdapter.prepare ──► PreparedInput(payload, context)
                                             │
                                             ▼
                                      model(**payload)  (or model(payload))
                                             │
                                             ▼
Sample, context, model_output ──► OutputAdapter.finalize ──► Prediction
```

## Protocols

```python
from rpx_benchmark import Sample
from rpx_benchmark.adapters import InputAdapter, OutputAdapter, PreparedInput

class InputAdapter:
    def setup(self) -> None: ...                                 # optional
    def prepare(self, sample: Sample) -> PreparedInput: ...


class OutputAdapter:
    def setup(self) -> None: ...                                 # optional
    def finalize(self, model_output, context, sample): ...       # -> task Prediction
```

`PreparedInput` has two fields:

- `payload` — whatever the model's forward call takes. Dicts go in as
  `model(**payload)`; anything else as `model(payload)`.
- `context` — a free-form dict the output adapter receives back.
  Stash things like `target_hw`, intrinsics, or other
  preprocessing metadata the postprocess step needs.

## Container

`BenchmarkableModel` composes an input adapter, a model callable, and
an output adapter into a class that satisfies the
[`BenchmarkModel`][rpx_benchmark.api.BenchmarkModel] ABC the runner
consumes:

```python
bm = BenchmarkableModel(
    task=TaskType.MONOCULAR_DEPTH,
    input_adapter=MyInputAdapter(),
    model=my_model,
    output_adapter=MyOutputAdapter(),
    name="my_model",
)
```

## Numpy fast paths

For every task, there is a one-line helper that wraps a plain
`numpy`-in / `numpy`-out callable as a `BenchmarkableModel`:

| Task | Factory |
|---|---|
| Monocular depth | `make_numpy_depth_model` |
| Object segmentation | `make_numpy_mask_model` |
| Object detection | `make_numpy_detection_model` |
| Visual grounding | `make_numpy_grounding_model` |
| Relative camera pose | `make_numpy_pose_model` |
| Sparse depth | `make_numpy_sparse_depth_model` |
| Novel view synthesis | `make_numpy_nvs_model` |
| Keypoint matching | `make_numpy_keypoint_model` |
| Object tracking | `make_numpy_tracking_model` |

Each factory documents the shape `fn` receives and returns — see
the docstrings in `rpx_benchmark/adapters/base.py`.
