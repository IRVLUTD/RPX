# Quickstart

From a blank environment to a written benchmark report, in Python.

## 1. Install

```bash
pip install 'rpx-benchmark[hub]'
```

`[hub]` pulls `huggingface_hub` for dataset downloads. Add
`[hf-datasets]` if you want the `datasets.load_dataset` path, and
`[schemas]` for strict manifest validation.

## 2. Bring your model and run

```python
import numpy as np
import rpx_benchmark as rpx

# 1. Define your model as a plain numpy callable.
def my_depth(rgb: np.ndarray) -> np.ndarray:
    """rgb: H x W x 3 uint8 -> H x W float32 (metres)."""
    return np.full(rgb.shape[:2], 2.0, dtype=np.float32)

# 2. Wrap it as a BenchmarkableModel.
model = rpx.make_numpy_depth_model(my_depth, name="my_depth")

# 3. Run.
cfg = rpx.MonocularDepthRunConfig(model=model, split="hard", device="cpu")
result, report, paths = rpx.run_monocular_depth(cfg)

print(result.aggregated)                   # absrel, rmse, delta1..3
print(report.weighted_phase_score)         # ESD-weighted phase score
print(paths["json"], paths["markdown"])    # written report files
```

That's it — no CLI, no registry. The same pattern works for every
task: `make_numpy_<task>_model(fn)` + the matching
`<TaskName>RunConfig` + `run_<task>(cfg)`.

## 3. Segmentation example

```python
import numpy as np
import rpx_benchmark as rpx

def my_seg(rgb: np.ndarray) -> np.ndarray:
    """rgb: H x W x 3 uint8 -> H x W int32 (instance IDs)."""
    return np.zeros(rgb.shape[:2], dtype=np.int32)

model = rpx.make_numpy_mask_model(my_seg, name="my_seg")
cfg = rpx.SegmentationRunConfig(model=model, split="hard", device="cpu")
result, report, _ = rpx.run_segmentation(cfg)
```

## 4. Full control: custom input/output adapters

When the numpy fast path doesn't fit (PyTorch, transformers,
JAX, cloud API, ...), implement `InputAdapter.prepare` and
`OutputAdapter.finalize` yourself — see
[Bring Your Own Model](bring-your-own-model.md) for four complete
templates including an API/cloud-model example.

## 5. Read the output

Every run writes two files under `./rpx_results/<model>/<split>/`:

- **`result.json`** — machine-readable. Contains
  `aggregated` metrics, `per_sample` rows with
  `id` / `phase` / `difficulty`, and the full
  `deployment_readiness` report (Weighted Phase Score,
  State-Transition Robustness, Temporal Stability, FLOPs, latency
  percentiles, peak CPU/CUDA/MPS memory, parameter count).
- **`summary.md`** — human-readable tables rendered from the same
  numbers.

## 6. Errors

Every exception subclasses `rpx.RPXError` and carries a `hint`
string. Catch the base class to log a single user-facing error:

```python
try:
    result, report, _ = rpx.run_monocular_depth(cfg)
except rpx.RPXError as e:
    print(f"benchmark failed: {e}")
```
