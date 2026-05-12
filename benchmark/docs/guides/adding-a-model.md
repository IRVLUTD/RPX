# Adding a Model to RPX

This guide covers how to add a new perception model to any RPX task.
The process is the same regardless of task — one adapter file, one
registry line, and you're benchmarking.

## Quick version (30 seconds)

```bash
# 1. Create your adapter
cp scripts/pose_models/opencv_baseline.py scripts/pose_models/my_model.py
# Edit my_model.py — implement __call__

# 2. Register it (one line in __init__.py)
# Add to MODEL_REGISTRY in scripts/pose_models/__init__.py:
#   "my_model": _build_my_model,

# 3. Run
PYTHONPATH=. python scripts/run_relative_pose.py --model my_model --split easy \
    --pairs-source on_the_fly --device cuda
```

## Adapter contract

Every model adapter is a **callable class**. The exact signature
depends on the task, but the pattern is identical:

### Depth adapter

```python
class MyDepthModel:
    """Receives RGB, returns metric depth in metres."""

    native_alignment: str = "none"  # "none" = metric, "ls_affine" = relative
    native_precision: str = "fp16"

    def __init__(self, device: str = "cuda", batch_size: int = 1):
        self.model = ...  # load your model here

    @property
    def torch_module(self):
        """Expose the nn.Module for profiling (params, FLOPs). Optional."""
        return self.model

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Args:
            images: list of H×W×3 uint8 RGB arrays

        Returns:
            list of H×W float32 depth maps in metres
        """
        ...
```

### Pose adapter

```python
class MyPoseModel:
    """Receives an image pair, returns relative R and t."""

    native_alignment: str = "none"  # "none" = metric t, "unit" = up-to-scale
    native_precision: str = "fp16"

    def __init__(self, device: str = "cuda", batch_size: int = 1):
        self.model = ...

    @property
    def torch_module(self):
        return self.model

    def __call__(self, pairs: list[dict]) -> list[dict]:
        """
        Args:
            pairs: list of {"rgb_a": H×W×3, "rgb_b": H×W×3}

        Returns:
            list of {"rotation": 3×3 float64, "translation": (3,) float64}
        """
        ...
```

### Segmentation adapter

```python
class MySegModel:
    """Receives RGB, returns instance mask."""

    def __init__(self, device: str = "cuda", batch_size: int = 1):
        self.model = ...

    @property
    def torch_module(self):
        return self.model

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Args:
            images: list of H×W×3 uint8 RGB arrays

        Returns:
            list of H×W int32 instance masks (pixel value = instance ID)
        """
        ...
```

### Detection adapter

```python
class MyDetector:
    """Receives RGB, returns boxes + labels + scores."""

    def __init__(self, device: str = "cuda", batch_size: int = 1):
        self.model = ...

    @property
    def torch_module(self):
        return self.model

    def __call__(self, images: list[np.ndarray]) -> list[dict]:
        """
        Returns:
            list of {"boxes": N×4 float32, "labels": list[str], "scores": N float32}
        """
        ...
```

## Registration

Each task has a `scripts/<task>_models/__init__.py` with a
`MODEL_REGISTRY` dict:

```python
# scripts/pose_models/__init__.py

def _build_my_model(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .my_model import MyPoseModel
    return MyPoseModel(device=device, batch_size=batch_size, **kwargs)

MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    ...
    "my_model": _build_my_model,  # ← add this line
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    ...
    "my_model": "My-Model-v1",    # ← human-readable name for reports
}
```

**That's it.** The builder is a one-line lazy import + construction.
Dependencies are checked at import time — if `torch` or `kornia` isn't
installed, the adapter raises a clear `ImportError` with install
instructions.

## Profiling (automatic)

The runner automatically profiles your model:

- **Params (M)**: counted from `torch_module.parameters()` if you
  expose `torch_module`
- **FLOPs (G)**: measured via `torch.utils.flop_counter.FlopCounterMode`
  on the first batch
- **Latency**: wall-clock p50/p95/p99 across all batches

No extra code needed. Just expose `torch_module` as a property.

For models without a torch module (classical methods like OpenCV SIFT),
return `None` — params and FLOPs will be reported as `N/A`.

## Using ModelProfiler directly (optional)

For custom scripts or notebooks:

```python
from rpx_benchmark.model_profiler import ModelProfiler

profiler = ModelProfiler(my_adapter_or_torch_module)
eff = profiler.pre_run_profile()
# ... run inference ...
# ... (runner fills eff.latency_*, eff.flops_g, eff.peak_*) ...
report = profiler.full_report(eff)
print(report.summary_line())
# → "524.4M params | 187.3G FLOPs | 142.7ms p50"
```

## Or use the library adapter directly (no scripts)

If you prefer the pip-installed library over the scripts:

```python
import rpx_benchmark as rpx

# Wrap any numpy callable
def my_depth_fn(rgb: np.ndarray) -> np.ndarray:
    return ...  # H×W float32 metres

model = rpx.make_numpy_depth_model(my_depth_fn, name="my_model")
result = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(model=model))
```

Available wrappers:
- `make_numpy_depth_model(fn)` — monocular depth
- `make_numpy_mask_model(fn)` — instance segmentation
- `make_numpy_detection_model(fn)` — object detection
- `make_numpy_pose_model(fn)` — relative pose
- `make_numpy_grounding_model(fn)` — visual grounding
- `make_numpy_keypoint_model(fn)` — keypoint matching
- `make_numpy_sparse_depth_model(fn)` — sparse depth
- `make_numpy_nvs_model(fn)` — novel view synthesis
- `make_numpy_tracking_model(fn)` — object tracking

## Checklist

- [ ] Adapter class with `__call__` matching the task contract
- [ ] `torch_module` property (return `None` for non-torch models)
- [ ] `native_alignment` / `native_precision` class attributes
- [ ] Builder function in `__init__.py`
- [ ] One line in `MODEL_REGISTRY`
- [ ] One line in `MODEL_DISPLAY_NAMES`
- [ ] `ImportError` with install instructions if deps missing
- [ ] Test: `PYTHONPATH=. python scripts/run_<task>.py --model my_model --split easy --max-samples 5`
