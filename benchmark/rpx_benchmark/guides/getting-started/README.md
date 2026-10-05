# Get started

## Install

Python 3.10 or newer is required. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install 'rpx-benchmark[hub,schemas]'
```

The core package supplies loaders, adapters, metrics, and reporting. The `hub` extra adds Hugging Face downloading and Parquet support; `schemas` adds manifest validation. Install model dependencies separately or use a reference model container from the repository.

## Discover tasks and metrics

```python
from rpx_benchmark.tasks import available_tasks
from rpx_benchmark.metrics import available_metrics

print([task.value for task in available_tasks()])
print(available_metrics())
```

## Run image depth

With an existing RPX manifest and a model callable that returns an `H × W` NumPy depth array in metres:

```python
from pathlib import Path
import rpx_benchmark as rpx

# Replace this with your actual model import.
from my_model import predict_depth

model = rpx.make_numpy_depth_model(predict_depth)
cfg = rpx.MonocularDepthRunConfig(
    manifest=Path('manifest.json'),
    out_dir=Path('results/my-depth-model'),
    model=model,
    max_samples=10,
)
result, report, paths = rpx.run_monocular_depth(cfg)
print(result.aggregated)
print(paths)
```

`my_model` is your code, not an installed RPX module. A small `max_samples` is useful for checking integration; a full benchmark needs the complete protocol and split.

## Inspect results

The result contains `per_sample`, `aggregated`, and `num_samples`. The returned `paths` identify generated reports and cell records. Check that the sample count and scene/phase IDs match your manifest before comparing model scores.

[Connect a model](../models/README.md) or [add a metric](../metrics/README.md).
