# Installation

## Recommended

```bash
pip install 'rpx-benchmark[hub]'
```

This pulls the core library plus `huggingface_hub` for pulling RPX
dataset splits. Enough to run any task once you bring your own model.

## From source

```bash
git clone https://github.com/IRVLUTD/RPX.git
cd RPX/benchmark
pip install -e '.[hub,hf-datasets,schemas,dev,docs]'
```

## Optional extras

| Extra | What it installs | When to use |
|---|---|---|
| *(core)* | `numpy`, `Pillow` | Always — manifest loading, metric computation |
| `hub` | `huggingface_hub[hf_xet]` | Pulling datasets via modality-aware snapshots |
| `hf-datasets` | `datasets>=2.18` | One-line `rpx_benchmark.data.load_hf(task, split)` |
| `schemas` | `pydantic>=2.5` | Strict manifest validation + JSON Schema export |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`, `pre-commit` | Contributing |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs locally |

## No model dependencies

The toolkit ships **no models** and no model-specific extras — you
bring your own model as a `BenchmarkableModel`. Any torch /
transformers / JAX / cloud-API dependency is yours to manage in your
own environment alongside `rpx-benchmark`.

## Verify your install

```python
import numpy as np
import rpx_benchmark as rpx


def fake_depth(rgb):
    return np.full(rgb.shape[:2], 2.0, dtype=np.float32)


bm = rpx.make_numpy_depth_model(fake_depth)
print(f"BenchmarkableModel ready: {bm.name}")
```

Any exit without an ImportError means the core is healthy.
