# Installation

## Recommended

```bash
pip install 'rpx-benchmark[depth]'
```

This pulls the core library, the HuggingFace downloader, the rich
terminal UI, `torch`, `torchvision`, and `transformers` — enough to
run the full monocular absolute depth slate from the CLI.

## From source

```bash
git clone https://github.com/IRVLUTD/RPX.git
cd RPX/benchmark
pip install -e '.[depth,dev,docs]'
```

## Optional extras matrix

| Extra | What it installs | When to use |
|---|---|---|
| *(core)* | `numpy`, `Pillow` | Always — manifest loading, metric computation |
| `hub` | `huggingface_hub[hf_xet]` | Pulling datasets from HuggingFace |
| `ui` | `rich` | Pretty terminal output (progress bar, tables) |
| `depth-hf` | `torch`, `torchvision`, `transformers`, `accelerate` | Any HuggingFace depth model (DA-v2, Depth Pro, ZoeDepth, PromptDA) |
| `depth-unidepth` | `depth-hf` + `einops`, `timm`, `huggingface_hub` | UniDepth V2 |
| `depth-metric3d` | `depth-hf` + `timm`, `mmcv-lite`, `mmengine` | Metric3D V2 (CUDA only) |
| `depth` | `depth-hf` + `ui` | **Recommended default** for monocular depth |
| `depth-all` | everything above | Monocular depth, full slate |
| `dev` | `pytest`, `ruff` | Contributing |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs locally |

## Models that need manual installs

### UniDepth V2

Ships via GitHub (not PyPI):

```bash
pip install 'rpx-benchmark[depth-unidepth]'
pip install 'unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git'
```

### Metric3D V2

```bash
pip install 'rpx-benchmark[depth-metric3d]'
# Weights are downloaded via torch.hub on first use; no install step.
```

!!! warning "CUDA-only"
    Metric3D V2's upstream decoder hardcodes
    `torch.linspace(..., device="cuda")`, so CPU inference is not
    supported. The adapter raises a clean error on CPU rather than
    failing mid-inference.

## Verify your install

```bash
rpx --help
rpx ls                 # lists tasks + splits
rpx models             # lists runnable + deferred model adapters
```

All three should exit 0. If you installed `[depth]` you can also run
a synthetic smoke without downloading a real dataset:

```bash
python -c "
import rpx_benchmark as rpx
import numpy as np

def fake_depth(rgb):
    return np.full(rgb.shape[:2], 2.0, dtype=np.float32)

bm = rpx.make_numpy_depth_model(fake_depth)
print(f'BenchmarkableModel ready: {bm.name}')
"
```
