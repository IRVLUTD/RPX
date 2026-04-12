# `reference.adapters.depth_hf`

Reference HuggingFace depth adapter — works with any checkpoint
loadable via `transformers.AutoModelForDepthEstimation` whose image
processor exposes `post_process_depth_estimation`.

Known compatible families:

- Depth Anything V2 (metric) — [depth-anything/Depth-Anything-V2-Metric-*-hf](https://huggingface.co/depth-anything)
- Depth Pro — [apple/DepthPro-hf](https://huggingface.co/apple/DepthPro-hf)
- ZoeDepth — [Intel/zoedepth-nyu-kitti](https://huggingface.co/Intel/zoedepth-nyu-kitti)
- Video Depth Anything (per-frame), PromptDA (without the prompt signal)

## Install

```bash
pip install 'rpx-benchmark[depth-hf]'
```

## Usage

```python
from rpx_benchmark.reference.adapters.depth_hf import make_hf_depth_model

bm = make_hf_depth_model(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    device="cuda",
)
```

For the full set of knobs (precision casting, custom name) see the
API reference below.

::: rpx_benchmark.reference.adapters.depth_hf
    options:
      show_root_toc_entry: false
      members_order: source
