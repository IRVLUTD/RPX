# `reference.adapters.depth_metric3d`

Reference adapter for [Metric3D V2](https://github.com/YvanYin/Metric3D)
loaded via `torch.hub`. The model is trained at a canonical focal
length, so recovering real metric depth needs the
`(fx_real / fx_canonical)` rescale and a letterbox un-pad on the
output side.

!!! warning "CUDA-only"
    Metric3D's upstream decoder hardcodes `device="cuda"` inside
    `torch.linspace`. Running on CPU fails with an assertion six
    frames deep. The adapter refuses non-CUDA devices at construction
    time with an `AdapterError` that names the upstream file.

## Install

```bash
pip install 'rpx-benchmark[depth-metric3d]'
```

## Usage

```python
from rpx_benchmark.reference.adapters.depth_metric3d import make_metric3d_v2_model

bm = make_metric3d_v2_model(
    device="cuda",
    fx_real=605.0,                    # D435 640×480 focal, approx.
    entry="metric3d_vit_large",
)
```

::: rpx_benchmark.reference.adapters.depth_metric3d
    options:
      show_root_toc_entry: false
      members_order: source
