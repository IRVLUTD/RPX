# `reference.adapters.depth_unidepth`

Reference adapter for [UniDepth V2](https://github.com/lpiccinelli-eth/UniDepth).
UniDepth bypasses the standard `AutoModelForDepthEstimation` API and
ships its own `UniDepthV2` class with an `.infer(rgb, camera=None)`
forward. This adapter handles that calling convention via a custom
invoker.

## Install

```bash
pip install 'rpx-benchmark[depth-unidepth]'
pip install 'unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git'
```

## Usage

```python
import numpy as np
from rpx_benchmark.reference.adapters.depth_unidepth import make_unidepth_v2_model

# Optional: override UniDepth's self-prompted intrinsics with real calibration.
K = np.array([[605.0, 0, 320.0], [0, 605.0, 240.0], [0, 0, 1]], dtype=np.float32)

bm = make_unidepth_v2_model(
    "lpiccinelli/unidepth-v2-vitl14",
    device="cuda",
    camera_k=K,
)
```

::: rpx_benchmark.reference.adapters.depth_unidepth
    options:
      show_root_toc_entry: false
      members_order: source
