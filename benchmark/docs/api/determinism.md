# Determinism (`rpx_benchmark.determinism`)

Best-effort cross-backend seeding for reproducible benchmark runs.
Covers Python ``random``, NumPy, PyTorch (CPU + CUDA + MPS), and
``transformers.set_seed`` where present.

```python
from rpx_benchmark import seed_all, deterministic

seed_all(42)                 # imperative, persistent

with deterministic(42):      # context manager, restores outer RNG state
    result = runner.run()
```

::: rpx_benchmark.determinism
    options:
      show_root_toc_entry: false
      members_order: source
