# RPX benchmark — monocular_depth

- **Model:** `MoGe-2 ViT-L`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 1

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.4504 |
| absrel | 0.1554 |
| delta1 | 0.9317 |
| delta2 | 0.9892 |
| delta3 | 0.9943 |
| latency_ms | 931.6261 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.1554 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0518** |
| Δ interaction (S_I − S_C) | -0.1554 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.1554
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 1.0000

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 931.6 |
| peak memory (MB) | 2925.2 |
