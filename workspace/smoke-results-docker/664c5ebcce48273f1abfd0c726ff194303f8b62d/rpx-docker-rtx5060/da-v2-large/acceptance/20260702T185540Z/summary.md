# RPX benchmark — monocular_depth

- **Model:** `DA-V2 Large`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 25

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.4864 |
| absrel | 0.0647 |
| delta1 | 0.9456 |
| delta2 | 0.9829 |
| delta3 | 0.9947 |
| latency_ms | 321.7039 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.0647 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0216** |
| Δ interaction (S_I − S_C) | -0.0647 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.0647
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 0.9836

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 309.8 |
| peak memory (MB) | 1909.9 |
