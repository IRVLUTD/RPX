# RPX benchmark — monocular_depth

- **Model:** `DA-V2 Large`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 1

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.3918 |
| absrel | 0.0575 |
| delta1 | 0.9375 |
| delta2 | 0.9899 |
| delta3 | 0.9945 |
| latency_ms | 606.9197 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.0575 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0192** |
| Δ interaction (S_I − S_C) | -0.0575 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.0575
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 1.0000

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 606.9 |
| peak memory (MB) | 1909.2 |
