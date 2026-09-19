# RPX benchmark — monocular_depth

- **Model:** `Depth Pro`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 25

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.5981 |
| absrel | 0.1390 |
| delta1 | 0.9092 |
| delta2 | 0.9636 |
| delta3 | 0.9860 |
| latency_ms | 1041.8351 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.1390 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0463** |
| Δ interaction (S_I − S_C) | -0.1390 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.1390
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 0.9749

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 1009.6 |
| peak memory (MB) | 3776.8 |
