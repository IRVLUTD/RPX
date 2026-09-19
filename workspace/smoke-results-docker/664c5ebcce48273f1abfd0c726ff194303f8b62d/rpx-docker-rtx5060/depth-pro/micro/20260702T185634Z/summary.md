# RPX benchmark — monocular_depth

- **Model:** `Depth Pro`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 1

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.4891 |
| absrel | 0.1425 |
| delta1 | 0.9099 |
| delta2 | 0.9843 |
| delta3 | 0.9942 |
| latency_ms | 1429.8867 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.1425 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0475** |
| Δ interaction (S_I − S_C) | -0.1425 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.1425
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 1.0000

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 1429.9 |
| peak memory (MB) | 3776.8 |
