# RPX benchmark — monocular_depth

- **Model:** `UniDepth V2`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 1

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.3500 |
| absrel | 0.0675 |
| delta1 | 0.9658 |
| delta2 | 0.9904 |
| delta3 | 0.9947 |
| latency_ms | 1307.2321 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.0675 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0225** |
| Δ interaction (S_I − S_C) | -0.0675 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.0675
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 1.0000

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 1307.2 |
| peak memory (MB) | 2641.0 |
