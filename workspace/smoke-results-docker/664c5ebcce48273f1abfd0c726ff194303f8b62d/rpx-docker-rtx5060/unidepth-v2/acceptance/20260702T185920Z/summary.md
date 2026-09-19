# RPX benchmark — monocular_depth

- **Model:** `UniDepth V2`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 25

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.4978 |
| absrel | 0.0707 |
| delta1 | 0.9479 |
| delta2 | 0.9771 |
| delta3 | 0.9875 |
| latency_ms | 134.4731 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.0707 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0236** |
| Δ interaction (S_I − S_C) | -0.0707 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.0707
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 0.9819

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 114.9 |
| peak memory (MB) | 2647.4 |
