# RPX benchmark — monocular_depth

- **Model:** `MoGe-2 ViT-L`
- **Split:** `easy`
- **Repo:** `IRVLUTD/RPX`
- **Samples:** 25

## Aggregated metrics

| metric | value |
|---|---|
| rmse | 0.6235 |
| absrel | 0.1769 |
| delta1 | 0.7502 |
| delta2 | 0.9638 |
| delta3 | 0.9794 |
| latency_ms | 309.4479 |

## Weighted Phase Score

| phase | score |
|---|---|
| clutter | 0.1769 |
| interaction | 0.0000 |
| clean | 0.0000 |
| **overall** | **0.0590** |
| Δ interaction (S_I − S_C) | -0.1769 |
| Δ recovery    (S_L − S_I) | +0.0000 |

- **STR C→I (interaction drop):** -0.1769
- **STR I→L (recovery):**         +0.0000
- **Temporal stability (TS):** 0.9771

## Efficiency

| metric | value |
|---|---|
| latency (ms/sample) | 290.8 |
| peak memory (MB) | 2926.3 |
