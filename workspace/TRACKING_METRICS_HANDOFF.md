# RPX Tracking Metrics (Frame Budget 250)

Metrics are macro-averaged over evaluation cells and shown as percentages except IDSW, latency, and FPS. Higher is better for HOTA/DetA/AssA/IDF1/MOTA; lower is better for IDSW and latency.

## MOS

| Model | Cells | Scenes | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM2Long | 300 | 100 | 95.96 | 95.15 | 96.84 | 98.48 | 97.08 | 196 | 699.58 | 1.43 |
| DAM4SAM | 300 | 100 | 95.52 | 94.45 | 96.70 | 97.94 | 96.08 | 291 | 376.85 | 2.65 |
| SAM2 | 300 | 100 | 95.27 | 94.24 | 96.40 | 97.98 | 96.15 | 209 | 96.98 | 10.31 |
| SAM2-Plus | 300 | 100 | 93.36 | 91.85 | 95.07 | 97.10 | 94.43 | 185 | 109.86 | 9.10 |
| EdgeTAM | 300 | 100 | 93.07 | 91.73 | 94.57 | 97.16 | 94.50 | 207 | 42.81 | 23.36 |
| Cutie | 300 | 100 | 93.00 | 91.66 | 94.50 | 97.64 | 95.44 | 188 | 24.43 | 40.94 |
| XMem | 300 | 100 | 90.56 | 88.57 | 92.84 | 95.86 | 92.13 | 179 | 18.35 | 54.49 |
| MiTS | 300 | 100 | 88.71 | 86.44 | 91.32 | 94.80 | 90.07 | 296 | 15.92 | 62.80 |

## Ego

| Model | Cells | Scenes | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM2Long | 100 | 100 | 95.23 | 94.98 | 95.58 | 97.25 | 95.27 | 173 | 965.96 | 1.04 |
| DAM4SAM | 100 | 100 | 94.19 | 93.79 | 94.70 | 96.50 | 93.77 | 270 | 441.88 | 2.26 |
| SAM2 | 100 | 100 | 90.37 | 88.79 | 92.14 | 93.83 | 88.78 | 200 | 87.78 | 11.39 |
| Cutie | 100 | 100 | 90.35 | 89.26 | 91.61 | 95.01 | 91.05 | 154 | 24.99 | 40.01 |
| EdgeTAM | 100 | 100 | 88.39 | 87.19 | 89.74 | 93.37 | 87.68 | 184 | 34.70 | 28.82 |
| XMem | 100 | 100 | 86.81 | 85.00 | 88.89 | 92.44 | 86.14 | 149 | 23.40 | 42.74 |
| MiTS | 100 | 100 | 82.67 | 80.82 | 84.89 | 88.54 | 78.52 | 373 | 23.32 | 42.88 |
| SAM2-Plus | 100 | 100 | 81.34 | 77.72 | 85.38 | 87.29 | 78.49 | 75 | 108.49 | 9.22 |

## Per-scene data

Each row contains `model`, `scene`, `phase`, `split` (difficulty), HOTA, DetA, AssA, IDF1, MOTA, IDSW, latency, throughput, VRAM, and parameter count.

MOS and Ego are complete for all eight models above. A separate partial MOS MOTIP run contains only 35/300 cells and is excluded.
