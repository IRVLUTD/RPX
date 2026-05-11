# RPX — Computation Log

Track all experiment runs here. Update as results land.

## Status Key
- ⬜ Not started
- 🔄 Running
- ✅ Done
- ❌ Failed (see notes)
- ⏭️ Skipped

---

## Dataset Statistics (no model needed)

| Statistic | Status | Value | Notes |
|-----------|--------|-------|-------|
| Objects/frame — Clutter | ⬜ | — | |
| Objects/frame — Interaction | ⬜ | — | |
| Objects/frame — Clean | ⬜ | — | |
| Occlusion rate — Clutter | ⬜ | — | |
| Occlusion rate — Interaction | ⬜ | — | |
| Occlusion rate — Clean | ⬜ | — | |
| Depth invalid — Clutter | ⬜ | — | |
| Depth invalid — Interaction | ⬜ | — | |
| Depth invalid — Clean | ⬜ | — | |
| Median Δt (cm/frame) | ⬜ | — | |
| Median ΔR (°/frame) | ⬜ | — | |
| 90th-percentile jerk | ⬜ | — | |

---

## Model Runs

### Monocular Depth (P1)

| Model | Status | S_overall | STR_C→I | STR_I→L | TS | JSON path |
|-------|--------|-----------|---------|---------|----|----|
| DA-V2 Large | ⬜ | — | — | — | — | |
| ZoeDepth | ⬜ | — | — | — | — | |
| Depth Pro | ⬜ | — | — | — | — | |
| DA-V2 Base | ⬜ | — | — | — | — | |
| Video DA | ⬜ | — | — | — | — | |
| UniDepth V2 | ⬜ | — | — | — | — | |
| Metric3D V2 | ⬜ | — | — | — | — | |
| Prompt DA | ⬜ | — | — | — | — | |
| MiDaS 3.1 | ⬜ | — | — | — | — | |

### Detection (P2)

| Model | Status | S_overall | STR_C→I | STR_I→L | JSON path |
|-------|--------|-----------|---------|---------|-----------|
| GroundingDINO | ⬜ | — | — | — | |
| OWL-ViT v2 | ⬜ | — | — | — | |
| YOLO-World | ⬜ | — | — | — | |
| DITO | ⬜ | — | — | — | |
| RO-ViT | ⬜ | — | — | — | |

### Visual Grounding (P2)

| Model | Status | S_overall | STR_C→I | STR_I→L | JSON path |
|-------|--------|-----------|---------|---------|-----------|
| GroundingDINO | ⬜ | — | — | — | |
| Qwen2-VL | ⬜ | — | — | — | |
| CogVLM2 | ⬜ | — | — | — | |

### Instance Segmentation (P3)

| Model | Status | S_overall | STR_C→I | STR_I→L | TS | SGC | JSON path |
|-------|--------|-----------|---------|---------|----|----|-----------|
| SAM2 Large | ⬜ | — | — | — | — | — | |
| SAM2 Small | ⬜ | — | — | — | — | — | |
| Mask2Former | ⬜ | — | — | — | — | — | |
| OneFormer | ⬜ | — | — | — | — | — | |
| iTeach-UOIS | ⬜ | — | — | — | — | — | |

### Tracking (P3)

| Model | Status | MOTA | IDF1 | JSON path |
|-------|--------|------|------|-----------|
| SAM2 video | ⬜ | — | — | |
| Cutie/DEVA | ⬜ | — | — | |

### Relative Pose (P4)

| Model | Status | S_overall | STR_C→I | STR_I→L | JSON path |
|-------|--------|-----------|---------|---------|-----------|
| MicKey | ⬜ | — | — | — | |
| DUSt3R | ⬜ | — | — | — | |
| MASt3R | ⬜ | — | — | — | |
| PoseNet | ⬜ | — | — | — | |

### Keypoint Correspondences (P4)

| Model | Status | S_overall | STR_C→I | STR_I→L | JSON path |
|-------|--------|-----------|---------|---------|-----------|
| LoFTR | ⬜ | — | — | — | |
| eLoFTR | ⬜ | — | — | — | |
| SP+LightGlue | ⬜ | — | — | — | |
| SIFT | ⬜ | — | — | — | |
| DeDoDe v2 | ⬜ | — | — | — | |

### Novel-View Synthesis (P5)

| Model | Status | PSNR | SSIM | LPIPS | JSON path |
|-------|--------|------|------|-------|-----------|
| 3DGS | ⬜ | — | — | — | |
| Nerfacto | ⬜ | — | — | — | |

---

## Deployment Metrics & Analysis

| Analysis | Status | Key result | Notes |
|----------|--------|------------|-------|
| TS computation (depth) | ⬜ | — | |
| TS computation (seg) | ⬜ | — | |
| STR all tasks | ⬜ | — | |
| SGC all tasks | ⬜ | — | |
| ESD weight MI computation | ⬜ | — | |
| ESD Spearman validation | ⬜ | — | |
| ESD sensitivity (1000 perturbations) | ⬜ | — | |
| Cross-task ranking agreement | ⬜ | — | |
| Surprising finding identification | ⬜ | — | |

## Figures

| Figure | Status | Notes |
|--------|--------|-------|
| Fig 1: Overview (exists) | ✅ | |
| Fig 2: STR scatter | ⬜ | |
| Fig 3: ESD violins | ⬜ | |
| Fig 4: Phase heatmap | ⬜ | |
| Fig 5: Deployment radar | ⬜ | |
| Fig S1: Motion stats | ⬜ | |
