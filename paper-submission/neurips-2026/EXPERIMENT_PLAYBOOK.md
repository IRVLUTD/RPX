# RPX Experiment Playbook — 5 People × 8 Tasks × 11 Days

**Start**: ~Apr 25 · **Hard stop**: May 5 (day before submission) · **Paper due**: May 6 AoE

---

## Task Allocation (5 people → 8 tasks)

The grouping is by computational similarity and shared infrastructure, not topic.

| Person | Tasks | Models (target count) | Priority |
|--------|-------|-----------------------|----------|
| **P1 — Depth lead** | Monocular Depth + Sparse Depth | 9 + 2 = **11 model runs** | 🔴 HIGHEST |
| **P2 — Detection lead** | Open-Vocab Detection + Visual Grounding | 5 + 3 = **8 model runs** | 🔴 HIGH |
| **P3 — Segmentation + Tracking lead** | Instance Segmentation + Tracking | 5 + 2 = **7 model runs** | 🔴 HIGH |
| **P4 — Geometry lead** | Relative Pose + Keypoint Correspondences | 4 + 5 = **9 model runs** | 🟡 MEDIUM |
| **P5 — NVS + Integration lead** | NVS + Deployment metrics + Downstream validation | 2 + integration = **2 model runs + analysis** | 🟡 MEDIUM |

**Total target: ~47 model evaluations across 8 tasks.**

---

## Per-Person Model Slates

### P1 — Depth (11 models)

These are the most important results in the paper. The depth task has the best-justified
model slate, the clearest connection to robot deployment, and will anchor the main
results table and the STR/TS analysis.

**Monocular Depth (9 models):**

| Model | Why | HF checkpoint / source | Priority |
|-------|-----|------------------------|----------|
| Depth Anything V2 — Base | Scaling anchor (small) | `depth-anything/Depth-Anything-V2-Small-hf` | 🔴 |
| Depth Anything V2 — Large | De facto robot-learning default | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | 🔴 |
| Video Depth Anything | Temporal consistency focus | `depth-anything/Video-Depth-Anything-Large-hf` (check) | 🔴 |
| Depth Pro | Sharp boundaries, metric | Apple `depth-pro` | 🔴 |
| UniDepth V2 | Intrinsics-free, uncertainty | `lpiccinelli/UniDepth-V2-ViTL14` (check) | 🟡 |
| Metric3D V2 | Monocular SLAM standard | GitHub release | 🟡 |
| Prompt Depth Anything | Sparse-depth prompted, robot-focused | GitHub release | 🟡 |
| ZoeDepth | Legacy anchor (3D Diffusion Policy) | `Intel/zoedepth-nyu-kitti` | 🟡 |
| MiDaS 3.1 | Pre-2023 reference | `Intel/dpt-large` | ⚪ appendix |

**Sparse Depth (2 models):**

| Model | Source | Priority |
|-------|--------|----------|
| Prompt Depth Anything (sparse mode) | Same as above | 🟡 |
| Depth Anything V2 + sparse prompt | Adapter | ⚪ stretch |

**Execution order**: DAv2-Large → ZoeDepth → Depth Pro → DAv2-Base → Video DA → UniDepth → Metric3D → Prompt DA → MiDaS.
(Run the two 🔴 models first so we have *something* in the table by Day 2.)

### P2 — Detection + Grounding (8 models)

**Open-Vocab Detection (5 models):**

| Model | HF checkpoint / source | Priority |
|-------|------------------------|----------|
| GroundingDINO | `IDEA-Research/grounding-dino-*` | 🔴 |
| OWL-ViT v2 | `google/owlv2-large-patch14-ensemble` | 🔴 |
| YOLO-World | `ultralytics` or HF | 🔴 |
| DITO | GitHub | 🟡 |
| RO-ViT | GitHub | ⚪ stretch |

**Visual Grounding (3 models):**

| Model | Source | Priority |
|-------|--------|----------|
| GroundingDINO (grounding mode) | Same checkpoint | 🔴 |
| Qwen-VL / Qwen2-VL | HF `Qwen/Qwen2-VL-7B-Instruct` | 🟡 |
| CogVLM2 | HF | ⚪ stretch |

### P3 — Segmentation + Tracking (7 models)

**Instance Segmentation (5 models):**

| Model | Source | Priority |
|-------|--------|----------|
| SAM2 — Large | `facebook/sam2.1-hiera-large` | 🔴 |
| SAM2 — Small | `facebook/sam2.1-hiera-small` | 🔴 |
| Mask2Former (Swin-T) | `facebook/mask2former-swin-tiny-coco-instance` | 🟡 |
| OneFormer (Swin-L) | `shi-labs/oneformer_coco_swin_large` | 🟡 |
| iTeach-UOIS | Custom (lab model) | 🔴 |

**Tracking (2 models):**

| Model | Source | Priority |
|-------|--------|----------|
| SAM2 (video propagation) | Same SAM2 checkpoint | 🔴 |
| Cutie / DEVA | GitHub | ⚪ stretch |

### P4 — Geometry: Pose + Keypoints (9 models)

**Relative Camera Pose (4 models):**

| Model | Source | Priority |
|-------|--------|----------|
| MicKey | GitHub | 🔴 |
| DUSt3R | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` | 🔴 |
| MASt3R | HF | 🟡 |
| PoseNet (baseline) | torchvision | ⚪ appendix |

**Keypoint Correspondences (5 models):**

| Model | Source | Priority |
|-------|--------|----------|
| LoFTR | GitHub/kornia | 🔴 |
| eLoFTR | GitHub | 🔴 |
| SuperPoint + LightGlue | GitHub `cvg/LightGlue` | 🟡 |
| SIFT + NN matcher | OpenCV | 🟡 (classical baseline) |
| DeDoDe v2 | GitHub | ⚪ stretch |

### P5 — NVS + Integration (2 models + all deployment analysis)

**Novel-View Synthesis (2 models):**

| Model | Source | Priority |
|-------|--------|----------|
| 3D Gaussian Splatting | `graphdeco-inria/gaussian-splatting` | 🔴 |
| Nerfacto (nerfstudio) | `nerfstudio-project/nerfstudio` | 🟡 |

**Integration duties** (P5 is also the analysis person):
- Compute TS, STR, SGC from all task results as they come in
- Generate all paper figures (scripts prepared in advance — see below)
- Run ESD weight computation + validation
- Run downstream validation experiment if time permits
- Compile final results tables

---

## Standardised Output Protocol

**Every person must output results in the same format** so P5 can integrate immediately.

Each model run produces a JSON file:

```
results/<task>/<model_name>.json
```

```json
{
  "model": "depth-anything-v2-large",
  "task": "monocular_depth",
  "checkpoint": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
  "params_M": 335.3,
  "flops_G": 210.7,
  "timestamp": "2026-04-26T14:30:00Z",
  "per_sample": [
    {
      "id": "scene_001_clutter_frame_042",
      "scene": "scene_001",
      "phase": "clutter",
      "difficulty": "hard",
      "metrics": {"abs_rel": 0.123, "rmse": 0.456, "delta_1": 0.789},
      "latency_ms": 45.2,
      "prediction_path": "predictions/depth/da-v2-large/scene_001_clutter_042.npy"
    }
  ],
  "aggregate": {
    "overall": {"abs_rel": 0.134, "rmse": 0.478, "delta_1": 0.812},
    "by_phase": {
      "clutter": {"abs_rel": 0.145},
      "interaction": {"abs_rel": 0.167},
      "clean": {"abs_rel": 0.112}
    },
    "by_difficulty": {
      "easy": {"abs_rel": 0.098},
      "medium": {"abs_rel": 0.134},
      "hard": {"abs_rel": 0.189}
    }
  }
}
```

**Key**: Include `per_sample` with scene/phase/difficulty metadata.
This lets P5 compute TS, STR, SGC, ESD, and weighted phase scores from the raw data.

Save predictions (masks, depth maps, boxes) so TS and SGC can be computed post-hoc.

---

## 11-Day Schedule

### Days 1–3 (Apr 25–27): First wave — 🔴 priority models only

| Person | Target | Output |
|--------|--------|--------|
| P1 | DAv2-Large, ZoeDepth, Depth Pro | 3 depth JSONs |
| P2 | GroundingDINO, OWL-ViT | 2 detection + 1 grounding JSON |
| P3 | SAM2-Large, iTeach-UOIS | 2 segmentation JSONs |
| P4 | MicKey, LoFTR, eLoFTR | 1 pose + 2 keypoint JSONs |
| P5 | 3DGS setup + run | 1 NVS JSON (may take longer) |

**Checkpoint**: By end of Day 3, we should have **≥12 model results**.
This is already enough to populate the main results table with real numbers.

### Days 4–6 (Apr 28–30): Second wave — 🟡 models + deployment metrics

| Person | Target |
|--------|--------|
| P1 | DAv2-Base, Video DA, UniDepth, Metric3D |
| P2 | YOLO-World, DITO, Qwen-VL |
| P3 | SAM2-Small, Mask2Former, OneFormer, SAM2 tracking |
| P4 | DUSt3R, MASt3R, SuperPoint+LightGlue, SIFT |
| P5 | **Compute TS, STR, SGC** from all Day 1–3 results. Start figures. |

**Checkpoint**: By end of Day 6, we should have **≥30 model results + deployment metrics for first wave**.

### Days 7–9 (May 1–3): Third wave — stretch models + analysis

| Person | Target |
|--------|--------|
| P1 | Prompt DA, MiDaS, sparse depth models |
| P2 | RO-ViT, CogVLM2 |
| P3 | Cutie/DEVA tracking |
| P4 | DeDoDe, PoseNet |
| P5 | **All figures done. ESD validation. Downstream experiment.** |

**Checkpoint**: By end of Day 9, **≥40 model results, all figures, ESD validated**.

### Days 10–11 (May 4–5): Integration + writing blitz

| Day | All hands |
|-----|-----------|
| 10 (May 4) | **Abstract submitted.** All \todo{} filled. Figures inserted. |
| 11 (May 5) | Final proofread. Appendix tables. Croissant. HF sample. Anonymise. |
| May 6 | **Submit.** |

---

## Pre-Experiment Preparation (THIS WEEK, Apr 16–24)

Things to do NOW so experiments plug in seamlessly:

### 1. Dataset statistics (no model needed)
Compute from annotations/data directly:
- [ ] Objects per frame (mean) per phase
- [ ] Occlusion rate per phase
- [ ] Depth invalid rate per phase
- [ ] Camera motion statistics (Δt, ΔR, jerk) from T265 poses
- [ ] Scene diversity summary (environments, categories)

### 2. Figure templates
Create Python scripts with placeholder data that generate each figure.
When real data arrives, swap in real numbers and re-run:
- [ ] `fig_str_scatter.py` — STR vs standard accuracy
- [ ] `fig_esd_violins.py` — per-task performance by ESD level
- [ ] `fig_phase_heatmap.py` — models × phases
- [ ] `fig_deployment_radar.py` — multi-axis radar for top models
- [ ] `fig_motion_stats.py` — RPX motion vs robot platform envelopes

### 3. Results aggregation script
- [ ] Script that reads all `results/<task>/<model>.json` files
- [ ] Computes weighted phase scores, STR, overall rankings
- [ ] Generates LaTeX table source for direct paste into paper
- [ ] Generates per-phase, per-ESD breakdown tables

### 4. Deployment metrics pipeline
- [ ] Verify `compute_str`, `compute_ts`, `compute_sgc` work on synthetic data
- [ ] Write wrapper that takes per-sample predictions + GT and outputs deployment report
- [ ] Test on 1 scene with dummy predictions

### 5. LaTeX preparation
- [ ] Create all figure placeholders in paper (`\includegraphics` with `example-image`)
- [ ] Create appendix table skeletons
- [ ] Verify compilation with current `neurips_2026.sty`
- [ ] Draft Croissant metadata JSON skeleton

### 6. Runbook per person
- [ ] Each person gets a 1-page doc: their models, checkpoints, exact CLI commands, output format, where to save results

---

## The Surprising Finding — What to Look For

As results come in (especially after Day 3), P5 should immediately check:

1. **Does standard-benchmark accuracy predict RPX deployment score?**
   - Scatter plot: published accuracy (e.g., NYU AbsRel for depth) vs RPX S_overall
   - If low correlation → **"The Robustness Inversion"**

2. **Do model rankings change between Easy and Hard ESD?**
   - Kendall τ between Easy ranking and Hard ranking
   - If τ < 0.5 → **"The ESD Reversal"**

3. **Does STR_{C→I} correlate with accuracy?**
   - If not → static accuracy doesn't predict interaction robustness

4. **Does any model have high accuracy but low TS?**
   - → "Flickering Giant" — accurate per-frame but temporally inconsistent

5. **Does SGC drop uniformly during interaction?**
   - If yes → geometric coherence is systematically fragile during manipulation

**Name whatever you find.** A named finding is 10× more memorable than a p-value.

---

## Minimum Viable Paper (if things go wrong)

If experiments run late, here's the survival plan:

**Absolute minimum** (still submittable):
- 3 tasks fully evaluated: Depth (7+ models), Detection (4+ models), Segmentation (3+ models)
- Deployment metrics (TS, STR, SGC) for depth and segmentation
- ESD validation on depth task
- 3 analytical figures
- All prose finalized

**Stretch** (best paper territory):
- All 8 tasks evaluated (40+ models)
- Downstream validation experiment
- 5+ analytical figures
- Cross-task analysis (do rankings agree across tasks?)
- Full appendix with per-scene breakdowns

---

## Shared Infrastructure

**Results directory structure:**
```
RPX/
└── experiments/
    └── neurips-2026/
        ├── results/
        │   ├── monocular_depth/
        │   │   ├── depth-anything-v2-large.json
        │   │   ├── zoedepth.json
        │   │   └── ...
        │   ├── object_detection/
        │   ├── segmentation/
        │   ├── visual_grounding/
        │   ├── relative_pose/
        │   ├── keypoint_matching/
        │   ├── novel_view_synthesis/
        │   └── object_tracking/
        ├── predictions/          # raw model outputs for TS/SGC
        ├── figures/              # generated figure PNGs
        ├── tables/               # generated LaTeX table fragments
        ├── scripts/              # analysis + figure generation scripts
        └── COMPUTATION_LOG.md    # what's done, what's running, what failed
```
