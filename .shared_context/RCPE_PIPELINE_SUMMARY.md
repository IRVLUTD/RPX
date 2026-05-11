# RPX — Full Pipeline Status (RCPE + NVS)

**Branch:** `jishnu/beautify-readmes`
**Latest commits:** `d7d5120` (RCPE+NVS code), `05c496d` (NVS metrics+PUS)
**Tests:** 518/518 passing

---

## Architecture Overview

RPX evaluates 7 perception tasks on the SAME 100 indoor+outdoor scenes under a three-phase protocol (Clutter → Interaction → Clean). Deployment readiness is reported along three axes (no composite score): task accuracy, scene-change robustness, compute cost.

---

## Committed Code

### New Library Modules (`benchmark/rpx_benchmark/`)

| File | Lines | What |
|---|---|---|
| `pose_pairs.py` | 655 | RCPE: on-the-fly deterministic pair generator. 3 pair types (intra-phase × 4 rotation bins, cross-phase Clutter↔Clean, temporal chains). ~60K pairs at full scale. Auto-extracts from HF tars, auto-downloads missing shards. Seeded, zero duplicates. |
| `pose_metrics.py` | 314 | RCPE: `evaluate_rcpe(per_pair)`. Standard: AUC@5°/10°/20°. Novel: Metric AUC@(θ°, d cm), cross-phase Δ, temporal drift, per-bin/per-type breakdowns. |
| `nvs_pairs.py` | 369 | NVS: on-the-fly evaluation sample generator. Context views (N=2,4,8,16) + target query + GT. Interpolation/extrapolation/cross-phase. ~25K samples at full scale. |
| `nvs_metrics.py` | 277 | NVS: `evaluate_nvs(per_sample)`. Standard: PSNR, SSIM. Novel: depth AbsRel/RMSE/δ<1.25 of rendered views vs D435 GT, Perceptual Utility Score (PUS), per-object PSNR via instance masks, per-type/per-context breakdowns. |
| `model_profiler.py` | 346 | Unified `ModelProfiler` for any task. Auto-discovers torch module. `pre_run_profile()` → `full_report(eff)`. |

### Modified Files

| File | Change |
|---|---|
| `evaluators.py` | `pose_metrics()` now returns 4 keys: added `translation_angular_deg` + `pose_error_max_deg` (fixes standard AUC — was computing on rotation only) |
| `run_relative_pose.py` | `--pairs-source on_the_fly`, `--skip-flops`, `_run_on_the_fly()`, writes `rcpe_metrics.json` with tiered efficiency |
| `local_manifest.py` | RCPE exclusion filtering for pose tasks |
| `generate_pose_pairs.py` | Exclusion constants (scene58, 5 ICP failures) |
| READMEs | RCPE quickstart, NVS docs, updated test counts |

### Other New Files

| File | What |
|---|---|
| `generate_pose_pairs_v2.py` | Standalone CLI for stratified pair generation |
| `docs/guides/adding-a-model.md` | Community guide: one adapter file + one registry line, any task |

---

## Design Decisions

### No Composite DRS/EDS Score
Three axes reported separately. Backed by: FDA AI/ML (checklists), Model Cards (disaggregated), TRADES theorem, BOP/nuScenes/GraspNet precedent.

### RCPE Design
- Only Clutter (phase 0) + Clean (phase 2) — Interaction excluded (noisy T265 VIO)
- Excluded: scene58, 5 scene/phase ICP failures
- Rotation bins: [5°-15°, 15°-45°, 45°-90°, 90°-180°], <5° excluded
- Scale: ~59,500 pairs (3.6× RUBIK, 40× ScanNet-1500)
- Novel metrics: Metric AUC (joint rot+trans), Cross-phase Δ, Temporal drift

### NVS Design
- Inference-only: no per-scene training, single forward pass
- 10 feed-forward models (DepthSplat, MVSplat, pixelSplat, NoPoSplat, Splatt3R, AnySplat, PF3plat, Flash3D, Splatter Image, FLARE)
- Novel: Perceptual Utility Score (PUS) = M_downstream(rendered) / M_downstream(real)
- Novel: rendered depth vs D435 sensor GT (no NVS benchmark does this)
- Novel: cross-phase NVS (context from Clutter → render Clean)
- Novel: per-object rendering quality via instance masks

### Efficiency Metrics
- Hardware-agnostic: Params (M), FLOPs (G)
- Measured latency: supplementary with GPU footnote
- No novel efficiency metric possible (confirmed by literature survey)

---

## How to Run

### RCPE
```bash
cd benchmark && pip install -e '.[hub,dev]'
PYTHONPATH=. python scripts/run_relative_pose.py \
    --model mast3r --split easy --pairs-source on_the_fly \
    --save-predictions --device cuda
# Output: result.json + summary.md + rcpe_metrics.json
```

### NVS (runner script NOT yet built — pairs + metrics ready)
```python
from rpx_benchmark.nvs_pairs import NVSPairGenerator
from rpx_benchmark.nvs_metrics import evaluate_nvs, perceptual_utility_score
gen = NVSPairGenerator(extracted_root, parquet_path, split="easy")
for sample in gen.iter_samples():
    # sample has context_rgb_paths, context_depth_paths, context_pose_paths,
    # target_pose_path, target_rgb_path (GT), target_depth_path (GT)
    pass
```

---

## Paper Draft (gitignored, on disk for Overleaf)

### Files in `paper-submission/overleaf/text/`

| File | Wired Into | What |
|---|---|---|
| `00_abstract_v2.tex` | `root.tex` | Updated abstract: 3 axes, RCPE, NVS, PUS, no composite |
| `01b_contributions_updated.tex` | `01_intro.tex` | 6 contributions: methodology, dataset, RCPE, NVS+PUS, findings, toolkit |
| `02b_related_rcpe.tex` | `02_related.tex` | RCPE benchmarks + models + deployment readiness lit |
| `03b_deployment_readiness.tex` | `03_method.tex` | Three-axis profile replaces EDS composite |
| `05b_rcpe_task.tex` | `05_tasks.tex` | D6: RCPE — pair types, novel metrics, model slate |
| `05c_task_table_updated.tex` | `05_tasks.tex` | 7-task table |
| `05d_nvs_task.tex` | `05_tasks.tex` | D7: NVS — rendered depth, cross-phase, PUS, model slate |
| `06b_rcpe_experiments.tex` | `06_experiments.tex` | RCPE experiments with \todo{} for results |
| `06c_nvs_experiments.tex` | `06_experiments.tex` | NVS experiments with \todo{} for results |

### Modified Existing Tex Files
- `root.tex` → points to `00_abstract_v2.tex`
- `01_intro.tex` → uses `01b_contributions_updated.tex`, removed EDS refs
- `02_related.tex` → includes `02b_related_rcpe.tex`
- `03_method.tex` → replaced EDS section with `03b_deployment_readiness.tex`
- `05_tasks.tex` → "seven" tasks, includes D6+D7, updated task table
- `06_experiments.tex` → includes RCPE+NVS experiments, cross-task analysis
- `10_conclusion.tex` → removed EDS refs, indoor+outdoor, three-axis language

---

## What's Left

1. **NVS runner script** (`run_nvs.py`) — model adapters for the 10 feed-forward models
2. **Run model slates** on full 100-scene dataset (RCPE: 10 models, NVS: 10 models, all other tasks)
3. **Fill `\todo{}` markers** in paper with actual experimental numbers
4. **PUS evaluation** — run downstream RPX models (depth, detection, seg) on rendered views
5. **Figures** — radar charts, per-bin plots, drift curves, PUS vs PSNR scatter

---

## RCPE Model Slate (10, all registered)
opencv_baseline, loftr, mast3r, dust3r, reloc3r, far, mickey, nope_sac, srpose, icp_open3d

## NVS Model Slate (10, inference-only)
DepthSplat, MVSplat, pixelSplat, NoPoSplat, Splatt3R, AnySplat, PF3plat, Flash3D, Splatter Image, FLARE

---

## Key Novel Contributions (for reviewer defense)

1. **Three-phase protocol** — controlled scene-state variation (Clutter/Interaction/Clean)
2. **Cross-task evaluation on shared scenes** — 7 tasks, same 100 scenes
3. **RCPE: largest structured benchmark** (60K pairs, 3.6× RUBIK) with cross-phase pairs + Metric AUC
4. **NVS: Perceptual Utility Score** — rendered view utility for downstream perception, not just PSNR
5. **NVS: rendered depth evaluation** against real sensor GT — first NVS benchmark to do this
6. **Three-axis deployment readiness** — no arbitrary composite, disaggregated reporting
