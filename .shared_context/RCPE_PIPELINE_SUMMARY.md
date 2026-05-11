# RPX — RCPE + NVS Pipeline Status

**Commit:** `d7d5120` on `jishnu/beautify-readmes`
**Date:** 2026-05-11

---

## What's Committed and Working

### New Library Modules (`benchmark/rpx_benchmark/`)

| File | Lines | What |
|---|---|---|
| `pose_pairs.py` | 655 | On-the-fly deterministic RCPE pair generator. Three pair types (intra-phase × 4 rotation bins, cross-phase Clutter↔Clean, temporal chains). ~60K pairs at full scale. Auto-extracts from HF tars. Seeded, zero duplicates, deterministic. |
| `pose_metrics.py` | 314 | `evaluate_rcpe(per_pair)` — full metric basket. Standard: AUC@5°/10°/20°. Novel: Metric AUC@(θ°, d cm), cross-phase Δ, temporal drift, per-bin/per-type breakdowns. |
| `model_profiler.py` | 346 | `ModelProfiler(model)` — unified profiler for any task. Auto-discovers torch module. `pre_run_profile()` → `full_report(eff)`. |
| `nvs_pairs.py` | 369 | On-the-fly NVS evaluation sample generator. Context views (N=2,4,8,16) + target query + GT. Interpolation/extrapolation/cross-phase. ~25K samples at full scale. |

### Modified Files

| File | Change |
|---|---|
| `evaluators.py` | `pose_metrics()` now returns 4 keys: added `translation_angular_deg` + `pose_error_max_deg` (fixes standard AUC) |
| `run_relative_pose.py` | Added `--pairs-source on_the_fly`, `--skip-flops`, `_run_on_the_fly()` function, writes `rcpe_metrics.json` with tiered efficiency |
| `local_manifest.py` | RCPE exclusion filtering for pose tasks |
| `generate_pose_pairs.py` | Exclusion constants (scene58, 5 ICP failures) |
| `scripts/README.md` | RCPE quickstart + output format docs |
| `benchmark/README.md` | On-the-fly stratified pair docs, exclusions |

### New Docs/Scripts

| File | What |
|---|---|
| `docs/guides/adding-a-model.md` | Community guide: one adapter file + one registry line, any task |
| `generate_pose_pairs_v2.py` | Standalone CLI for stratified pair generation |

### Verification

- 518/518 tests passing (DRS/ERS tests removed by prior PR-B)
- End-to-end: `--pairs-source on_the_fly` → pair gen → runner → standard + novel metrics → JSON ✅
- Determinism ✅, zero duplicates ✅, auto-extraction ✅

---

## Key Design Decisions

### No Composite DRS/EDS Score
Three axes reported separately:
1. **Task Accuracy** — AUC, Metric AUC, AbsRel, etc.
2. **Scene-Change Robustness** — STR, Cross-phase Δ, Temporal drift
3. **Compute Cost** — Params (M), FLOPs (G), Latency† (supplementary)

Backed by: FDA AI/ML (checklists), Model Cards (disaggregated), TRADES theorem (tradeoff fundamental), BOP/nuScenes/GraspNet precedent.

### RCPE Pair Design
- Only Clutter (phase 0) + Clean (phase 2) — Interaction excluded (noisy T265 VIO)
- Excluded: scene58 entirely, 5 scene/phase ICP failures
- Rotation bins: [5°-15°, 15°-45°, 45°-90°, 90°-180°], <5° excluded as degenerate
- Cross-phase: Clutter↔Clean same scene — unique to RPX
- Scale: ~59,500 pairs (3.6× RUBIK, 40× ScanNet-1500)

### Efficiency Metrics
- Hardware-agnostic: Params (M), FLOPs (G) only
- Measured latency: supplementary with GPU footnote
- No novel efficiency metric (literature survey confirmed none exists)

---

## How to Run

```bash
cd benchmark
pip install -e '.[hub,dev]'

# RCPE — on-the-fly stratified pairs
PYTHONPATH=. python scripts/run_relative_pose.py \
    --model mast3r --split easy --pairs-source on_the_fly \
    --save-predictions --device cuda

# Large models
    --skip-flops

# Available pose models:
# opencv_baseline, loftr, mast3r, dust3r, reloc3r,
# far, mickey, nope_sac, srpose, icp_open3d
```

Output: `result.json` + `summary.md` + `rcpe_metrics.json`

---

## What's Left

1. **NVS runner script** — `nvs_pairs.py` generates samples but no `run_nvs.py` or model adapters (DepthSplat, MVSplat, etc.) exist yet
2. **Run model slates** on full 100-scene dataset (RCPE: 10 models, NVS: 10 models)
3. **Paper LaTeX** — draft tex files were created but not committed (in `paper-submission/overleaf/text/`). Need to integrate into Overleaf.
4. **Fill experimental results** — all `\todo{}` markers in paper

### NVS Model Slate (10 inference-only feed-forward models)
| # | Model | Venue | Needs Poses? | Uses Depth? | Checkpoint |
|---|---|---|---|---|---|
| 1 | DepthSplat | CVPR 2025 | Yes | Yes (in+out) | HuggingFace |
| 2 | MVSplat | ECCV 2024 | Yes | Predicts | Google Drive |
| 3 | pixelSplat | CVPR 2024 | Yes | Predicts | Google Drive |
| 4 | NoPoSplat | ICLR 2025 | No | Predicts | HuggingFace |
| 5 | Splatt3R | — | No | MASt3R | HuggingFace |
| 6 | AnySplat | SIGGRAPH Asia 2025 | No | Predicts | HuggingFace |
| 7 | PF3plat | ICML 2025 | No | Predicts | Available |
| 8 | Flash3D | ECCV 2024 | No (1 image) | Predicts | Available |
| 9 | Splatter Image | CVPR 2024 | No (1 image) | Predicts | Available |
| 10 | FLARE | 2025 | No | Predicts | Available |

### NVS Metrics
- **Axis 1**: PSNR, SSIM, LPIPS (standard) + depth AbsRel/RMSE/δ<1.25 of rendered views vs D435 GT (novel)
- **Axis 2**: Cross-phase rendering degradation (train Clutter → render Clean)
- **Axis 3**: Params, FLOPs, inference time

### NVS Research Briefs (on disk, not committed)
- `research-nvs-benchmarks.md` (22 sources)
- `research_nvs_models.md` (30 sources)
- `research-nvs-robot-learning.md` (15 sources)
- `outputs/nvs-survey-consolidated.md`

---

## Files to Review

```
# Core (committed)
benchmark/rpx_benchmark/pose_pairs.py
benchmark/rpx_benchmark/pose_metrics.py
benchmark/rpx_benchmark/model_profiler.py
benchmark/rpx_benchmark/nvs_pairs.py
benchmark/rpx_benchmark/evaluators.py
benchmark/scripts/run_relative_pose.py
benchmark/scripts/generate_pose_pairs.py
benchmark/scripts/local_manifest.py
benchmark/docs/guides/adding-a-model.md

# Paper drafts (NOT committed — in overleaf text/)
paper-submission/overleaf/text/00_abstract_v2.tex
paper-submission/overleaf/text/01b_contributions_updated.tex
paper-submission/overleaf/text/02b_related_rcpe.tex
paper-submission/overleaf/text/03b_deployment_readiness.tex
paper-submission/overleaf/text/05b_rcpe_task.tex
paper-submission/overleaf/text/05c_task_table_updated.tex
paper-submission/overleaf/text/06b_rcpe_experiments.tex
```
