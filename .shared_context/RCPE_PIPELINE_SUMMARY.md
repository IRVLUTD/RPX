# RPX-RCPE Pipeline & Paper Draft — Complete Summary

## Decision: No Composite DRS Score

We report three deployment-readiness axes separately:
1. **Task Accuracy** — AUC, Metric AUC, AbsRel, etc. (per task)
2. **Scene-Change Robustness** — STR, Cross-phase Δ, Temporal drift (unique to RPX)
3. **Compute Cost** — Params (M), FLOPs (G), Latency† (supplementary)

No formula combining them. Backed by: FDA AI/ML (checklists not composites), Model Cards (disaggregated reporting), TRADES theorem (accuracy-robustness tradeoff is fundamental), and every successful robotics benchmark (BOP, nuScenes, GraspNet — none use accuracy+efficiency composites).

## Code Deliverables

### New Library Modules (`benchmark/rpx_benchmark/`)
1. **`pose_pairs.py`** — On-the-fly deterministic pair generator (3 pair types, 4 rotation bins, auto-extraction from HF)
2. **`pose_metrics.py`** — Full RCPE metric basket (standard AUC + Metric AUC + cross-phase Δ + temporal drift)
3. **`model_profiler.py`** — Unified profiler for any task (Params, FLOPs, latency)

### Modified Files
- `scripts/run_relative_pose.py` — `--pairs-source on_the_fly`, `--skip-flops`, evaluate_rcpe integration
- `rpx_benchmark/evaluators.py` — Fixed: added `translation_angular_deg` + `pose_error_max_deg`
- `rpx_benchmark/deployment.py` — ERS prefers hardware-agnostic Tier 1/2 over measured Tier 3
- `scripts/generate_pose_pairs.py` + `scripts/local_manifest.py` — RCPE exclusions
- READMEs updated (top-level, benchmark, scripts)
- `docs/guides/adding-a-model.md` — Community guide for adding models to any task

### Verification
- 555/555 tests passing, deterministic, zero duplicates, end-to-end smoke tested

## Paper Draft Updates

### Files Modified (existing tex)
| File | Change |
|---|---|
| `root.tex` | Points to `00_abstract_v2.tex` instead of `00_abstract.tex` |
| `01_intro.tex` | Removed EDS references, updated contributions via `\input{01b_contributions_updated}`, updated findings candidates |
| `02_related.tex` | Added `\input{02b_related_rcpe}` at end |
| `03_method.tex` | Removed EDS composite subsection, replaced with `\input{03b_deployment_readiness}` |
| `05_tasks.tex` | Changed "five" → "six", replaced task table with `\input{05c_task_table_updated}`, added `\input{05b_rcpe_task}` after D5 |
| `06_experiments.tex` | Replaced downstream EDS validation with `\input{06b_rcpe_experiments}` + cross-task analysis |
| `10_conclusion.tex` | Replaced EDS references with three-axis profile language, fixed indoor→indoor+outdoor |

### New Files Created
| File | Purpose |
|---|---|
| `00_abstract_v2.tex` | Updated abstract — multi-axis, RCPE, no composite |
| `01b_contributions_updated.tex` | 5 contributions: methodology, dataset, RCPE, findings, toolkit |
| `02b_related_rcpe.tex` | RCPE benchmarks + models + deployment readiness literature |
| `03b_deployment_readiness.tex` | Three-axis profile: accuracy, robustness, cost (replaces EDS) |
| `05b_rcpe_task.tex` | D6 task: pair types, novel metrics, model slate |
| `05c_task_table_updated.tex` | 6-task table (adds D6: rel. camera pose) |
| `06b_rcpe_experiments.tex` | RCPE experiments: main table, per-bin, cross-phase, drift, metric AUC, cross-task |

### TODO markers
All experimental results are marked `\todo{}` — fill once the model slate runs on the full dataset. 20 TODOs total across experiment files.

## Files to Review

### Code
```
benchmark/rpx_benchmark/pose_pairs.py
benchmark/rpx_benchmark/pose_metrics.py
benchmark/rpx_benchmark/model_profiler.py
benchmark/rpx_benchmark/evaluators.py
benchmark/rpx_benchmark/deployment.py
benchmark/scripts/run_relative_pose.py
benchmark/docs/guides/adding-a-model.md
```

### Paper
```
paper-submission/overleaf/root.tex
paper-submission/overleaf/text/00_abstract_v2.tex
paper-submission/overleaf/text/01_intro.tex (modified)
paper-submission/overleaf/text/01b_contributions_updated.tex
paper-submission/overleaf/text/02_related.tex (modified)
paper-submission/overleaf/text/02b_related_rcpe.tex
paper-submission/overleaf/text/03_method.tex (modified)
paper-submission/overleaf/text/03b_deployment_readiness.tex
paper-submission/overleaf/text/05_tasks.tex (modified)
paper-submission/overleaf/text/05b_rcpe_task.tex
paper-submission/overleaf/text/05c_task_table_updated.tex
paper-submission/overleaf/text/06_experiments.tex (modified)
paper-submission/overleaf/text/06b_rcpe_experiments.tex
paper-submission/overleaf/text/10_conclusion.tex (modified)
```

## Next Steps
1. Run 10-model RCPE slate on full 100-scene dataset
2. Run all other tasks (depth, detection, etc.)
3. Fill `\todo{}` markers with actual numbers
4. Write headline finding based on cross-task, cross-phase analysis
5. Generate figures (radar chart, per-bin plots, drift curves)
