# Benchmark scripts

> CLI runners and helpers that sit alongside the `rpx_benchmark` package.
> Every script is invocable as `PYTHONPATH=. python scripts/<name>.py`,
> or via a `make` shortcut in the parent [`Makefile`](../Makefile).

**What's in here, by use case:**

| Use case | Go to |
|---|---|
| 👀 *I want to see what a scene looks like* | [`visualize_rerun.py`](visualize_rerun.py) (`make visualize`) |
| 🎯 *I want to benchmark a depth model on RPX* | [`run_depth.py`](run_depth.py) — covered in detail in [`../README.md`](../README.md#canonical-sweep) |
| 📐 *I want to benchmark a relative-pose model* | [`run_relative_pose.py`](run_relative_pose.py) |
| 📊 *I want to aggregate my sweep into the paper table* | Read each `rpx_results/<model>/<split>/result.json`'s three top-level axes (`aggregated`, `robustness`, `compute_cost`) directly. *(The legacy single-score `run_drs_sweep.py` aggregator was removed alongside the DRS composite — see `SHARED_CONTEXT.md`.)* |
| ☁️ *I want to mirror my results to UTD Box* | [`sync_results_to_box.py`](sync_results_to_box.py) |
| 📦 *I want to upload a new dataset version to HuggingFace* | [`upload_to_hf.py`](upload_to_hf.py) + [`build_hf_shards.py`](build_hf_shards.py) |
| 🔬 *I want to add a new depth model adapter* | [`depth_models/`](depth_models/) + look at `hf_pipeline.py` for the simplest pattern |

For the full BYO-model workflow (the most common reason an external
user lands in this directory), see
[`../README.md#bring-your-own-model`](../README.md#bring-your-own-model).

## Contents

- [Visualization](#visualization)
- [Benchmarking](#benchmarking)
- [Box helpers (UTD)](#box-helpers-utd)
- [Other scripts](#other-scripts)

---

## Visualization

| Script | What it does | Make target |
|---|---|---|
| [`visualize_rerun.py`](visualize_rerun.py) | Interactive 6- or 18-panel Rerun viewer for any RPX (scene, phase) directly from the HuggingFace cache. Single phase or all-3-phases comparison; cividis-equalized depth; royal jewel-tone masks; T265 trajectory + Pinhole frustum; per-frame status bar. | `make visualize` / `make visualize-phases` / `make visualize-lite` / `make visualize-headless` / `make visualize-save` |
| [`visualize_modalities.py`](visualize_modalities.py) | Static 6-panel matplotlib figure of one frame across all modalities. Useful for paper figures and offline preview. | — |
| [`test_hub_visualize.py`](test_hub_visualize.py) | Smoke test: pulls a single RGB+mask sample from the HF dataset and renders it. Verifies the cache + decode path works end-to-end. | — |
| [`compare_depth_colormaps.py`](compare_depth_colormaps.py) | Side-by-side comparison of 8 colormaps (turbo, plasma, viridis, cividis, magma, inferno, gray, jet) on a single depth frame, all using the same per-frame percentile stretch. Helps pick the right colormap for a given paper or dashboard. | — |

The interactive viewer is the main entry point. See `python scripts/visualize_rerun.py --help` for full options.

### Resource expectations

| Mode | .rrd size | Build time | Peak RSS |
|---|---|---|---|
| Default (250 frames, single phase) | streamed (no .rrd) | ~13 s | ~310 MB |
| `--save` (250 frames, single phase) | ~87 MB | ~18 s | ~315 MB |
| `--lite` (stride 3, jpeg 70) | ~25 MB if `--save` | ~7 s | ~290 MB |
| `--all-phases` (3 × 250 frames) | streamed | ~80 s | ~600 MB |

Numbers from a workstation; SBC-class hardware (Pi 4, Jetson Nano) should add ~5–10× wall time and stay roughly within the same RAM ceiling. The Rerun *viewer* itself benefits from a GPU; if you're on a headless box, build with `--no-spawn --save` and open the `.rrd` on a workstation.

## Benchmarking

| Script | What it does |
|---|---|
| [`run_depth.py`](run_depth.py) | End-to-end runner for the **Monocular Metric Depth** task (Tracker §4). Builds a manifest, calls `BenchmarkRunner`, writes `result.json` + `summary.md` to `./rpx_results/<model>/<split>/`. Two paths: `--use-official` (toolkit `download_split` once HF manifests are published) or local-manifest fallback (default; uses `local_manifest.py`). **True batched inference**: `--batch-size N` (default 4) dispatches the whole batch through the adapter at once via `BatchedDepthBenchmarkModel`, skipping the toolkit's stock `BenchmarkableModel.predict` per-sample loop. The model itself is untouched — only the dispatch path changes — so the metric values are identical batch-vs-no-batch but throughput improves (~1.3× at bs=8 on ZoeDepth, RTX 5070 Laptop). **Predictions are saved scene/phase-wise** when `--save-predictions` is on: `predictions/<scene>/<phase>/<frame>.npz` (mirrors the dataset's on-disk shape, so `--upload-to-box` lands them on Box in the same scene/phase tree). `--comprehensive-metrics` runs the comprehensive metrics post-processor; `--upload-to-box` ships the result dir to UTD Box at `<root>/<task>/<model>/<split>/`. **Resource reporting**: every run emits 3-tier efficiency under `result.json["efficiency"]` — Tier 1 hw-agnostic (params, FLOPs, MACs, memory traffic, arithmetic intensity), Tier 2 roofline bounds (A100 / RTX 4090 / Jetson Orin), Tier 3 measured (wall-clock, peak memory, system card). **Per-stage timing with 95% CIs**: `result.json["timing"]` gives `data_load` / `model_run` / `metric_calc` in ms with `mean` / `median` / `std` / `p5` / `p95` / `ci95_low_boot` / `ci95_high_boot` / `n` per stage. The model_run pass starts with 3 warmup forwards so the latency CI excludes cudnn-autotune startup overhead. **Per-metric CIs**: every depth metric (AbsRel, RMSE, δ1/2/3, etc.) is also reported under `result.json["timing"]["metrics_with_ci"]` (and the comprehensive basket carries the same shape under `comprehensive_metrics.json["aggregated_with_ci"]`). Community default: metrics only, no save, no upload. |
| [`stat_utils.py`](stat_utils.py) | Shared statistical aggregator. `summarize_with_ci(values)` returns `{mean, median, std, p5, p95, ci95_low_t, ci95_high_t, ci95_low_boot, ci95_high_boot, n}` for a 1-D sample; `aggregate_per_sample_with_ci(rows)` does the same per numeric key across a list of per-sample dicts. Bootstrap uses 2000 replicates with the project-wide canonical seed `RPX_SEED = 5_062_026` (MMDDYYYY 05/06/2026, exported from `rpx_benchmark.determinism`); the same seed is used by every other stochastic step (ORD pair sampler, sparse-depth sampler, keypoint-pair sampler, test data generators) so result.json diffs are bit-deterministic across reruns and across modules. t-CI is asymptotic (z=1.96); bootstrap CI is robust to right-skew (matters for latency tails and depth-band errors). Both are reported so reviewers can pick the one they trust. |
| [`run_relative_pose.py`](run_relative_pose.py) | End-to-end runner for the **Relative Camera Pose Estimation** task. Two pair-source modes: `--pairs-source manifest` (pre-saved JSON) or `--pairs-source on_the_fly` (deterministic stratified pairs via `PosePairGenerator` — rotation-binned intra-phase + cross-phase Clutter↔Clean + temporal chains, ~60K pairs at full dataset scale, seeded for reproducibility). Writes `result.json` + `summary.md` + `rcpe_metrics.json` (full novel metric basket: standard AUC@5°/10°/20°, metric AUC@(θ°, d cm), per-rotation-bin breakdown, per-pair-type breakdown, cross-phase Δ, temporal drift, 3-tier efficiency). `--skip-flops` for large models (MASt3R, DUSt3R) that OOM during FLOP counting. `--save-predictions` writes per-pair CSV. `--comprehensive-metrics` runs the extended pose basket post-processor. `--upload-to-box` ships results to UTD Box. |
| [`generate_pose_pairs_v2.py`](generate_pose_pairs_v2.py) | Standalone CLI for the stratified pair sampler. Writes a manifest JSON with intra-phase (4 rotation bins × N/bin/phase), cross-phase (Clutter↔Clean), and temporal chains. Use `--pairs-per-bin`, `--cross-pairs-per-bin`, `--chain-count`, `--chain-stride` to tune. Equivalent to `PosePairGenerator` in the library but writes to disk. |
| [`pose_models/`](pose_models/) | Adapters for the 10-model relative pose zoo: `opencv_baseline`, `loftr`, `mast3r`, `dust3r`, `reloc3r`, `far`, `mickey`, `nope_sac`, `srpose`, `icp_open3d`. Each adapter is a callable accepting an image pair; exposes `torch_module` for profiling. Lookup goes through `MODEL_REGISTRY` in `pose_models/__init__.py`. |
| [`pose_comprehensive_metrics.py`](pose_comprehensive_metrics.py) | Post-processor for the full pose metric basket. Reads per-pair predictions CSV + manifest. Emits rotation error, translation L2, translation angular, pose_error_max, AUC@5°/10°/20°, per-phase breakdown, per-stride breakdown, 95% CIs (bootstrap + t-CI). |
| [`local_manifest.py`](local_manifest.py) | Bridges a known dataset_hub gap: reads the HF cache's `manifest/frames_v1.parquet` and emits `manifests/<task>/<split>.json` consumable by `RPXDataset.from_manifest`. Applies **scene-wise splits** (mode-aggregated, STR-bias-safe), filters to per-task required modalities, extracts the needed `(modality, frame)` pairs from tar shards into a flat `<snap>/extracted/` layout. CLI: `--task --split [--max-samples] [--repo]`. |
| [`comprehensive_depth_metrics.py`](comprehensive_depth_metrics.py) | Post-processor for the full metric basket on monocular depth. Reads per-frame `.npz` predictions + GT depth + GT masks. Emits 9 error metrics (AbsRel, SqRel, RMSE, RMSElog, SIlog, log10, MAE, iRMSE, iMAE) + 3 accuracy thresholds (δ1/2/3) + alignment modes (none / median / ls_affine / ls_disparity) + depth-band stratification (near/mid/far) + in-mask/out-of-mask + boundary F-score + ORD. **NaN-explicit validity**: `_valid()` requires `np.isfinite(gt) & (gt > DEPTH_MIN_M) & (gt < DEPTH_MAX_M) & np.isfinite(pred) & (pred > 0)`, so D435 holes (NaN or 0 mm) are excluded from every metric. **Per-object depth basket** (robotics-relevant): for each SAM2 instance with ≥200 pixels, the post-processor reports `(n_pixels, n_valid, coverage, abs_rel, rmse, δ1/2/3, …)`; aggregated as instance-weighted means under `per_object/*` and `per_object_aggregated_with_ci` (so a small graspable bottle counts the same as a large monitor). **Hole stats** under `holes/{overall,in_mask,out_mask}_fraction` flag frames where the sensor saw <90% of an object — the model isn't penalised for those pixels but the fraction is reported so reviewers can separate "model bad here" from "sensor saw nothing here". **Every aggregated metric carries 95% CIs** under `aggregated_with_ci` (mean ± t-CI ± bootstrap-CI, plus median / std / p5 / p95). Used via `run_depth.py --comprehensive-metrics`, or standalone via `--predictions-dir --manifest`. |
| [`smoke_depth.py`](smoke_depth.py) | Bypass-everything direct cache reader. Loads RGB + GT depth from cached tar shards, runs ZoeDepth, prints per-frame δ1/2/3, RMSE, AbsRel. Validates the model adapter on RPX imagery without the toolkit pipeline; useful when the manifest plumbing is broken. |
| [`depth_models/`](depth_models/) | Adapters for the 20-model monocular-depth zoo (10 metric + 10 relative — see [`docs/methods/mono_depth_models.md`](../docs/methods/mono_depth_models.md)). Each adapter is a callable accepting a single RGB or a list (for batched dispatch); declares `native_alignment` (`"none"` for metric, `"ls_affine"` / `"ls_disparity"` for relative); and exposes `torch_module` so the profiler walker can count parameters. Lookup goes through `MODEL_REGISTRY` in `depth_models/__init__.py` — adding a new model is one line. **Live (9 of 20)**: `zoedepth`, `depth_pro` (with D435 FOV correction), `da_v2_metric_indoor` (Hypersim head), `da_v2_metric_outdoor` (VKITTI head), `unidepth_v2`, `da_v2_relative`, `da_v1`, `midas_v31`, `distill_any_depth`. **Pending**: Marigold v1.1, Marigold-LCM, Lotus-2, GeoWizard, MoGe-2, MoGe v1, PatchFusion, HyDen-metric, HyDen-relative, Metric3D v2, MetricSolver. |
| [`depth_models/hf_pipeline.py`](depth_models/hf_pipeline.py) | Generic `HFDepthEstimationAdapter(model_id, native_alignment)` covering every HF-pipeline-friendly depth model. Used by `da_v2_relative`, `da_v1`, `midas_v31`, `distill_any_depth`, and the DA-V2 Metric heads. Models with bespoke post-processing (Depth Pro's FOV correction, ZoeDepth's metric scaling, UniDepth V2's intrinsics estimation, Marigold's diffusion ensembling) keep their own per-model file. |

### Quickstart — run depth benchmark on the test mirror

```bash
# pick any model from the registry (--alignment auto reads native_alignment off the adapter)
PYTHONPATH=. python scripts/run_depth.py --model {zoedepth|depth_pro|da_v2_metric_indoor|da_v2_metric_outdoor|unidepth_v2|da_v2_relative|da_v1|midas_v31|distill_any_depth} --split easy

# default: aggregated metrics + 3-tier efficiency + per-stage timing (community-friendly)
PYTHONPATH=. python scripts/run_depth.py --model zoedepth --split easy

# team workflow: per-frame predictions + comprehensive metrics + Box upload
export BOX_DEVELOPER_TOKEN='...'   # 60-min token from app.box.com/developers/console
PYTHONPATH=. python scripts/run_depth.py --model zoedepth --split easy \
    --save-predictions --comprehensive-metrics --upload-to-box
```

`result.json` for any run carries:

```jsonc
"efficiency": {
  "tier1_hw_agnostic":  {"params_m": 345.07, "flops_g": 4878.9, "macs_g": 2439.4,
                          "memory_traffic_gb": 4.14, "arithmetic_intensity": 1178.2},
  "tier2_roofline":     {"A100-80GB":     {"latency_ms": 250.2, "bottleneck": "compute"},
                          "RTX 4090":      {"latency_ms":  59.1, "bottleneck": "compute"},
                          "Jetson Orin 64GB": {"latency_ms": 920.5, "bottleneck": "compute"}},
  "tier3_measured":     {"latency_ms_per_sample": 227.4, "peak_cuda_mb": 1791.4,
                          "system_card": {"gpu_name": "...", "pytorch_version": "..."}}
},
"timing": {
  "data_load":   {"mean": 10.3, "median": 10.3, "std": 0.1, "p5": 10.2, "p95": 10.4,
                   "ci95_low_t": 10.27, "ci95_high_t": 10.34,
                   "ci95_low_boot": 10.28, "ci95_high_boot": 10.34, "n": 3000},
  "model_run":   {"mean": 195.9, "median": 192.1, "std": 14.7, "p5": 178.4, "p95": 222.0,
                   "ci95_low_boot": 195.4, "ci95_high_boot": 196.4, "n": 3000},
  "metric_calc": {"mean":  1.4, "median":  1.4, "std": 0.05, "p5": 1.32, "p95": 1.49,
                   "ci95_low_boot": 1.39, "ci95_high_boot": 1.41, "n": 3000},
  "metrics_with_ci": {
    "absrel":   {"mean": 0.085, "ci95_low_boot": 0.083, "ci95_high_boot": 0.087, "n": 3000},
    "delta1":   {"mean": 0.942, "ci95_low_boot": 0.937, "ci95_high_boot": 0.946, "n": 3000},
    "rmse":     {"mean": 0.21,  "ci95_low_boot": 0.20,  "ci95_high_boot": 0.22,  "n": 3000}
    /* ...one entry per metric */
  }
}
```

Numbers above are illustrative — real values land in `rpx_results/<model>/<split>/result.json`. The CI half-width is what you should compare across model adapters: tight CIs (e.g. ±0.5 ms on latency at n=3000) mean the run is statistically converged; wide CIs flag a smoke run that hasn't seen enough frames yet.

### Quickstart — run RCPE benchmark (on-the-fly pairs)

```bash
# On-the-fly stratified pairs — no manifest file needed, fully deterministic from seed
PYTHONPATH=. python scripts/run_relative_pose.py \
    --model mast3r --split easy --pairs-source on_the_fly --device cuda

# With predictions + full metric basket
PYTHONPATH=. python scripts/run_relative_pose.py \
    --model reloc3r --split easy --pairs-source on_the_fly \
    --save-predictions --comprehensive-metrics --device cuda

# For large models that OOM during FLOP counting
PYTHONPATH=. python scripts/run_relative_pose.py \
    --model mast3r --split easy --pairs-source on_the_fly --skip-flops

# Available models:
#   opencv_baseline, loftr, mast3r, dust3r, reloc3r,
#   far, mickey, nope_sac, srpose, icp_open3d
```

`rcpe_metrics.json` for any `--pairs-source on_the_fly` run carries:

```jsonc
{
  "n_pairs": 59530,
  "aggregated": {
    "rotation_error_deg": 12.34,
    "translation_error_m": 0.087,
    "pose_error_max_deg": 18.56
  },
  "standard_auc": {"auc@5": 0.32, "auc@10": 0.58, "auc@20": 0.74},
  "metric_auc": {
    "metric_auc@(5deg,5cm)": 0.12,
    "metric_auc@(10deg,10cm)": 0.41,
    "metric_auc@(20deg,20cm)": 0.68
  },
  "per_bin": {
    "easy":    {"n_pairs": 14800, "rotation_error_deg": 5.2},
    "medium":  {"n_pairs": 14800, "rotation_error_deg": 11.8},
    "hard":    {"n_pairs": 14800, "rotation_error_deg": 15.3},
    "extreme": {"n_pairs": 14800, "rotation_error_deg": 17.1}
  },
  "per_type": {
    "intra_phase":    {"n_pairs": 38600, "rotation_error_deg": 11.2},
    "cross_phase":    {"n_pairs": 11280, "rotation_error_deg": 14.8},
    "temporal_chain": {"n_pairs":  9650, "rotation_error_deg":  6.1}
  },
  "cross_phase_delta": {
    "delta_overall": 3.6,  // cross-phase is 3.6° harder
    "delta_easy": 2.1, "delta_hard": 5.4
  },
  "temporal_drift": {
    "n_chains": 965,
    "mean_drift_rot_deg": 48.2,
    "mean_drift_trans_m": 0.42
  },
  "efficiency": { /* Tier 1 / Tier 2 / Tier 3 — see depth section above */ }
}
```

Numbers above are illustrative — actual values depend on the model.

**Three pair types** (generated deterministically from seed `5_062_026`):
- **Intra-phase**: within one (scene, phase) sequence, stratified into 4 rotation bins (5°–15°, 15°–45°, 45°–90°, 90°–180°). Tests standard geometric difficulty.
- **Cross-phase**: Clutter ↔ Clean of the same scene. Same physical space, objects rearranged. Tests scene-change robustness. *Unique to RPX.*
- **Temporal chains**: ordered consecutive pairs for drift measurement. Tests stability for SLAM integration.

**Novel metrics** (not in any existing RCPE benchmark):
- **Metric AUC@(θ°, d cm)**: joint rotation + metric translation threshold. Needs metric-scale GT.
- **Cross-phase Δ**: performance drop from intra to cross-phase at matched rotation bins.
- **Temporal drift**: accumulated error over chain length.

## Box helpers (UTD)

| Script | What it does |
|---|---|
| [`box_fetch.py`](box_fetch.py) | Pull files from UTD Box without manual downloads. Supports vanity (`/s/<token>`), file, and folder URLs. Anonymous shared links work without auth via Box's public download endpoint; auth-gated content uses `BOX_DEVELOPER_TOKEN` (REST API). Cached at `~/.cache/rpx-box/`. Also has the writer side: `upload_tree(local_dir, remote_path, root_folder_id=...)` ships a directory tree to Box, idempotent (size-matched skip / new-version on size change), used by `run_depth.py --upload-to-box`. |

```bash
# fetch a single file
python scripts/box_fetch.py 'https://utdallas.box.com/s/<token>'

# list a folder (no download)
python scripts/box_fetch.py '<folder-url>' --list

# recurse into subfolders (needs BOX_DEVELOPER_TOKEN; anonymous scope is one-folder-deep)
BOX_DEVELOPER_TOKEN=... python scripts/box_fetch.py '<root-url>' --list --recurse
```

## Other scripts

- `build_hf_shards.py` — pack a capture tree into the per-modality tar shards the dataset hub expects.
- `upload_to_hf.py` — push a packed tree to a HuggingFace dataset repo.
- `generate_keypoint_pairs.py`, `generate_pair_task_manifests.py`, `generate_sparse_depth.py` — various data-prep helpers, see top of each file for details.
