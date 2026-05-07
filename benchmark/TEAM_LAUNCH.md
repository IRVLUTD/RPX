# RPX Team Launch — HF upload + depth benchmark

One-page guide to running the canonical sequence on the full dataset.

## 0. Prerequisites

```bash
pip install -e 'benchmark[hub]'
hf auth login                                    # for HF push (Itay)
export BOX_DEVELOPER_TOKEN='<60-min-token>'      # https://app.box.com/developers/console (refresh hourly)
```

The Box token expires every 60 minutes. Refresh + re-export every time you re-launch a sweep.

---

## 1. HF upload (one-time per dataset version) — run on the system with the captures

```bash
# Pack captures → tar shards
python -m rpx_benchmark.dataset_hub.cli pack \
    --src DATA --staging STAGE --overwrite

# Frames Parquet + per-task per-split manifests + current.json
# (manifest now also writes manifests/<task>/<split>.json automatically — 21 JSONs
#  for the 7 supported tasks × 3 splits. Without --splits the step FAILS LOUDLY.)
python -m rpx_benchmark.dataset_hub.cli manifest \
    --src DATA --staging STAGE \
    --splits benchmark/data/splits/scene_splits.json

# Per-tier scene lists, dataset card, Croissant
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli dataset-card    --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging STAGE --overwrite

# Push to HF (idempotent — re-run any time)
python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE --repo-id IRVLUTD/RPX
```

After upload, any user calling `rpx.load("monocular_depth", "easy")` will:
1. Fetch JSON from `manifests/monocular_depth/easy.json`.
2. Download the tar shards via `snapshot_download`.
3. **Auto-extract** tars into `<snapshot>/extracted/scenes/<scene>/<phase>/<modality>/<frame>.{png|npz}` (new in this release — no manual extraction step needed).
4. Return a working `RPXDataset`.

## 2. Depth benchmark sweep — run per registered model

```bash
# Single model on a single split (community-friendly default — no save, no upload)
PYTHONPATH=. python scripts/run_depth.py --model zoedepth --split easy

# Team-internal: save predictions + comprehensive metrics + Box upload
PYTHONPATH=. python scripts/run_depth.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box
```

**`--model` keys (9 live):**

| key | display | type | precision |
|---|---|---|---|
| `zoedepth` | ZoeDepth_NK | metric | fp32 |
| `depth_pro` | DepthPro | metric (D435 FOV-corrected) | fp32 |
| `da_v2_metric_indoor` | DA-V2-Metric-Indoor-L | metric (Hypersim) | fp16 |
| `da_v2_metric_outdoor` | DA-V2-Metric-Outdoor-L | metric (VKITTI) | fp16 |
| `unidepth_v2` | UniDepth-V2-ViTL14 | metric | fp16 |
| `da_v2_relative` | DA-V2-Relative-L | up-to-scale (`ls_affine`) | fp16 |
| `da_v1` | DA-V1-Large | up-to-scale (`ls_affine`) | fp32 |
| `midas_v31` | MiDaS-v3.1-DPT-BEiT-L | up-to-scale (`ls_affine`) | fp32 |
| `distill_any_depth` | Distill-Any-Depth-L | up-to-scale (`ls_affine`) | fp32 |

**Pending** (Turns C–F): Marigold, Marigold-LCM, Lotus-2, GeoWizard, MoGe-2, MoGe v1, PatchFusion, HyDen-metric, HyDen-relative, Metric3D v2, MetricSolver.

**Per-run output**: `rpx_results/<display>/<split>/`:
- `result.json`  — primary metric, DR report (Tier 1/2/3 + DRS OperatingPoint), per-stage timing, per-metric CIs.
- `summary.md`  — human-readable.
- `comprehensive_metrics.json`  — full Feynman-spec basket (9 error + 3 accuracy + alignment modes + depth-band stratification + per-object basket + holes + ORD).
- `predictions/<scene>/<phase>/<frame>.npz`  — per-frame raw depth (when `--save-predictions`).

**Box mirror** (when `--upload-to-box`): `<box_folder_id>/monocular_depth/<display>/<split>/...` matches the local tree exactly.

## 3. Sweep aggregation — once all models have run

```bash
# DRS table per split
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy
PYTHONPATH=. python scripts/run_drs_sweep.py --split medium
PYTHONPATH=. python scripts/run_drs_sweep.py --split hard

# DRS + paper-appendix sensitivity analysis (Kendall's τ across exponent /
# E-function / anchor perturbations — Feynman's drs_sensitivity)
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy --sensitivity
```

Outputs land at `rpx_results/_sweep/drs_<split>.{csv,json}` and `sensitivity_<split>.json`.

## 4. Quality gates

| Check | Status |
|---|---|
| Test suite | 524 passing / 2 expected skips (`pytest tests/`) |
| Data loader | 7 of 8 task recipes round-trip through `RPXDataset.from_manifest` (vqa awaits labels) |
| 9 mono-depth adapters | reproducible from a single HF / pip URL each |
| Per-stage timing + 95% CIs | bootstrap + t-CI on every aggregated metric |
| Per-object metrics | mean across SAM2 instances, ≥200 px, with `coverage` (hole-aware) |
| DRS pipeline | OperatingPoint per run, sweep aggregator computes DRS + sensitivity |
| Box upload | scene/phase tree mirrored, idempotent (size-matched skip), 2 contract tests |

## 5. Known limits / call-outs

- **`DEPTH_MAX_M = 5.0`** in `rpx_benchmark/metrics/depth_alignment.py`. D435 is unreliable beyond ~6 m; pixels with GT depth >5 m are silently excluded from every metric. **Outdoor scenes (35 of 99) have GT > 5 m**; those models are evaluated only on the <5 m subset of outdoor pixels. Team decision needed: raise the cap, stratify by domain, or document and move on.
- **Box token expires every 60 min** — every relaunch needs a fresh export. The pipeline raises `ConfigError("Box API 401 — token expired", hint="…regenerate at developer console")` cleanly when stale.
- **VQA recipe** is wired but emits zero entries (waits on the team's VQA label-generation pipeline). All 7 other tasks ship.
- **Paired-task pair-stride** is fixed at 5 frames (matches `generate_keypoint_pairs.py`'s convention). Override via `_RelativePoseSpec.pair_stride` if needed.

## 6. Coordination

- **Itay**: §1 (HF upload).
- **Naren / depth lead**: §2 (per-model sweeps), §3 (DRS aggregation).
- **Feynman**: paper section + DRS theory + 20-model survey (`docs/methods/`).
- **Session B (Claude on Jishnu's machine)**: pipeline plumbing, adapter rollout, Box upload.

Cross-session log lives at `benchmark/SHARED_CONTEXT.md`.
