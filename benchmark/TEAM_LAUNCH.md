# RPX Team Launch — HF upload + depth benchmark

One-page guide to running the canonical sequence on the full dataset.

## 0. Prerequisites

```bash
pip install -e 'benchmark[hub]'
hf auth login                                    # for HF push
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

**`--model` keys (19 live):**

*Metric (9, `native_alignment="none"`)* — `zoedepth`, `depth_pro`,
`da_v2_metric_indoor`, `da_v2_metric_outdoor`, `unidepth_v2`, `moge_2`,
`patchfusion`, `hyden_metric`, `metric3d_v2`.

*Relative (10, `native_alignment="ls_affine"`)* — `da_v2_relative`,
`da_v1`, `midas_v31`, `distill_any_depth`, `marigold`, `marigold_lcm`,
`lotus_2`, `geowizard`, `moge_v1`, `hyden_relative`.

Full source / precision table in [`README.md`](README.md#for-the-team--canonical-sweep).
**Optional install extras** (per adapter): `pip install diffusers
accelerate` (Marigold, Lotus-2, GeoWizard), `pip install moge` (MoGe
pair), `pip install unidepth` (UniDepth V2). Each adapter raises a
clear `ImportError` with the install hint when missing.

**Dropped from the 20-model target**: MetricSolver — no clean release surfaced as of May 2026.

**Per-run output**: `rpx_results/<display>/<split>/`:
- `result.json`  — three reporting axes (task accuracy / scene-change robustness / compute cost) + per-stage timing + per-metric CIs. **No combined score** — see [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md) for the policy and the deprecation path off the legacy `deployment_readiness` block.
- `summary.md`  — human-readable.
- `comprehensive_metrics.json`  — full metric basket (9 error + 3 accuracy + alignment modes + depth-band stratification + per-object basket + holes + ORD).
- `predictions/<scene>/<phase>/<frame>.npz`  — per-frame raw depth (when `--save-predictions`).

**Box mirror** (when `--upload-to-box`): `<box_folder_id>/monocular_depth/<display>/<split>/...` matches the local tree exactly.

## 3. Relative-pose benchmark sweep — RCPE axis

```bash
# Single model on a single split (no GPU required for the default key)
PYTHONPATH=. python scripts/run_relative_pose.py --model opencv_baseline --split easy

# Team-internal: per-pair predictions CSV + comprehensive metrics + Box mirror
PYTHONPATH=. python scripts/run_relative_pose.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box
```

**`--model` keys (10 live):**

*Direct pose regression (7)* — `reloc3r`, `dust3r`, `mast3r`, `far`,
`srpose`, `nope_sac`, `mickey`.
*Feature matching + solver (1)* — `loftr`.
*Classical (2)* — `opencv_baseline`, `icp_open3d`.

`native_alignment` per adapter: `none` for metric translation, `unit`
for essential-matrix decomposition (translation up-to-scale). Full
table in [`README.md`](README.md#3-run-the-relative-pose-benchmark-rcpe).

**Optional install extras**: `pip install kornia opencv-contrib-python`
(loftr, opencv_baseline), `pip install open3d` (icp_open3d), `pip
install reloc3r` (reloc3r). The dust3r / mast3r / far / srpose /
nope_sac / mickey adapters require a clone + checkpoint of the
upstream repo. Each adapter raises a clear `ImportError` with the hint.

**Per-run output**: `rpx_results/<display>/<split>/`:
- `result.json` — `rotation_error_deg` primary + the three reporting axes (task accuracy / scene-change robustness / compute cost) + timing CIs. No combined score.
- `summary.md` — human-readable.
- `pose_comprehensive_metrics.json` — full pose basket (rotation +
  translation L2 + translation angular + AUC@5°/10°/20° with 95% CIs,
  by-phase, by-stride breakdown).
- `predictions.csv` — single CSV (`scene_id, phase, frame_a, frame_b,
  R00..R22, tx, ty, tz`), header written exactly once, resume-safe.

**Box mirror** (when `--upload-to-box`): `<box_folder_id>/relative_pose/<display>/<split>/`.

**Safety net — re-sync any local run to Box later.** If a sweep ran without `--upload-to-box`, or the 60-min Box token expired mid-sweep, you don't need to re-run anything:

```bash
export BOX_DEVELOPER_TOKEN='<fresh-token>'
PYTHONPATH=. python scripts/sync_results_to_box.py                     # every run under ./rpx_results
PYTHONPATH=. python scripts/sync_results_to_box.py --task relative_pose # filter by task
PYTHONPATH=. python scripts/sync_results_to_box.py --dry-run            # show what would upload, no Box calls
```

Idempotent — files whose Box copy matches in size are skipped automatically. So you can run it after every sweep without worrying about wasted bandwidth.

## 4. Sweep aggregation — once all models have run

> **Note.** The script name and the legacy `drs_*.csv` output naming
> are being retired alongside the DRS scalar (see
> [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md)). The script still runs
> and the per-axis components in each row are still correct — but the
> combined DRS column should not be cited going forward. PR-B will
> rename to `aggregate_results.py` and drop the combined column.

```bash
# Per-split aggregation
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy
PYTHONPATH=. python scripts/run_drs_sweep.py --split medium
PYTHONPATH=. python scripts/run_drs_sweep.py --split hard

# Paper-appendix sensitivity analysis (Kendall's τ across exponent /
# E-function / anchor perturbations) — also deprecated; for legacy
# composite only.
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy --sensitivity
```

Outputs land at `rpx_results/_sweep/drs_<split>.{csv,json}` and `sensitivity_<split>.json`.

## 5. Quality gates

| Check | Status |
|---|---|
| Test suite | 545 passing / 2 expected skips (`pytest tests/`) |
| Data loader | 7 of 8 task recipes round-trip through `RPXDataset.from_manifest` (vqa awaits labels) |
| 19 mono-depth adapters | reproducible from a single HF / pip URL each |
| 10 RCPE pose adapters | registry + base contract covered by import smoke + math tests |
| Per-stage timing + 95% CIs | bootstrap + t-CI on every aggregated metric |
| Per-object metrics | mean across SAM2 instances, ≥200 px, with `coverage` (hole-aware) |
| Per-axis aggregation | OperatingPoint + STR + Tier 1/2/3 efficiency emitted per run; legacy DRS combiner deprecated (see [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md)) |
| Box upload | scene/phase tree mirrored, idempotent (size-matched skip), 2 contract tests |

## 6. Known limits / call-outs

- **`DEPTH_MAX_M = 5.0`** in `rpx_benchmark/metrics/depth_alignment.py`. D435 is unreliable beyond ~6 m; pixels with GT depth >5 m are silently excluded from every metric. **Outdoor scenes (35 of 99) have GT > 5 m**; those models are evaluated only on the <5 m subset of outdoor pixels. Team decision needed: raise the cap, stratify by domain, or document and move on.
- **Box token expires every 60 min** — every relaunch needs a fresh export. The pipeline raises `ConfigError("Box API 401 — token expired", hint="…regenerate at developer console")` cleanly when stale.
- **VQA recipe** is wired but emits zero entries (waits on the team's VQA label-generation pipeline). All 7 other tasks ship.
- **Paired-task pair-stride** is fixed at 5 frames (matches `generate_keypoint_pairs.py`'s convention). Override via `_RelativePoseSpec.pair_stride` if needed.

## 7. Sections by topic

- §1 — HF dataset upload (one-time per dataset version).
- §2 — Monocular-depth sweep (per-model + Box upload).
- §3 — Relative-pose sweep (RCPE — 10 pose adapters).
- §4 — Sweep aggregation (per-axis components per split; legacy DRS combiner being retired — see [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md)).
- §5–6 — Quality gates + known limits.
- Method background: see [`docs/methods/`](docs/methods/).
