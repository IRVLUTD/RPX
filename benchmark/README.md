# RPX Benchmark Toolkit

**Benchmark robot-perception models on real-world RGB-D data — across clutter, human-interaction, and clean phases of the same 100 indoor + outdoor scenes (RCPE / NVS evaluate on clutter + clean only — interaction omitted).**

`rpx-benchmark` is the reference toolkit for [RPX — Robot Perception
X](https://github.com/IRVLUTD/RPX). The toolkit core ships the **dataset
loader, task-specific metrics, hardware-agnostic profiling, and a
deployment-readiness scoring framework**. The team's reference
implementations of nine monocular-depth adapters and the per-task run
scripts live alongside it under `scripts/` for the paper sweep.

External users with their own models: [skip to the BYO-model
quickstart](#bring-your-own-model). Team members running the canonical
sweep: [start here](#for-the-team-canonical-sweep).

---

## For the team — canonical sweep

This is the day-the-data-lands launch sequence. **Single-page version:
[`TEAM_LAUNCH.md`](TEAM_LAUNCH.md)**.

### 0. One-time setup

```bash
pip install -e 'benchmark[hub,viz]'
hf auth login                                    # for HF dataset push
```

#### Box auth (only if you'll pass `--upload-to-box`)

Two modes, **pick one**. OAuth is required for multi-hour sweeps — Box
dev tokens hard-expire at 60 minutes.

| | OAuth 2.0 (recommended) | Developer token (quick / one-off) |
|---|---|---|
| Access token lifetime | 60 min | 60 min |
| Auto-renews? | **Yes** — refresh-token rotation, ~60-day rolling window | No — manual refresh every hour |
| Setup | one-time browser login | copy + paste from console |
| Survives a long sweep? | yes | no — uploads start failing at minute 60 |

**OAuth (recommended).** One-time browser auth captures a refresh token
that's stored at `~/.config/rpx_benchmark/box_tokens.json` (mode 600) and
auto-rotated on every upload:

```bash
# From your Box app at https://app.box.com/developers/console
export BOX_CLIENT_ID=...
export BOX_CLIENT_SECRET=...
export BOX_REDIRECT_URI=http://localhost:8765/callback  # add this exact URI in the app config

python -m rpx_benchmark.box_upload login    # one browser tab; close when it says "complete"
python -m rpx_benchmark.box_upload whoami   # sanity probe
```

**Developer token (legacy / smoke tests only).** Refresh hourly from the
[Box developer console](https://app.box.com/developers/console):

```bash
export BOX_DEVELOPER_TOKEN='<60-min-token>'
```

If both are configured, OAuth wins — the dev token is only consulted
when no OAuth tokens exist on disk. To forget OAuth state:
`python -m rpx_benchmark.box_upload logout`.

### 1. Push the dataset to HuggingFace

```bash
# Run once per dataset version, on the system that holds the ~890 GB captures.
python -m rpx_benchmark.dataset_hub.cli pack            --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli manifest        --src DATA --staging STAGE \
                                                          --splits benchmark/data/splits/scene_splits.json
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli dataset-card    --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli upload          --staging STAGE --repo-id IRVLUTD/RPX
```

Idempotent — re-run any time. Without `--splits` the `manifest` step
**fails loudly** rather than silently producing an unusable HF tree.
After upload, any consumer calling `rpx.load("monocular_depth", "easy")`
gets: JSON fetched → tar shards downloaded → tars **auto-extracted**
into `<snapshot>/extracted/scenes/...` (no manual extraction step).

Full details: [`rpx_benchmark/dataset_hub/README.md`](rpx_benchmark/dataset_hub/README.md).

### 2. Run the depth benchmark

```bash
# Per model, per split. Add --upload-to-box to mirror the result dir to UTD Box.
PYTHONPATH=. python scripts/run_depth.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box
```

19 model keys registered. All reproducible from a single HF / pip / torch.hub URL each.

**Metric (9)** — `native_alignment="none"`:

| `--model <key>`           | Source                                                          | Precision |
|---------------------------|-----------------------------------------------------------------|-----------|
| `zoedepth`                | `Intel/zoedepth-nyu-kitti`                                       | fp32 |
| `depth_pro`               | `apple/DepthPro-hf` (D435 FOV-corrected)                         | fp32 |
| `da_v2_metric_indoor`     | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`        | fp16 |
| `da_v2_metric_outdoor`    | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`       | fp16 |
| `unidepth_v2`             | `lpiccinelli/unidepth-v2-vitl14` (PyPI `unidepth`)               | fp16 |
| `moge_2`                  | `Ruicheng/moge-2-vitl-normal` (PyPI `moge`)                      | fp16 |
| `patchfusion`             | `zhyever/patchfusion_zoedepth`                                   | fp32 |
| `hyden_metric`            | `facebook/hyden-da2-metric-depth`                                | fp16 |
| `metric3d_v2`             | torch.hub `YvanYin/Metric3D` / `metric3d_vit_giant2`             | fp32 |

**Relative / up-to-scale (10)** — `native_alignment="ls_affine"`:

| `--model <key>`           | Source                                                  | Precision |
|---------------------------|---------------------------------------------------------|-----------|
| `da_v2_relative`          | `depth-anything/Depth-Anything-V2-Large-hf`             | fp16 |
| `da_v1`                   | `LiheYoung/depth-anything-large-hf`                     | fp32 |
| `midas_v31`               | `Intel/dpt-beit-large-384`                              | fp32 |
| `distill_any_depth`       | `xingyang1/Distill-Any-Depth-Large-hf`                  | fp32 |
| `marigold`                | `prs-eth/marigold-depth-v1-1` (PyPI `diffusers`)         | fp16 |
| `marigold_lcm`            | `prs-eth/marigold-depth-lcm-v1-0`                        | fp16 |
| `lotus_2`                 | `jingheya/Lotus-2`                                       | fp16 |
| `geowizard`               | `lemonaddie/geowizard`                                   | fp16 |
| `moge_v1`                 | `Ruicheng/moge-vitl`                                     | fp16 |
| `hyden_relative`          | `facebook/hyden-da2-relative-depth`                      | fp16 |

**Dropped from the original target list**: MetricSolver — no clean
release surfaced as of May 2026. Will revisit when a maintained
checkpoint lands.

**Optional install extras** (each adapter raises a clear ImportError on
construct when missing):
```bash
pip install diffusers accelerate            # marigold, marigold_lcm, lotus_2, geowizard
pip install moge                            # moge_2, moge_v1
pip install unidepth                        # unidepth_v2
# torch.hub for metric3d_v2 — first call clones YvanYin/Metric3D + downloads weights
```

Per-run output lands under `rpx_results/<DisplayName>/<split>/`:
- `result.json` — three independent axes (`aggregated` task perf · `robustness` · `compute_cost` with Tier 1/2/3 profiler + `operating_point`) and per-stage timing with 95% CIs. No composite score.
- `summary.md` — human-readable.
- `comprehensive_metrics.json` — full metric basket (9 errors + 3 accuracy + alignment modes + depth-band stratification + per-object basket + hole stats + ORD).
- `predictions/<scene>/<phase>/<frame>.npz` — per-frame raw depth (when `--save-predictions`).

Box mirror (when `--upload-to-box`):
`<box_root>/monocular_depth/<DisplayName>/<split>/...` matches the local
tree exactly. Idempotent (size-matched skip on re-upload).

#### Run all 19 across all 3 splits (one shell loop)

```bash
# Sweep loop. Drop --upload-to-box if you haven't run `box_upload login` (or have no fresh dev token).
for split in easy medium hard; do
  for model in $(PYTHONPATH=.:scripts python3 -c \
       "from depth_models import list_models; print(' '.join(list_models()))"); do
    PYTHONPATH=. python scripts/run_depth.py \
        --model "$model" --split "$split" \
        --batch-size 4 \
        --save-predictions --comprehensive-metrics --upload-to-box \
      || echo "[skip] $model/$split — see traceback above"
  done
done
```

The `|| echo "[skip] ..."` keeps the sweep going when one adapter
fails (e.g. an optional package is missing or a checkpoint id has
shifted). Remaining models still run; the failing one shows up in the
log and gets fixed in a follow-up.

#### Output schema (`result.json` essentials)

```jsonc
{
  "task": "monocular_depth",
  "model": "ZoeDepth_NK",
  "split": "easy",
  "num_samples": 3000,
  "aggregated":  { "absrel": 0.0848, "delta1": 0.951, "rmse": 0.205, ... },
  "robustness": {
    "weighted_phase_score": { "score": 0.86, ... },
    "temporal_stability":   { "ts_score": 0.91, ... },
    "state_transition":     { "str_score": -0.085, ... },
    "geometric_coherence":  { "sgc_score": 0.78, ... }
  },
  "compute_cost": {
    "params_m": 345.07,
    "flops_g": 4878.9,
    "macs_g": 2439.4,
    "actmem_gb_fp16": 0.62,
    "memory_traffic_gb": 4.14,
    "arithmetic_intensity": 1178.2,
    "roofline": {
      "A100-80GB":         { "latency_ms": 250.2, "bottleneck": "compute" },
      "RTX 4090":          { "latency_ms":  59.1, "bottleneck": "compute" },
      "Jetson Orin 64GB":  { "latency_ms": 920.5, "bottleneck": "compute" }
    },
    "latency_ms_per_sample": 158.5,
    "peak_memory_mb": 1820.3,
    "system_card": { "gpu_name": "...", "pytorch_version": "...", "cuda_version": "..." },
    "operating_point": {
      "precision": "fp32",
      "task_metric": 0.0848, "task_metric_name": "absrel", "higher_is_better": false,
      "flops_g": 4878.9, "params_m": 345.07
    }
  },
  "timing": {
    "data_load":   { "mean": 10.4, "ci95_low_boot": 10.3, "ci95_high_boot": 10.5, "n": 3000 },
    "model_run":   { "mean": 158.5, "ci95_low_boot": 157.8, "ci95_high_boot": 159.2, "n": 3000 },
    "metric_calc": { "mean":  1.4, "ci95_low_boot":  1.3, "ci95_high_boot":  1.5, "n": 3000 },
    "metrics_with_ci": { "absrel": { "mean": 0.085, "ci95_low_boot": 0.083, ... }, ... }
  }
}
```

`comprehensive_metrics.json` adds 9 errors × {none, median, ls_affine,
ls_disparity} alignments × {near, mid, far} depth bands × {in_mask,
out_mask}, plus `per_object_aggregated_with_ci` (instance-weighted),
`holes/{overall,in_mask,out_mask}_fraction`, and ORD pair accuracy —
each with the same 95%-CI shape as `timing.metrics_with_ci`.


### 3. Run the relative-pose benchmark (RCPE)

```bash
PYTHONPATH=. python scripts/run_relative_pose.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box
```

10 model keys registered. Each adapter raises a clear `ImportError` on
construct when its optional package isn't installed, so the registry
loads cleanly even on a bare environment.

| `--model <key>`    | Source                                                      | `native_alignment` | Precision |
|---------------------|------------------------------------------------------------|--------------------|-----------|
| `reloc3r`           | `siyan824/reloc3r-512` (PyPI `reloc3r`)                     | `none`             | fp16      |
| `dust3r`            | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`                 | `none`             | fp16      |
| `mast3r`            | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`    | `none`             | fp16      |
| `far`               | github `crockwell/far` (CVPR 2024)                          | `none`             | fp32      |
| `srpose`            | github `frank-mengfei/SRPose` (ECCV 2024)                   | `unit`             | fp32      |
| `nope_sac`          | github `IRMVLab/NOPE-SAC` (TPAMI 2023)                      | `unit`             | fp32      |
| `mickey`            | github `nianticlabs/mickey` (CVPR 2024 Oral)                | `none`             | fp16      |
| `loftr`             | `kornia.feature.LoFTR` + OpenCV `findEssentialMat`           | `unit`             | fp16      |
| `opencv_baseline`   | SIFT + ratio-test + USAC-MAGSAC `recoverPose`                | `unit`             | fp32      |
| `icp_open3d`        | Colored ICP (RGBD) — Open3D                                  | `none`             | fp64      |

`native_alignment="unit"` declares that the adapter recovers
translation only up-to-scale (essential-matrix decomposition); the
metrics post-processor uses `translation_angular_deg` for those models
and `translation_l2` for the metric ones.

**Optional install extras:**
```bash
pip install kornia opencv-contrib-python   # loftr, opencv_baseline
pip install open3d                          # icp_open3d
pip install reloc3r                         # reloc3r
# dust3r / mast3r / far / srpose / nope_sac / mickey: clone upstream repo + checkpoint
```

Per-run output lands under `rpx_results/<DisplayName>/<split>/`:
- `result.json` — three axes (`aggregated.rotation_error_deg` · `robustness` · `compute_cost`) and timing CIs. No composite score.
- `summary.md` — human-readable.
- `pose_comprehensive_metrics.json` — full pose basket (rotation +
  translation L2 + translation angular + pose_error_max_deg + AUC@5°/10°/20°,
  with 95% CIs, by-phase, by-stride breakdown).
- `predictions.csv` — single CSV when `--save-predictions`. Columns:
  `scene_id, phase, frame_a, frame_b, R00..R22, tx, ty, tz` (16 cols,
  resume-safe — header written exactly once).

Box mirror (when `--upload-to-box`):
`<box_root>/relative_pose/<DisplayName>/<split>/...` matches the local
tree exactly. Idempotent (size-matched skip on re-upload).

#### Run all 10 across all 3 splits

```bash
for split in easy medium hard; do
  for model in $(PYTHONPATH=.:scripts python3 -c \
       "from pose_models import list_models; print(' '.join(list_models()))"); do
    PYTHONPATH=. python scripts/run_relative_pose.py \
        --model "$model" --split "$split" \
        --save-predictions --comprehensive-metrics --upload-to-box \
      || echo "[skip] $model/$split — see traceback above"
  done
done
```

#### Pair sampling: on-the-fly stratified generation (recommended)

The **recommended** pair source is `--pairs-source on_the_fly`, which
uses `rpx_benchmark.pose_pairs.PosePairGenerator` to generate ~60K
deterministic pairs directly from the HF-cached poses — no manifest
file needed, no disk writes, fully reproducible from seed.

```bash
# Recommended: on-the-fly stratified pairs
PYTHONPATH=. python scripts/run_relative_pose.py --model <KEY> --split easy \
    --pairs-source on_the_fly --save-predictions --comprehensive-metrics
```

**Three pair types** (seed `5_062_026`):

| Type | What | Count (full dataset) |
|------|------|---------------------|
| **Intra-phase** | Within one (scene, phase), 4 rotation bins (5°–15°, 15°–45°, 45°–90°, 90°–180°) | ~38,600 |
| **Cross-phase** | Clutter ↔ Clean of same scene (objects rearranged) | ~11,280 |
| **Temporal chains** | Ordered consecutive pairs for drift measurement | ~9,650 |
| **Total** | | **~59,500** |

**Novel metrics** in `rcpe_metrics.json`:
- Metric AUC@(θ°, d cm) — joint rotation + metric translation threshold
- Cross-phase Δ — performance drop intra → cross at matched bins
- Temporal drift — accumulated error over chains
- Per-rotation-bin + per-pair-type breakdowns

**Exclusions** (ICP-optimized poses unavailable):
- `scene58` entirely (missing from object-corr-out-GT-masks)
- Phase 0: scene83.jsom.garden.pot, scene98.jsom.atrium, scene100.jsom.atrium
- Phase 2: scene50.ecss.out.stairs, scene72.ecsw.atriumStairs
- Interaction phase (1) excluded globally — T265 VIO too noisy during manipulation

Auto-extraction: if cam_pose / RGB tar shards aren’t extracted yet,
the generator downloads them from HuggingFace and extracts
automatically on first use.

**Legacy**: `--pairs-source manifest` still works with pre-saved JSON
manifests (stride-5, Poisson-disk, or any custom format).


### 4. Run the NVS benchmark

```bash
PYTHONPATH=. python scripts/run_nvs.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --upload-to-box
```

11 model keys registered (1 baseline + 10 feed-forward 3DGS slots):

| `--model <key>` | Status | Source |
|---|---|---|
| `identity_passthrough`              | ✅ live  | zero-deps baseline; returns closest-pose context view |
| `splatter_image`                    | scaffolded | `szymanowiczs/splatter-image-multi-category-v1` (clone + `pip install -e .`) |
| `depthsplat` · `mvsplat` · `pixelsplat` · `nopo_splat` · `splatt3r` · `anysplat` · `pf3plat` · `flash3d` · `flare` | pending | each raises `AdapterError` with the upstream URL + install hint when invoked |

Pair generation is **on-the-fly** via `rpx_benchmark.nvs_pairs.NVSPairGenerator`
— deterministic stratification across context counts `K ∈ {2, 4, 8, 16}`,
sample types (interpolation / extrapolation / cross-phase). Like RCPE,
NVS evaluates on **2 phases** (clutter + clean only; interaction omitted).

**Per-run output**: `rpx_results/<DisplayName>/<split>/`:
- `result.json` — three-axis report (`aggregated` task-perf · `robustness` per-context-count / per-sample-type / cross-phase Δ / by-difficulty · `compute_cost` Tier 1/2/3 + OperatingPoint).
- `summary.md` — human-readable.
- `predictions/<scene>/<phase>/<frame>.npz` — per-sample rendered RGB + depth (when `--save-predictions`).

**Novel NVS axes** computed in `evaluate_single_sample`:
- **Per-object PSNR** via SAM2 instance masks — `per_object_psnr_mean` / `per_object_psnr_min` / `n_objects_evaluated`. Surfaces rendering quality at the *object* level, not just frame-averaged.
- **Rendered depth vs sensor GT** — when the adapter returns a depth map, scored against the D435 depth. No other public NVS benchmark does this.
- **Cross-phase Δ** — `rendered(clutter) → clean` performance shift on the same scene.
- **LPIPS (AlexNet)** — opt-in via `--compute-lpips` (+50–100 ms/sample CPU); falls back to NaN silently if `pip install lpips` is absent.

**Robustness extras**: graceful `--strict-io` toggle (default = skip
missing frames and log to `result.json["skipped_samples"]`, warn at
>5% skip rate). LRU-cached modality loaders (~2.8× I/O speedup at
~85% hit rate on a typical sweep slice).

Box mirror (when `--upload-to-box`):
`<box_root>/novel_view_synthesis/<DisplayName>/<split>/...`.

### 5. Aggregate the sweep — three axes, no composite

Per the RPX policy (`SHARED_CONTEXT.md` / `docs/methods/comprehensive_metrics.md`),
**there is no composite-score sweep aggregator**. RPX reports three
independent axes (task performance, scene-change robustness, compute
cost); the paper's headline finding is that *rankings disagree across
axes*. Build the sweep table by reading per-axis components from each
`rpx_results/<model>/<split>/result.json` directly:

```python
import json
from pathlib import Path

rows = []
for path in Path("rpx_results").glob("*/easy/result.json"):
    r = json.loads(path.read_text())
    rows.append({
        "model":             r["model"],
        # Axis 1 — Task Performance
        "task_metric":       r["aggregated"].get("absrel") or r["aggregated"].get("psnr"),
        # Axis 2 — Scene-change robustness
        "str_c_to_i":        r["robustness"]["state_transition"]["str_c_to_i"],
        # Axis 3 — Compute cost
        "params_m":          r["compute_cost"]["params_m"],
        "flops_g":           r["compute_cost"]["flops_g"],
        "latency_ms":        r["compute_cost"]["latency_ms_per_sample"],
    })
# Sort by *whichever axis* your paper section is making a claim about.
```

Same shape works for `run_relative_pose.py` and `run_nvs.py` outputs.

### 6. Where to look

| Channel | When to consult |
|---|---|
| [`TEAM_LAUNCH.md`](TEAM_LAUNCH.md) | running the sweep — one-page handoff |
| [`scripts/README.md`](scripts/README.md) | extending the depth zoo or the metric basket |
| [`rpx_benchmark/dataset_hub/README.md`](rpx_benchmark/dataset_hub/README.md) | HF upload questions |
| `paper-submission/BRIEF_FOR_ADVISOR.tex` | scope discussions / advisor brief |

### Known limits

- **`DEPTH_MAX_M = 5.0`** in `rpx_benchmark/metrics/depth_alignment.py`. D435 is unreliable beyond ~6 m; pixels with GT > 5 m are silently excluded. RPX has 35 outdoor scenes with GT > 5 m → those models are evaluated only on their <5 m subset. Decision needed before paper sweep: raise the cap, stratify by domain, or document and move on.
- **Box dev tokens expire every 60 min** — every relaunch needs a fresh export. The pipeline raises `ConfigError("Box API 401 — token expired", hint="…regenerate at developer console")` cleanly when stale.
- **VQA recipe** is wired but emits zero entries (waits on the team's VQA label-generation pipeline). All 7 other tasks ship.

---

## Bring Your Own Model

External users with their own perception model want this section.

```bash
pip install 'rpx-benchmark[hub]'
```

```python
import numpy as np
import rpx_benchmark as rpx


def my_depth(rgb: np.ndarray) -> np.ndarray:
    """rgb: H x W x 3 uint8  ->  H x W float32 (metres)."""
    ...


model = rpx.make_numpy_depth_model(my_depth, name="my_depth")
cfg = rpx.MonocularDepthRunConfig(model=model, split="hard", device="cpu")
result, report, paths = rpx.run_monocular_depth(cfg)

print(result.aggregated)              # absrel, rmse, delta1..3
print(report.weighted_phase_score)    # ESD-weighted phase score
print(paths["json"])                  # result.json
```

Same shape for every task: `make_numpy_<task>_model(fn)` →
`<TaskName>RunConfig(model=...)` → `run_<task>(cfg)`.

### Three integration paths

1. **Plain numpy callable.** Ten factories, one per task. Shown above.
2. **Custom `InputAdapter` / `OutputAdapter`.** When you need full
   control over preprocessing, batching, or post-processing.
3. **API / cloud model.** Wrap a remote inference service as an
   adapter and mark it `EfficiencyMetadata(model_type="api")`.

See [`docs/getting-started/bring-your-own-model.md`](docs/getting-started/bring-your-own-model.md)
for complete templates of all three.

### True-batched dispatch (optional)

The stock `BenchmarkableModel.predict` iterates samples one at a time,
which is fine for CPU and one-shot debugging. For GPU pipelines that
batch internally (HuggingFace pipelines, DataLoader-driven inference),
wrap your callable as a `BatchedDepthBenchmarkModel` (or
`BatchedSegmentationBenchmarkModel` / `BatchedRelativePoseBenchmarkModel`)
to dispatch the entire batch through your adapter in one call. Model
weights and forward semantics are unchanged — only the dispatch shape.

```python
from rpx_benchmark import BatchedDepthBenchmarkModel
model = BatchedDepthBenchmarkModel(my_depth_callable, name="my_model",
                                    save_dir="rpx_results/my_model/easy/predictions")
```

The `scripts/depth_models/` directory shows nine reference adapters
following this pattern (HF pipelines, custom post-processing for
focal-length-dependent metric depth, fp16 declarations, etc.).

## What the toolkit ships

| Area | Module | What's in it |
|---|---|---|
| Data contracts | `rpx_benchmark.api` | `TaskType`, `Sample`, 10 × GroundTruth, 10 × Prediction, `BenchmarkModel` ABC |
| Dataset | `rpx_benchmark.loader`, `rpx_benchmark.hub`, `rpx_benchmark.data` | `RPXDataset`, modality-aware snapshot downloads (now with auto-extraction), `load_hf()` one-liner |
| Manifest validation | `rpx_benchmark.schemas` | Pydantic v2 schemas + JSON Schema export (opt-in extra) |
| Adapter framework | `rpx_benchmark.adapters` | `BenchmarkableModel`, **batched-dispatch base class** for true GPU batching, 9 numpy fast-path factories |
| Metrics | `rpx_benchmark.metrics.*` | Per-task `MetricCalculator` registry; `depth_alignment` shared between runner + comprehensive post-processor |
| Tasks | `rpx_benchmark.tasks.*` | One module per task, each ~30 lines; self-register `TaskSpec` |
| Runner | `rpx_benchmark.runner` | Orchestrates dataset → model → metrics → three-axis report; persists `OperatingPoint` (precision · task metric · FLOPs · params) |
| Profiler | `rpx_benchmark.profiler` | Tier 1 (params/FLOPs/MACs/memory traffic/AI), Tier 2 (roofline for A100/4090/Orin), Tier 3 (measured + system card) |
| Robustness | `rpx_benchmark.deployment` | Per-axis components only — ESD-weighted Phase Score, STR, Temporal Stability, SGC. No composite score (three axes reported independently) |
| Determinism | `rpx_benchmark.determinism` | `seed_all()`, `deterministic()` context manager, project-wide `RPX_SEED = 5_062_026` |
| Reports | `rpx_benchmark.reports` | JSON + Markdown writers |

## Tasks (10 contracts; loader-side support varies)

| Task | Primary metric | Numpy factory | Loader status |
|---|---|---|---|
| Monocular depth | AbsRel ↓ | `make_numpy_depth_model` | ✓ live |
| Object segmentation | mIoU ↑ | `make_numpy_mask_model` | ✓ live |
| Object detection | mAP ↑ | `make_numpy_detection_model` | ✓ live |
| Open-vocab detection | mAP ↑ | `make_numpy_detection_model` | ✓ live |
| Object tracking | MOTA ↑ | `make_numpy_tracking_model` | ✓ live (manifest done) |
| Visual grounding | grounding_acc ↑ | `make_numpy_grounding_model` | ⚠ awaits VQA labels |
| Relative camera pose | rotation_error_deg ↓ | `make_numpy_pose_model` | ✓ live (paired manifest) |
| Sparse depth | RMSE ↓ | `make_numpy_sparse_depth_model` | ✓ live |
| Novel view synthesis | PSNR ↑ | `make_numpy_nvs_model` | ✓ live |
| Keypoint matching | keypoint_acc ↑ | `make_numpy_keypoint_model` | ✓ live |

## Dataset access

```python
# 1. HuggingFace `datasets` (streaming, one-liner)
from rpx_benchmark.data import load_hf
ds = load_hf("monocular_depth", split="hard")

# 2. Modality-aware snapshot (offline-friendly bulk; auto-extracts tars)
from rpx_benchmark.hub import load
ds = load("monocular_depth", "hard")
```

Both yield `list[Sample]` batches the runner consumes interchangeably.

## Visualize a scene

One command opens an interactive [Rerun](https://rerun.io/) viewer with all six modalities (RGB, depth, fisheye stereo, instance masks, and the T265 6-DoF camera trajectory) on a shared timeline.

![RPX in Rerun: 3 phases × 6 panels (cam pose, RGB, cividis depth, royal-tone masks, fisheye L, fisheye R)](docs/assets/rpx_rerun_viewer.png)

```bash
make visualize            # stream into viewer; no disk artifact (HF cache only)
make visualize-phases     # all 3 phases (clutter / interaction / clean) side-by-side
make visualize-save       # also write a portable .rrd to benchmark/site/
make visualize-lite       # constrained devices (Pi 4 / Jetson Nano): stride=3, smaller
make visualize-headless   # build .rrd only, no viewer (CI / SSH)
```

### Single-phase layout

```
┌──────────────┬──────────────┬──────────────┐
│   Cam Pose   │  Fisheye L   │  Fisheye R   │
├──────────────┼──────────────┼──────────────┤
│     RGB      │ Depth (mm)   │    Masks     │
└──────────────┴──────────────┴──────────────┘
```

### Visual conventions

- **Depth**: cividis colormap, **per-frame histogram-equalized** so subtle structure remains visible. Invalid (`raw == 0`) pixels render black.
- **Masks**: hand-curated **royal jewel-tone palette** with deterministic `id → palette[id mod 16]` mapping. Same instance ID → same color across frames *and* phases.
- **Camera pose**: per-phase trajectory rendered as a colored polyline (magenta = clutter, orange = interaction, cyan = clean) with a Pinhole frustum sweeping along the path.
- **Fisheye stereo**: T265 monochrome, percentile-stretched per frame.

See `python scripts/visualize_rerun.py --help` for the full flag set.

## Install extras

| Extra | Pulls | Use when |
|---|---|---|
| *(core)* | `numpy`, `Pillow` | Always |
| `hub` | `huggingface_hub[hf_xet]` | Downloading from HF |
| `hf-datasets` | `datasets>=2.18` | `load_hf(...)` one-liner |
| `schemas` | `pydantic>=2.5` | Strict manifest validation |
| `viz` | `rerun-sdk`, `matplotlib`, `Pillow`, `tqdm` | `make visualize` (interactive Rerun) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`, `pre-commit` | Contributing |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs locally |

## Quality status

- **524 tests passing** (`PYTHONPATH=. pytest tests/`), 2 expected skips on paired-task mocks.
- **Single canonical seed** `RPX_SEED = 5_062_026` (MMDDYYYY 05/06/2026) shared across bootstrap CIs, ORD pair sampler, sparse-depth sampler, keypoint-pair sampler, and tests.
- **Single source of truth for alignment** in `rpx_benchmark.metrics.depth_alignment` — no duplicated math between runner and post-processor.
- **9 live mono-depth adapters**, all reproducible from a single URL each. 11 more pending.
- **Zero bare exceptions** in the team's hot-path code; everything raises typed `AdapterError` / `ConfigError` / `DatasetError` with hints.
- **2 contract tests** for Box upload (mock-based, no token needed in CI) verify the scene/phase tree shape and idempotency on re-upload.
- **7 integration tests** for the writer → loader round-trip catch silent contract bugs (entry-key drift, file-extension drift, extraction-gap regressions).

## Docs

Full site: <https://irvlutd.github.io/RPX/>

- [Quickstart](docs/getting-started/quickstart.md)
- [Load the Dataset](docs/getting-started/load-the-dataset.md)
- [Bring Your Own Model](docs/getting-started/bring-your-own-model.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Adding a Task](docs/guides/adding-a-task.md)
- [Adding a Metric](docs/guides/adding-a-metric.md)
- [Contributing](docs/guides/contributing.md)

## License

MIT — see [LICENSE](LICENSE).
