# RPX Benchmark Toolkit

> Benchmark robot-perception models on real-world RGB-D across
> clutter / interaction / clean phases of 99 indoor + outdoor scenes —
> with ESD difficulty splits and three separately-reported axes
> (task accuracy · scene-change robustness · compute cost).

```bash
pip install 'rpx-benchmark[depth]'
rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard
```

Writes `rpx_results/<model>/<split>/result.json` reporting all three
axes — task accuracy, scene-change robustness, compute cost — plus
per-stage timing with 95% bootstrap CIs. No combined score.

---

### Quickstart walkthrough

```bash
pip install 'rpx-benchmark[hub,depth]'
hf auth login                      # only if the dataset repo is private

rpx bench monocular_depth \
    --hf-checkpoint Intel/zoedepth-nyu-kitti \
    --split easy \
    --max-samples 50               # smoke run on 50 frames
```

The toolkit downloads the easy split into `~/.cache/huggingface/`
(first run only), fetches the checkpoint, runs the model frame-by-frame,
and writes `rpx_results/zoedepth-nyu-kitti/easy/{result.json, summary.md}`.

`summary.md` is the load-bearing sanity check. RPX reports each model
on **three independent axes** — never combining them — so you can pick
the one your deployment cares about:

```text
zoedepth-nyu-kitti — easy

  Task accuracy            AbsRel  = 0.0848   (lower is better)
                           δ₁.₂₅   = 0.951    (higher is better)

  Scene-change robustness  STR     = 0.91     (1.0 = perfectly stable)
                           Δ(interaction−clean)  AbsRel = +0.024

  Compute cost             Params  = 345 M
                           FLOPs   = 4.9 G
                           Latency†= 158.5 ms / sample  (RTX 5070 Laptop)
```

†Latency is hardware-dependent — reported for reference, not for
ranking. If those numbers print, your install is healthy. Move on to
[**Bring your own model**](#bring-your-own-model).

<details>
<summary><b>Reading <code>result.json</code> — what every field means</b></summary>

`result.json` is organised around the three reporting axes. The table
below groups each key by which axis it belongs to.

**Axis 1 — Task accuracy**

| Key | Meaning | Use when |
|---|---|---|
| `aggregated.absrel`, `delta1`, … | Per-task primary metrics, averaged over all frames. | Comparing models head-to-head on the same split. |
| `timing.metrics_with_ci` | Each metric with 95% bootstrap CIs and `n` samples. | Reporting "AbsRel = 0.085 ± 0.002 (95% CI)". |
| `comprehensive_metrics.aggregated_with_ci` *(separate file)* | The full metric basket — errors × alignment modes × depth bands × in/out-of-mask × per-object. | Detailed per-object and per-depth-band analysis. |

**Axis 2 — Scene-change robustness**

| Key | Meaning | Use when |
|---|---|---|
| `deployment_readiness.operating_point.str_score` *(legacy path; see deprecation note)* | State-Transition Robustness — how stable the output is across clutter → interaction → clean. Higher is better. | The headline robustness number for any per-model comparison. |
| Per-phase entries in `comprehensive_metrics` | Phase-stratified versions of every metric (e.g. `absrel_per_phase.{clutter, interaction, clean}`). | Computing cross-phase Δ explicitly: `Δ = metric(interaction) − metric(clean)`. |

**Axis 3 — Compute cost**

| Key | Meaning | Use when |
|---|---|---|
| `deployment_readiness.params_m`, `flops_g` *(legacy path)* | Static cost: parameters in millions, multiply-add FLOPs in billions. | "Is this model small enough for my robot?" |
| `deployment_readiness.roofline.<gpu>.latency_ms` *(legacy path)* | Theoretical lower-bound latency on the named GPU, from FLOPs + memory traffic. | Estimating performance on hardware you don't have. |
| `timing.model_run.mean` | Measured wall-clock latency per sample, with 95% bootstrap CIs. Hardware-dependent — supplementary. | Reporting "on RTX 5070 Laptop: 158 ± 0.5 ms". |
| `timing.data_load.mean` / `metric_calc.mean` | Per-stage wall-clock with CIs. | Diagnosing whether the bottleneck is I/O, model, or metric code. |

**Rule of thumb on the CIs.** At `n ≥ 1000` samples, CI half-widths
should be small (< 5% of the mean). Wide CIs mean the run hasn't
converged yet — add more `--max-samples`.

> **Deprecation notice.** The `deployment_readiness` block name is
> being retired. A future release will move its contents to three
> top-level keys mirroring the axes: `task_accuracy`, `robustness`,
> `compute_cost`. The old composite DRS scalar (`TP × R × E`) is
> being removed — it disguised the disagreement across axes that is
> the paper's central finding. Track this in
> [`benchmark/SHARED_CONTEXT.md`](SHARED_CONTEXT.md).

</details>

---

### Pick your path

|                                  |                                  |                                  |
|----------------------------------|----------------------------------|----------------------------------|
| 🔌 **External — your own model** | 🤖 **Team — canonical sweep**     | 📚 **Contributor — toolkit**     |
| Wrap a HF checkpoint or any      | Reproduce the full 19-depth +    | Add a task, a metric, an adapter.|
| numpy callable in one function.  | 10-pose sweep behind the paper.  | Plugin registries, 552 tests.    |
| → [Bring your own model](#bring-your-own-model) | → [Canonical sweep](#canonical-sweep) (or [`TEAM_LAUNCH.md`](TEAM_LAUNCH.md)) | → [What the toolkit ships](#what-the-toolkit-ships) |

---

## Bring your own model

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

<details>
<summary><b>Mirror outputs to Box automatically</b></summary>

Every `TaskRunConfig` accepts `upload_to_box=True` so analysis artefacts
(`result.json`, `summary.md`, comprehensive metrics, predictions) are
pushed to Box right after the run finishes:

```python
cfg = rpx.MonocularDepthRunConfig(
    model=model, split="hard",
    upload_to_box=True,                      # off by default
    # box_folder_id="380510613151",          # team's RPX-Outputs by default
)
result, report, paths = rpx.run_monocular_depth(cfg)
print(paths["box_remote"])  # "monocular_depth/<name>/hard"
```

Requires `BOX_DEVELOPER_TOKEN` in the environment (60-min token from
the Box developer console). Upload is **size-matched idempotent** — safe
to re-run; files already on Box are skipped. If the upload fails
(e.g. token expired mid-sweep) the local artefacts on disk are
preserved and you can recover with the post-hoc sync helper:

```bash
PYTHONPATH=. python scripts/sync_results_to_box.py    # mirrors everything under rpx_results/
```

</details>

<details>
<summary><b>Three integration paths</b></summary>

1. **Plain numpy callable.** Ten factories, one per task. Shown above.
2. **Custom `InputAdapter` / `OutputAdapter`.** When you need full
   control over preprocessing, batching, or post-processing.
3. **API / cloud model.** Wrap a remote inference service as an
   adapter and mark it `EfficiencyMetadata(model_type="api")`.

See [`docs/getting-started/bring-your-own-model.md`](docs/getting-started/bring-your-own-model.md)
for complete templates of all three.

</details>

<details>
<summary><b>True-batched dispatch (GPU pipelines)</b></summary>

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
following this pattern.

</details>

---

## Canonical sweep

The day-the-data-lands launch sequence for the team. **One-page
version: [`TEAM_LAUNCH.md`](TEAM_LAUNCH.md).**

```bash
# 0. One-time
pip install -e 'benchmark[hub,viz]'
hf auth login
export BOX_DEVELOPER_TOKEN='<60-min-token>'

# 1. Push dataset to HuggingFace (run once per dataset version)
python -m rpx_benchmark.dataset_hub.cli pack       --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli manifest   --src DATA --staging STAGE \
       --splits benchmark/data/splits/scene_splits.json
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli dataset-card    --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli upload          --staging STAGE --repo-id IRVLUTD/RPX

# 2. Run depth — per model, per split
PYTHONPATH=. python scripts/run_depth.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box

# 3. Run relative pose (RCPE)
PYTHONPATH=. python scripts/run_relative_pose.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box

# 4. Aggregate into the paper table
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy --sensitivity
```

Idempotent at every step. Outputs land under `rpx_results/<model>/<split>/`
and `rpx_results/_sweep/`. Hub upload details:
[`rpx_benchmark/dataset_hub/README.md`](rpx_benchmark/dataset_hub/README.md).

<details>
<summary><b>Depth zoo — 19 registered keys (9 metric · 10 relative)</b></summary>

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

**Run all 19 across all 3 splits:**

```bash
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

</details>

<details>
<summary><b>RCPE zoo — 10 relative-pose adapters</b></summary>

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

```bash
pip install kornia opencv-contrib-python   # loftr, opencv_baseline
pip install open3d                          # icp_open3d
pip install reloc3r                         # reloc3r
# dust3r / mast3r / far / srpose / nope_sac / mickey: clone upstream repo + checkpoint
```

</details>

<details>
<summary><b>Output schema (per-run <code>result.json</code>)</b></summary>

```jsonc
{
  "task": "monocular_depth",
  "model": "ZoeDepth_NK",
  "split": "easy",
  "num_samples": 3000,
  "aggregated":  { "absrel": 0.0848, "delta1": 0.951, "rmse": 0.205, ... },
  "deployment_readiness": {
    "params_m": 345.07,
    "flops_g": 4878.9,
    "macs_g": 2439.4,
    "memory_traffic_gb": 4.14,
    "arithmetic_intensity": 1178.2,
    "roofline": {
      "A100-80GB":         { "latency_ms": 250.2, "bottleneck": "compute" },
      "RTX 4090":          { "latency_ms":  59.1, "bottleneck": "compute" },
      "Jetson Orin 64GB":  { "latency_ms": 920.5, "bottleneck": "compute" }
    },
    "latency_ms_per_sample": 158.5,
    "system_card": { "gpu_name": "...", "pytorch_version": "...", "cuda_version": "..." },
    "operating_point": {
      "precision": "fp32",
      "task_metric": 0.027,  "task_metric_name": "absrel",  "higher_is_better": false,
      "str_score": -0.085,  "flops_g": 4878.9, "params_m": 345.07
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

**Predictions (when `--save-predictions`)**:
- Depth: `predictions/<scene>/<phase>/<frame>.npz` — per-frame raw depth.
- Pose: `predictions.csv` — single CSV with columns
  `scene_id, phase, frame_a, frame_b, R00..R22, tx, ty, tz` (16 cols,
  resume-safe — header written exactly once).

**Box mirror (when `--upload-to-box`)**: `<box_root>/<task>/<model>/<split>/...`
mirrors the local tree exactly. Idempotent (size-matched skip on re-upload).

</details>

<details>
<summary><b>Known limits</b></summary>

- **`DEPTH_MAX_M = 5.0`** in `rpx_benchmark/metrics/depth_alignment.py`.
  D435 is unreliable beyond ~6 m; pixels with GT > 5 m are silently
  excluded. RPX has 35 outdoor scenes with GT > 5 m → those models are
  evaluated only on their <5 m subset. Decision needed before paper
  sweep: raise the cap, stratify by domain, or document and move on.
- **Box dev tokens expire every 60 min** — every relaunch needs a fresh
  export. The pipeline raises
  `ConfigError("Box API 401 — token expired", hint="…regenerate at developer console")`
  cleanly when stale.
- **VQA recipe** is wired but emits zero entries (waits on the team's
  VQA label-generation pipeline). All 7 other tasks ship.

</details>

---

## What the toolkit ships

| Area | Module | What's in it |
|---|---|---|
| Data contracts | `rpx_benchmark.api` | `TaskType`, `Sample`, 10 × GroundTruth, 10 × Prediction, `BenchmarkModel` ABC |
| Dataset | `rpx_benchmark.loader`, `rpx_benchmark.hub`, `rpx_benchmark.data` | `RPXDataset`, modality-aware snapshot downloads (auto-extracts tars), `load_hf()` one-liner |
| Manifest validation | `rpx_benchmark.schemas` | Pydantic v2 schemas + JSON Schema export (opt-in extra) |
| Adapter framework | `rpx_benchmark.adapters` | `BenchmarkableModel`, batched-dispatch base class, 9 numpy fast-path factories |
| Metrics | `rpx_benchmark.metrics.*` | Per-task `MetricCalculator` registry; `depth_alignment` shared between runner + comprehensive post-processor |
| Tasks | `rpx_benchmark.tasks.*` | One module per task, each ~30 lines; self-register `TaskSpec` |
| Runner | `rpx_benchmark.runner` | Orchestrates dataset → model → metrics → report; persists per-axis components into `result.json` |
| Profiler | `rpx_benchmark.profiler` | Compute-cost axis: Tier 1 (params/FLOPs/MACs/memory traffic/AI), Tier 2 (roofline for A100/4090/Orin), Tier 3 (measured + system card) |
| Per-axis aggregators | `rpx_benchmark.deployment` | Scene-change-robustness axis (STR, ESD-weighted Phase Score, Temporal Stability, SGC) — emitted separately, never combined into a single score. *Legacy DRS aggregator deprecated — see [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md).* |
| Sweep utilities | `scripts/run_drs_sweep.py`, `rpx_benchmark.drs_sensitivity` | Sweep aggregator + sensitivity Kendall's τ. *Both being retired alongside the DRS scalar — track in `SHARED_CONTEXT.md`.* |
| Determinism | `rpx_benchmark.determinism` | `seed_all()`, `deterministic()` context manager, project-wide `RPX_SEED = 5_062_026` |
| Reports | `rpx_benchmark.reports` | JSON + Markdown writers |
| CLI UX | `rpx_benchmark.cli_ux` | Industry-grade splash + progress for all runnable scripts (rich-based; CI-degrading) |

<details>
<summary><b>The 10 task contracts (loader-side status)</b></summary>

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

</details>

<details>
<summary><b>Dataset access · install extras · visualization</b></summary>

**Two ways to load:**

```python
# 1. HuggingFace `datasets` (streaming, one-liner)
from rpx_benchmark.data import load_hf
ds = load_hf("monocular_depth", split="hard")

# 2. Modality-aware snapshot (offline-friendly bulk; auto-extracts tars)
from rpx_benchmark.hub import load
ds = load("monocular_depth", "hard")
```

Both yield `list[Sample]` batches the runner consumes interchangeably.

**Install extras:**

| Extra | Pulls | Use when |
|---|---|---|
| *(core)* | `numpy`, `Pillow`, `rich` | Always |
| `hub` | `huggingface_hub[hf_xet]` | Downloading from HF |
| `hf-datasets` | `datasets>=2.18` | `load_hf(...)` one-liner |
| `schemas` | `pydantic>=2.5` | Strict manifest validation |
| `viz` | `rerun-sdk`, `matplotlib`, `Pillow`, `tqdm` | `make visualize` (interactive Rerun) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`, `pre-commit` | Contributing |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs locally |

**Visualize a scene** — one command opens an interactive
[Rerun](https://rerun.io/) viewer with all six modalities (RGB, depth,
fisheye stereo, instance masks, T265 6-DoF camera trajectory) on a
shared timeline.

![RPX in Rerun: 3 phases × 6 panels (cam pose, RGB, cividis depth, royal-tone masks, fisheye L, fisheye R)](docs/assets/rpx_rerun_viewer.png)

```bash
make visualize            # stream into viewer; no disk artifact (HF cache only)
make visualize-phases     # all 3 phases (clutter / interaction / clean) side-by-side
make visualize-save       # also write a portable .rrd to benchmark/site/
make visualize-lite       # constrained devices (Pi 4 / Jetson Nano): stride=3, smaller
make visualize-headless   # build .rrd only, no viewer (CI / SSH)
```

Visual conventions: **depth** uses cividis with per-frame histogram
equalization (invalid pixels black); **masks** use a hand-curated royal
jewel-tone palette with deterministic `id → palette[id mod 16]`;
**camera pose** is a colored polyline per phase
(magenta = clutter, orange = interaction, cyan = clean);
**fisheye stereo** is percentile-stretched per frame.

</details>

<details>
<summary><b>Quality status</b></summary>

- **552 tests passing** (`PYTHONPATH=. pytest tests/`), 2 expected
  skips on paired-task mocks.
- **Single canonical seed** `RPX_SEED = 5_062_026` (MMDDYYYY 05/06/2026)
  shared across bootstrap CIs, ORD pair sampler, sparse-depth sampler,
  keypoint-pair sampler, and tests.
- **Single source of truth for alignment** in
  `rpx_benchmark.metrics.depth_alignment` — no duplicated math
  between runner and post-processor.
- **19 live mono-depth adapters · 10 live RCPE adapters**, all
  reproducible from a single URL each.
- **Zero bare exceptions** in the team's hot-path code; everything
  raises typed `AdapterError` / `ConfigError` / `DatasetError`
  with hints.
- **Industry-grade CLI UX** — every script opens with the animated
  RPX splash; helpers in `rpx_benchmark.cli_ux` are CI-degrading,
  honour `NO_COLOR`, `RPX_QUIET`, `RPX_ANIM`.

</details>

---

## Docs

Full site: <https://irvlutd.github.io/RPX/>

- [Quickstart](docs/getting-started/quickstart.md)
- [Load the dataset](docs/getting-started/load-the-dataset.md)
- [Bring your own model](docs/getting-started/bring-your-own-model.md)
- [Architecture overview](docs/architecture/overview.md)
- [Adding a task](docs/guides/adding-a-task.md)
- [Adding a metric](docs/guides/adding-a-metric.md)
- [Contributing](docs/guides/contributing.md)

## License

MIT — see [LICENSE](LICENSE).
