# RPX Benchmark — Shared Session Context

> **Purpose**: Coordination file between parallel Claude sessions working on the RPX monocular depth benchmarking pipeline.  
> **Both sessions should read this before starting and append updates with timestamps.**

---

## Current State (2025-05-07)

### Session A (Feynman): Model Survey & Documentation
- Completed survey of SOTA monocular metric depth estimation models (2024–2025)
- Verified open-source code + checkpoint availability for each
- Created `docs/methods/mono_depth_models.md` with full details
- Models confirmed available for benchmarking: UniDepth V2, Depth Pro, Depth Anything V2 (metric), Metric3D v2, MoGe-2, MetricSolver, Marigold, Lotus/Lotus-2, ZoeDepth

### Session B (Claude — HF data pipeline): [STATUS PENDING]
- Working on: test HF data loading for benchmark pipeline
- [Please update below when progress is made]

---

## Models Ready for Benchmark Integration

These have **verified** open code + public checkpoints as of 2025-05-07:

| Model | HF Checkpoint | Metric Depth? | Priority |
|-------|---------------|---------------|----------|
| UniDepth V2 | `lpiccinelli/unidepth-v2-vitl14` | Yes | P0 |
| Depth Pro | `apple/DepthPro` / `apple/DepthPro-hf` | Yes | P0 |
| Depth Anything V2 (Metric-Hypersim) | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` | Yes (indoor) | P0 |
| Metric3D v2 | weights in repo (auto-download) | Yes | P1 |
| MoGe-2 | `Ruicheng/moge-2-vitl-normal` | Yes | P1 |
| MetricSolver | weights in repo | Yes | P1 |
| Marigold | `prs-eth/marigold-*` | Relative (needs alignment) | P2 |
| Lotus-2 | `jingheya/Lotus-2` | Relative (needs alignment) | P2 |
| ZoeDepth | weights in repo | Yes | P2 (archived) |

## Models NOT Available (paper-only as of 2025-05-07)

| Model | Paper | Why unavailable |
|-------|-------|-----------------|
| AsyncMDE | arXiv:2603.10438 | No code or weights released |
| RTS-Mono | arXiv:2511.14107 | No code or weights released |
| PTC-Depth | arXiv:2604.01791 | No code or weights released |
| AnyDepth | arXiv:2601.02760 | No code or weights released |
| KineDepth | arXiv:2409.19490 | Code tightly coupled to specific robot setup |

---

## Interface Contract (for adapter writing)

The benchmark uses `BenchmarkableModel` ABC. For depth, the numpy fast-path is:

```python
from rpx_benchmark import make_numpy_depth_model

model = make_numpy_depth_model(my_inference_fn)
# my_inference_fn(rgb: np.ndarray[H,W,3] uint8) -> np.ndarray[H,W] float32 (meters)
```

No model registry exists (removed in v0.3.0). Each model needs a standalone adapter script.

---

## Task Metrics

**IMPORTANT**: Compute ALL metrics, not just primary ones. Full list in `docs/methods/comprehensive_metrics.md`.

### Depth — compute ALL of these:
- **Error**: AbsRel, SqRel, RMSE, RMSElog, SIlog, log10, MAE, iRMSE, iMAE
- **Accuracy**: δ1 (1.25), δ2 (1.25²), δ3 (1.25³)
- **Boundary**: F-score, edge accuracy, edge completeness
- **Alignment modes**: None (for metric models), Least-squares affine (for relative models), Least-squares in disparity
- **Stratify by**: Near/Mid/Far depth bands, object vs background, edge vs interior, per-phase (clutter/interaction/clean)
- **Robotics**: Planarity error on table surface, object-relative depth ordering

### Segmentation — compute ALL of these:
- mIoU, AP, AP50, AP75, AP@[.5:.95], APˢ/APᴹ/APᴸ, PQ, SQ, RQ, Boundary IoU, Pixel Accuracy, Mean Accuracy, FWIoU

### Detection — compute ALL of these:
- mAP, mAP@[.5:.95], AP50, AP75, APˢ/APᴹ/APᴸ, AR₁/AR₁₀/AR₁₀₀, per-category AP

### Pose — compute ALL of these:
- Rotation error (deg), Translation error (m), Translation angular error (deg), AUC@5°/10°/20°, Median errors, Pose recall@(5°,5cm)/(5°,10cm)/(10°,25cm)

### Keypoints — compute ALL of these:
- Accuracy @3px/@5px/@10px, Mean error, Precision, Recall, Pose estimation AUC, Inlier ratio

### NVS — compute ALL of these:
- PSNR, SSIM (windowed, not global), LPIPS, MS-SSIM

### Tracking — compute ALL of these:
- MOTA, MOTP, IDF1, HOTA, DetA, AssA, MT, ML, FP, FN, IDsw, Frag

### Robot VQA — compute ALL of these:
- Overall accuracy, per-T1–T6, per-phase, per-operation, STR, consistency across viewpoints

---

## Updates Log

### 2025-05-07T[now] — Feynman
- Created this file
- Created `docs/methods/mono_depth_models.md`
- Survey complete for monocular depth models

### 2026-05-06T~21:40 — Session B (Claude — HF data pipeline)

**What I've done:**
- Built `scripts/box_fetch.py` to pull from UTD Box (anonymous shared links + `BOX_DEVELOPER_TOKEN` REST-API path); cached at `~/.cache/rpx-box/`. Fetched the team's tracker xlsx — task list there is **7** (NVS, RelativePose, UOIS, MonocularMetricDepth, VisualGrounding, MultiViewCorrespondences, Real2Sim2Real); the brief still lists 6 and needs an update.
- Built `scripts/depth_models/zoedepth.py` (HF transformers `Intel/zoedepth-nyu-kitti` adapter; rgb→metric metres) and `scripts/smoke_depth.py` — direct cache reader, validates the adapter end-to-end on RPX imagery without going through the toolkit's `download_split`.
- Smoke result on 4 frames (3 phases of `scene23.jo.4f` + 1 of `scene54.TENNISCOURTS.Ground`): AbsRel 0.08–0.24, δ₁ 0.40–0.99, ~0.18 s/frame on RTX 5070 after warm-up. Phase-state sensitivity already visible (δ₁ for clutter=0.85 / interaction=0.99 / **clean=0.40** on the same scene) — RPX's three-phase signal is doing what it should.

**Architectural gap I hit (worth fixing before more model adapters):**
- `rpx_benchmark/hub.py:download_split` expects `manifests/<task>/<split>.json` on the HF repo. The dataset_hub writer side has scanner / packer / Croissant / dataset_card / **frames Parquet manifest** (`manifest.py:build_frame_manifest`) — but **no code that emits the per-task per-split JSON manifests** the reader needs. The test mirror (`itaykadosh/rpx-test`) has the raw shards but no `manifests/monocular_depth/easy.json`.
- This is a real bridging hole in `dataset_hub/`. Two ways to close it:
  - (a) Add `dataset_hub/split_manifests.py` that, given the all-frames Parquet + `MULTI_OBJECT_TASK_RECIPES` + `splits/scene_splits.json`, emits one `manifests/<task>/<split>.json` per (task, split) cell. **Proper fix**; once committed and re-uploaded, every task pipeline works on the test mirror without any bypass.
  - (b) Local-manifest generator that writes the same JSON to a temp path; `RPXDataset.from_manifest(local)` works without HF round-trip. Quicker to land, doesn't fix the underlying gap.
- Going to do (b) first to unblock benchmarking against the test mirror, then (a) so the team's hosted artefacts are self-sufficient.

**Files added:**
- `benchmark/scripts/box_fetch.py` — Box CLI + Python API
- `benchmark/scripts/depth_models/zoedepth.py` — first reference adapter
- `benchmark/scripts/run_depth.py` — wires adapter into `run_monocular_depth` (waiting on manifest plumbing)
- `benchmark/scripts/smoke_depth.py` — direct-cache smoke test (works now)

**Aligning with Feynman's priorities:**
- ZoeDepth was just the smoke (you marked it P2-archived — fair). After fixing the manifest gap, next adapters in priority order are **UniDepth V2 (P0)**, **Depth Pro (P0)**, **Depth Anything V2 Metric (P0)**. Each follows the same template: HF-transformers / torch-hub load → wrap as `predict(rgb_uint8)→depth_metres_float32` → `make_numpy_depth_model(fn)`.
- Marigold/Lotus-2 (P2, relative depth) need scale alignment — separate adapter pattern; will design once the metric ones land.

**Open question for Feynman:** the brief at `paper-submission/BRIEF_FOR_ADVISOR.tex` lists 6 tasks; tracker has 7 (adds NVS, Pose, Real2Sim2Real; renames "segmentation" → "UOIS" reflecting IRVLUTD's line). Has the prof seen the brief yet, and if so was the task list contested? If not, I'll update the brief to mirror the tracker.

### 2026-05-06T~22:30 — Session B (Claude — depth pipeline iteration)

**What landed since last update:**
- `scripts/local_manifest.py` — bridges the dataset_hub gap. Reads the cached `manifest/frames_v1.parquet`, applies **scene-wise splits** (mode-aggregated; STR-bias-safe), filters to the per-task required modalities, extracts only the needed (modality, frame) pairs from tar shards into a flat `<snap>/extracted/` layout, and writes the missing `manifests/<task>/<split>.json`. Handles `phase: int` correctly (toolkit's `_PHASE_INDEX` maps to enum names).
- `scripts/run_depth.py` — two paths: official `download_split` if manifests are published on HF; otherwise local-manifest. Wraps the model's predict callable with a `_PredictionDumper` when `--save-predictions` is on (off by default; community-friendly). On `--upload-to-box`, ships everything to UTD Box at `<root>/<task>/<model>/<split>/`. The Box folder id default is the team's RPX-Outputs root (380510613151).
- `scripts/box_fetch.py` — added the writer side: `_api_ensure_folder`, `_api_resolve_path`, `_api_upload_file`, `upload_tree`. Idempotent (skips files whose Box copy has matching size; uses new-version upload on size-mismatch).
- `scripts/comprehensive_depth_metrics.py` — full Feynman spec post-processor. Reads per-frame `.npz` predictions + GT depth + GT masks (loaded directly from cached `masks/v1.tar`). Emits 9 error metrics (AbsRel, SqRel, RMSE, RMSElog, SIlog, log10, MAE, iRMSE, iMAE) + 3 accuracy thresholds + alignment modes (none / median / ls_affine / ls_disparity) + depth-band stratification (near/mid/far) + in-mask/out-of-mask stratification + boundary metrics (F-score, edge accuracy, edge completeness) + ORD (object-relative depth ordering). Skipped for v1: planarity (needs plane priors), depth-discontinuity recall.
- `--comprehensive-metrics` flag on `run_depth.py` auto-enables `--save-predictions` and runs the post-processor; output lands in `comprehensive_metrics.json`.

**Validation — ZoeDepth on 3 frames of `scene23.jo.4f` phase 0, alignment=none:**
- AbsRel 0.0825, RMSE 0.205 m, RMSElog 0.114, SIlog 7.30, log10 0.039, MAE 0.157, iRMSE 0.066, iMAE 0.052, δ1 0.948, δ2/δ3 1.0. All values land in plausible ranges for ZoeDepth on indoor tabletop imagery.
- Mid-band metrics dominate (this scene is ~1.3–1.5 m); near and far stratifications report no samples (nothing in 0.3–1 m or 3–5 m range for these specific frames).
- Mask-dependent metrics (in_mask/out_mask, boundary/*, ord/*) didn't show up in this 3-frame smoke — debugging next; mask tar exists at the expected path so it's likely a member-name or sample-row mismatch.

**End-to-end Box upload verified:** team-mode run (`--save-predictions --upload-to-box`) shipped 3 .npz predictions + result.json + summary.md to `Box/RPX-Outputs/monocular_depth/ZoeDepth_NK/easy/` (folder id 380511544736, auto-created by `_api_resolve_path`).

**Architectural posture:**
- Local-manifest path is shipped (b). The proper dataset_hub fix (a) — promote `local_manifest.build_local_manifest` to `dataset_hub/split_manifests.py` and re-upload — is still on the list but unblocked: every team-side dev workflow works without it.
- Comprehensive metrics module lives in `scripts/`, not `rpx_benchmark/`, since it's a post-processor. If we want it inside the toolkit's metric registry later (so the standard `BenchmarkRunner` aggregated dict carries the full basket), it can move to `rpx_benchmark/metrics/depth_comprehensive.py` and register against `TaskType.MONOCULAR_DEPTH`.

**Remaining mask-loading bug to fix:** `_load_mask_from_cache` in `comprehensive_depth_metrics.py` — the v1.tar is at the expected path but the per-frame metrics didn't include `in_mask/`, `boundary/`, or `ord/` keys. Need to verify the mask member name format (likely `masks/<frame>.png`) and the sample's `phase` integer agrees with the tar layout.

**Next moves (in order):**
1. Fix the mask-loading bug so in_mask + boundary + ORD metrics start emitting.
2. Add the P0 adapters: UniDepth V2, Depth Pro, Depth Anything V2 Metric. ~30 lines each, follow the ZoeDepth template.
3. Promote `local_manifest.py` to `dataset_hub/split_manifests.py` so the team's writer side is whole.

**Per the user's directive (just-now):** every change to this project should update code + README + brief PDF + this file together, in the same turn. Acknowledged.

### 2025-05-07 — Feynman (model survey + profiler)
- Created `docs/methods/mono_depth_models.md` + PDF — model survey: 9 available, 5 unavailable with dates/reasons
- Created `docs/methods/comprehensive_metrics.md` + PDF — exhaustive metrics for ALL 10 tasks
- **Implemented 3-tier hardware-agnostic resource reporting** in `rpx_benchmark/profiler.py`:
  - **Tier 1** (hw-agnostic): Params, FLOPs, MACs, memory traffic (GB), arithmetic intensity (FLOP/Byte)
  - **Tier 2** (roofline): latency lower-bounds for A100, RTX 4090, Jetson Orin via roofline model
  - **Tier 3** (measured): wall-clock latency + SystemCard auto-detection (GPU, CPU, CUDA, PyTorch, OS)
- New classes: `GPUSpec`, `RooflineBound`, `SystemCard`, `estimate_memory_traffic_gb()`
- `profile_model()` now fills all three tiers automatically
- `EfficiencyMetadata.derive_tier1()` → MACs, traffic, arithmetic intensity
- `EfficiencyMetadata.compute_roofline()` → bounds for 3 reference GPUs
- `REFERENCE_GPUS` dict: A100-80GB, RTX 4090, Jetson Orin 64GB
- 32 tests passing (22 new in `test_roofline_profiler.py` + 10 existing)
- Updated `__init__.py` exports: `GPUSpec`, `RooflineBound`, `SystemCard`, `REFERENCE_GPUS`, `estimate_memory_traffic_gb`
- PDFs regenerated for both docs

### 2026-05-04T~13:30 — Session B (Claude — profiler integration + per-stage timing)
**Wired Feynman's 3-tier profiler into `run_depth.py`'s local-manifest path; added per-stage timing breakdown.**

- **Profiler walk-in**: ZoeDepth adapter now exposes `torch_module` so the runner can reach `transformers.ZoeDepthForDepthEstimation` (345.073 M params). The walker tries `("torch_module", "_pipe.model", "_model", "model")` against the *original adapter* (not the BenchmarkableModel — it gets re-wrapped by `_PredictionDumper` when `--save-predictions` is on, which would otherwise hide the underlying nn.Module).
- **Resilient tier computation**: I do NOT call `profile_model()` directly because its `count_flops_torch` step OOMs on 8 GB consumer GPUs (e.g. RTX 5070 Laptop, 7.5 GB). Instead I compute each Tier-1 component independently (`count_parameters`, `count_flops_torch`, `estimate_memory_traffic_gb`) with per-call try/except so a single failure doesn't drop the rest. Then `eff.derive_tier1()` for MACs/AI, `eff.compute_roofline()` for Tier 2, `SystemCard.auto_detect()` for Tier 3 metadata.
- **Re-derive after runner**: `BenchmarkRunner.run_with_deployment_readiness` populates `eff.flops_g` via its own flop counter (`first_batch_flops_g`). I therefore re-call `eff.derive_tier1()` and `eff.compute_roofline()` *after* the runner returns, so MACs/AI/roofline pick up the runner's flops_g when the local profiler couldn't.
- **DR report drops most efficiency fields**. The runner's `DeploymentReadinessReport` only persists 5 of the new `EfficiencyMetadata` fields. I added `_augment_result_json()` post-runner — it injects every Tier-1/2/3 field + system card under `result.json["efficiency"]` so downstream analytics has the full Feynman-spec picture without needing changes on Feynman's side.
- **Per-stage timing** (user request "data loading time + model running time + metric calculation time"): `_stage_timing(adapter, dataset)` re-runs the dataset with three independent timers — sample decode (`dataset._load_sample`), model forward (CUDA-synced before/after), and a single AbsRel pass. Reports mean / median / p95 / n in ms per stage at top-level `result.json["timing"]`. Keeps the runner's own end-to-end `latency_ms` untouched so adapters stay comparable on the official metric.

**ZoeDepth on `scene23.jo.4f` ph0, n=2, RTX 5070 Laptop (7.5 GB, fp32, 480×640):**
- Tier 1: params 345.07 M, FLOPs 4878.9 G, MACs 2439.4 G, mem-traffic 4.14 GB, arithmetic intensity 1178.2 FLOP/B
- Tier 2 (roofline): A100-80GB → 250.2 ms / RTX 4090 → 59.1 ms / Jetson Orin 64 GB → 920.5 ms (all compute-bound)
- Tier 3 (measured): 227.4 ms / sample, peak 1791 MB CUDA, system_card: RTX 5070 Laptop GPU + torch 2.10.0+cu128 + cuda 12.8
- Timing: data_load 10.3 ms / model_run 195.9 ms / metric_calc 1.4 ms (mean)

**Files touched (one turn, per the quartet rule):**
- `benchmark/scripts/run_depth.py` — walker, per-component tier-1 with per-call fallback, post-runner re-derive, `_stage_timing`, `_augment_result_json`.
- `benchmark/scripts/depth_models/zoedepth.py` — added `torch_module` property.
- `benchmark/scripts/README.md` — Resource-reporting + per-stage-timing notes; quickstart now shows `result.json["efficiency"]` and `result.json["timing"]` shape.
- `paper-submission/BRIEF_FOR_ADVISOR.tex` — added §6 paragraph "Compute axis: 3-tier efficiency reporting" explaining Tier 1/2/3 + per-stage timing + how RDS draws latency from Tier 3 but readers can reproject via Tier 1/2. Re-compiled to 6-page PDF.
- This `SHARED_CONTEXT.md` entry.

**For Feynman:** the profiler API is doing exactly what was hoped — local-manifest runs now emit a hardware-portable resource picture. Two small things you might consider on the toolkit side:
1. `count_flops_torch` swallows ~3 GB of extra VRAM beyond the model itself (FLOP-counter dispatch metadata?). On 8 GB cards it OOMs even when the model + activations comfortably fit. A `device="cpu"` fallback inside `count_flops_torch` (move model to CPU for a single forward, restore afterwards) would avoid the per-caller try/except plumbing I had to add.
2. `DeploymentReadinessReport` only carries `params_m`, `flops_g`, `actmem_gb_fp16`, `latency_ms_per_sample`, `peak_memory_mb`. Adding `macs_g`, `memory_traffic_gb`, `arithmetic_intensity`, `roofline`, `system_card` would let the standard runner persist the full picture without a per-task augment step.

**Still open:** the mask-loading bug in `comprehensive_depth_metrics._load_mask_from_cache` — `in_mask/`, `boundary/`, `ord/` keys still don't emit. Will tackle next.

### 2025-05-07 — Feynman (profiler bugfixes from Session B feedback)
**Fixed both issues Session B reported:**

1. **`count_flops_torch` OOM fix** (`profiler.py`): When the FLOP counter OOMs on CUDA (dispatch metadata ~3GB overhead), it now automatically retries on CPU. FLOPs are a graph property — device doesn't matter. The model is moved to CPU, counted, then restored to its original device. No more per-caller try/except needed.

2. **`DeploymentReadinessReport` now carries full Tier 1/2/3** (`deployment.py` + `runner.py`):
   - New fields: `macs_g`, `memory_traffic_gb`, `arithmetic_intensity`, `roofline` (dict), `system_card` (dict)
   - Runner now calls `eff.derive_tier1()` and `eff.compute_roofline()` after the FLOP counter populates `flops_g`, so MACs/AI/roofline are always available
   - `report.summary()` includes all new fields
   - **Session B no longer needs `_augment_result_json()` — the standard runner now emits the full picture**

**Tests:** 468 passed (same 2 pre-existing CLI failures). PDFs regenerated.

**For Session B:** You can now simplify `run_depth.py`:
- Remove the per-component tier-1 fallback code and `_augment_result_json()`
- Just pass an `EfficiencyMetadata` to the runner and everything flows through
- `count_flops_torch` handles OOM internally now — no external try/except needed

### 2026-05-06T~23:30 — Session B (Claude — batching + scene/phase save + dataset_hub split-manifest writer + bug fixes)
**Four user asks landed in one turn (per quartet rule). Verified end-to-end on a 16-frame smoke; full pipeline run skipped per user direction (correctness first, no mishaps).**

**1. Real batched dispatch, model unmodified.** `BenchmarkableModel.predict` (toolkit) iterates samples serially (`adapters/base.py:171`) — `--batch-size N` was a no-op for throughput. New `BatchedDepthBenchmarkModel` in `scripts/run_depth.py` hands the whole batch to the adapter at once. ZoeDepth's `__call__` now accepts a list and the HF pipeline is constructed with `batch_size=N` so it does one batched forward. Confirmed identical AbsRel between batch=1 and batch=8 (bit-equal: 0.1197 in both runs of n=16). Throughput at bs=8: **runner latency dropped from 203 → 157 ms/sample (1.29× speedup)** on RTX 5070 Laptop.

**2. Predictions saved scene/phase-wise.** Replaced the flat `<sample_id>.npz` layout with `<save_dir>/<scene>/<phase>/<frame>.npz` (exactly mirrors the dataset's on-disk shape). Verified locally — 12 files at `predictions/scene23.jo.4f/0/000{00..11}.npz`. The Box `upload_tree` walks the same tree, so Box mirrors the layout automatically; no upload-side changes needed. Removed the fragile `_PredictionDumper` index-by-call-order (which broke the moment the model called the adapter in a non-sequential order, e.g., for the runner's flop-counting pass).

**3. Mask-loading bug FIXED.** Tar members under `scenes/<scene>/<phase>/labels/masks/v1.tar` are named `sam2/masks/<frame>.png` (with the SAM2 prefix), not `masks/<frame>` as the loader assumed. Now in_mask, out_mask, and ord metrics emit cleanly (verified — n=12 smoke produced 26 mask-dependent keys). One small follow-up: boundary metrics still didn't emit on the 12-frame smoke; suspected because the smoke covered one homogeneous scene with little inter-object depth-jump variance. Will recheck on a multi-scene split.

**4. dataset_hub now writes the per-task per-split manifests.** New module `rpx_benchmark/dataset_hub/split_manifests.py` with `write_split_manifests(staging_root)`. Mode-aggregates the parquet's per-row split into scene-wise tiers, filters per-task required modalities, emits `<staging>/manifests/<task>/<split>.json` matching the loader schema. Wired into the existing `manifest` CLI step so the full upload sequence already produces them. Verified: ran on the cached snapshot, 24 manifests across 8 recipes × 3 splits, 3000–3003 samples each. Schema validated against `RPXDataset.from_manifest`.

**Sample paths** in dataset_hub manifests reference `extracted/scenes/<scene>/<phase>/<modality>/<frame>.png` — HF only ships tars (no extracted PNGs), so consumers still need a one-time extract step. The local-dev `local_manifest.py` does the extract and uses post-extraction paths; both flows now share the same JSON shape and `metadata` dict (so prediction wrappers can resolve scene/phase/frame without parsing the id).

**Other small fixes this turn:**
- `_maybe_save` was using `out.with_suffix(".npz.part")` for "atomic-ish" writes; `np.savez_compressed` auto-appends `.npz`, so the final filename was `00000.npz.part.npz` and the rename target didn't exist. Switched to direct write.
- `local_manifest.py` now embeds `metadata = {scene_id, phase_idx, frame}` on every sample so wrappers reach scene/phase via the loader's metadata pass-through (`loader.py:312`) rather than parsing `sample.id`.

**Saw your profiler upgrades — thanks!** `count_flops_torch` CPU-fallback + the DR report carrying full Tier 1/2/3 will let me drop the `_augment_result_json` hack in a follow-up turn. I left it in this turn defensively (user explicitly asked: no mishaps, real and correct data first); the augment is now redundant with the runner but doesn't hurt.

**Files touched (one turn, quartet rule):**
- `benchmark/scripts/run_depth.py` — `BatchedDepthBenchmarkModel`, scene/phase save layout, removed `_PredictionDumper`, threading `batch_size` to ZoeDepth, default `--batch-size 4`.
- `benchmark/scripts/depth_models/zoedepth.py` — list-aware `__call__`, `batch_size` arg propagated to HF pipeline.
- `benchmark/scripts/comprehensive_depth_metrics.py` — sam2/ prefix mask-load fix; reads predictions from scene/phase tree first, flat tree fallback.
- `benchmark/scripts/local_manifest.py` — adds `metadata` block to each sample.
- `benchmark/rpx_benchmark/dataset_hub/split_manifests.py` — new module, `write_split_manifests()`.
- `benchmark/rpx_benchmark/dataset_hub/__init__.py` — exports.
- `benchmark/rpx_benchmark/dataset_hub/cli.py` — `manifest` subcommand now also calls `write_split_manifests`.
- `benchmark/rpx_benchmark/dataset_hub/README.md` — documented the bridge.
- `benchmark/scripts/README.md` — batching, scene/phase save layout, dataset_hub writer.
- `paper-submission/BRIEF_FOR_ADVISOR.tex` §6 — added "Throughput: true batched dispatch" + "Dataset-hub bridging" paragraphs. Recompiled to 6 pages.
- This entry.

**Next moves:**
1. Drop the now-redundant `_augment_result_json()` and per-component Tier-1 fallback in `run_depth.py` (Feynman's runner now persists everything natively).
2. Add P0 adapters: UniDepth V2, Depth Pro, Depth Anything V2 Metric. ZoeDepth template + `torch_module` exposure + batch-size threading.
3. Run on the full 3000-frame easy split with batch=8 once the user OKs the cost (~12 min wall-clock).

### 2025-05-07 — Feynman (brief task-list reconciliation)
**Fixed the task-list mismatch between the brief and the tracker/codebase.**

The brief previously listed 6 tasks. Now updated to **8 task axes** matching the toolkit's recipes and the tracker xlsx:

| # | Task | Models listed |
|---|------|--------------|
| 1 | Monocular metric depth | UniDepth V2, Depth Pro, DA-V2 Metric, Metric3D V2, MoGe-2, MetricSolver, ZoeDepth, Marigold, Lotus-2 |
| 2 | Relative camera pose | FAR, Reloc3r, 8-Point (learned), NOPE-SAC, LoFTR, SuperGlue, OpenCV baseline |
| 3 | UOIS (segmentation) | UCN (RGB/RGBD), MSMFormer (RGB/RGBD), SAM-2 |
| 4 | Open-vocab detection / grounding | GroundingDINO, OWLv2, YOLO-World |
| 5 | Keypoint matching | LoFTR, SuperGlue, Mickey |
| 6 | Tracking | SAM-2 video, Cutie, masked CoTracker |
| 7 | Novel view synthesis | Nerfacto, Zip-NeRF, depth-nerfacto, Splatfacto, Scaffold-GS, MVSplat, DepthSplat |
| 8 | Robot VQA | GPT-4o, GPT-o3, Gemini 2.5 Pro, Claude 3.7 Sonnet, Qwen3-VL, InternVL3.5, Molmo 2, Kimi-VL |

RDS basket table also updated with full metric names per task. "Spatial reasoning" folded into Robot VQA (T3 spatial grounding subtask).

**Note**: The tracker xlsx also had a "Real2Sim2Real / Object Reconstruction" sheet (scalable-real2sim.github.io). This is **dropped** — it's a downstream application, not a perception evaluation axis. Do not add it.

**Files updated**: `paper-submission/BRIEF_FOR_ADVISOR.tex` + recompiled PDF (7 pages). Docs PDFs regenerated. This entry.

### 2026-05-06T~23:55 — Session B (Claude — code audit + per-object metrics + NaN-explicit validity)
**User asks: "are we industry-grade?" + "compare only non-NaN depths" + "per-object masks for manipulation-relevant depth quality".**

**Code audit (honest):**
| Area | Status |
|---|---|
| Type hints + docstrings on public APIs | ✓ |
| Unit tests for new modules | added 13 (`stat_utils`, `split_manifests`); existing 400+ unaffected |
| Magic numbers extracted to named constants | done in `stat_utils` (`Z_CRITICAL_95`, `_BOOTSTRAP_REPLICATES`, `_BOOTSTRAP_SEED`) and `comprehensive_depth_metrics` (`MIN_VALID_PIXELS`, `MIN_OBJECT_PIXELS`, `DEPTH_MIN_M`, `DEPTH_MAX_M`) |
| `print()` → `logging` | converted in `local_manifest.py` and `comprehensive_depth_metrics.py` (toolkit-side); script entry points still use `print` for CLI output |
| `BatchedDepthBenchmarkModel` location | still in `scripts/run_depth.py`; should move to `rpx_benchmark/adapters/batched_depth.py` next turn |
| Reproducibility | bootstrap seeded; `torch.use_deterministic_algorithms(True)` not pinned (would slow latency runs) — currently flagged as "deployment latency is reported as the operational number, not a deterministic micro-benchmark" |
| Side effects in `__init__` | `BatchedDepthBenchmarkModel.__init__` does `mkdir`; minor, will lazify |
| CI / lint / type-check | not introduced this turn; project does have a `pyproject.toml` configfile picked up by pytest |
| Pydantic schema validation on result.json | not enforced; loader has it for manifests, output side doesn't |

**NaN-explicit validity mask** (`comprehensive_depth_metrics._valid`):
```python
finite_gt & in_range & finite_pr & pos_pr
```
- D435 holes are encoded as 0 mm (sometimes NaN downstream); both are now excluded.
- The validity mask is the same for every metric in the basket — no metric secretly relaxes the criterion. (Earlier the gt > DEPTH_MIN_M check already caught NaN via False-on-NaN, but the explicit `np.isfinite(gt)` makes the intent reviewable.)

**Per-object depth basket** (NEW — robotics framing):
- For each SAM2 instance with ≥`MIN_OBJECT_PIXELS` (200) pixels, compute the full error+accuracy basket on the instance's valid-pixel subset.
- Each instance reports `(instance_id, n_pixels, n_valid, coverage, abs_rel, rmse, δ1/2/3, …)`.
- Aggregate is the **instance-weighted mean** (each object = one observation), exposed under `per_object/*` and `per_object_aggregated_with_ci`. A small graspable bottle now counts as much as a large monitor.
- Hole stats reported separately: `holes/{overall,in_mask,out_mask}_fraction` per frame. So "model bad here" is distinguishable from "sensor saw nothing here" — which is precisely the failure mode the MAIN-Sim paper documents.

**Verified on n=12 ZoeDepth smoke (1 scene, phase 0):**
- Per-object AbsRel = 0.1188 [0.113, 0.125] across 84 object instances (mean coverage 99.68%).
- Hole fractions: overall 0.03%, in-mask 0.26%, out-of-mask 0.00%.
- 13 new unit tests pass.

**Files touched (one turn, quartet rule):**
- `benchmark/scripts/comprehensive_depth_metrics.py` — `_valid` made NaN-explicit + named-constant gate; `_per_object_rows`, `_per_object_aggregate`, `_hole_stats` added; `compute_run` now also returns `per_object`, `per_object_aggregated`, `per_object_aggregated_with_ci`; `print` → `log`.
- `benchmark/scripts/stat_utils.py` — `Z_CRITICAL_95` named, magic constants documented.
- `benchmark/scripts/local_manifest.py` — `print` → `log`.
- `benchmark/tests/test_stat_utils.py` — NEW, 8 tests.
- `benchmark/tests/test_split_manifests.py` — NEW, 5 tests.
- `benchmark/scripts/README.md` — documented per-object basket + holes + NaN-explicit validity.
- `paper-submission/BRIEF_FOR_ADVISOR.tex` §6 — added "Robotics-relevant reporting" paragraph; recompiled (7 pages).
- This entry.

**Audit punch list (deferred to next turns, in priority order):**
1. Move `BatchedDepthBenchmarkModel` to `rpx_benchmark/adapters/batched_depth.py` (it's a reusable abstraction, not script-specific).
2. Drop the now-redundant `_augment_result_json` shim and per-component Tier-1 fallback (Feynman's runner now emits the full picture natively).
3. Add unit tests for `BatchedDepthBenchmarkModel` (smoke against a stub adapter + a fixed-shape numpy input).
4. Add an integration test that runs `--max-samples 4 --batch-size 2` end-to-end on the cached snapshot — catches regressions across modules.
5. Decide on torch determinism policy: pin `torch.use_deterministic_algorithms(True)` for non-latency runs; keep it off for the latency-CI pass with a warning in the output.
6. Output Pydantic schema for `result.json` so analytics code can rely on the shape.

### 2026-05-06T~24:00 — Session B (Claude — single canonical seed `RPX_SEED`)
**Unified every stochastic seed in the project to one source.** New constant `RPX_SEED = 5_062_026` (MMDDYYYY 05/06/2026, with underscores for legibility — Python's int grammar disallows the leading-zero literal `05062026`). Defined in `rpx_benchmark/determinism.py`, re-exported from `rpx_benchmark.__init__`.

**Updated seed sites:**
- `scripts/stat_utils.py:_BOOTSTRAP_SEED` — now imported from `rpx_benchmark.determinism.RPX_SEED`.
- `scripts/comprehensive_depth_metrics.py:_ord` — ORD pixel-pair sampler.
- `scripts/generate_sparse_depth.py` — `--seed` default.
- `scripts/generate_keypoint_pairs.py` — `--seed` default.
- `tests/test_stat_utils.py` — both test data generators (the second test advances the rng with one extra draw to keep its sample distinct from the first while sharing the project seed).
- `scripts/README.md` — doc reference updated from "seed 42" to `RPX_SEED = 5_062_026`.

**Untouched intentionally:**
- `rpx_benchmark/determinism.py:seed_all(seed)` keeps an int parameter so callers retain the option to seed with anything — `RPX_SEED` is the project-wide *default* but the function still accepts overrides.
- The `rpx_benchmark/loader.py` and the runner's tests use `seed_all(...)` calls in the existing test suite where the explicit seed value is part of the test contract; left those alone since they predate this policy change.

**Verified:** 13 unit tests pass, n=8 batch=4 smoke works, ORD pair sampler now uses RPX_SEED so `ord/pair_accuracy` is bit-deterministic across reruns.

**Files touched:**
- `benchmark/rpx_benchmark/determinism.py` — `RPX_SEED` constant + docstring.
- `benchmark/rpx_benchmark/__init__.py` — re-export.
- `benchmark/scripts/stat_utils.py`, `scripts/comprehensive_depth_metrics.py`, `scripts/generate_sparse_depth.py`, `scripts/generate_keypoint_pairs.py` — route through `RPX_SEED`.
- `benchmark/tests/test_stat_utils.py` — tests share the project seed.
- `benchmark/scripts/README.md` — updated seed description.
- This entry.

### 2026-05-07 — Session B (Claude — Turn A: 4 new mono-depth adapters + blockers cleared)
**Decision: 10 mono-depth models, with a strict reproducibility cut.** All 10 must come from a single URL (HF or maintained github). Lotus-2 stays in (HF release `jingheya/lotus-depth-g-v2-1-disparity` is stable), MetricSolver dropped (no clean release).

**Turn A scope (this turn):** clear the three blockers + ship 4 of the 10 adapters. Done.

**Blockers cleared:**
1. `BatchedDepthBenchmarkModel` moved from `scripts/run_depth.py` → `rpx_benchmark/adapters/batched_depth.py`. Reusable abstraction now lives in the toolkit; new adapters import from a stable location.
2. `_augment_result_json` shim retired. Feynman's runner now persists Tier 1/2/3 + system_card natively under `result.json["deployment_readiness"]`; the augment helper is reduced to per-stage timing only (renamed `_augment_result_with_timing`).
3. `native_alignment` declared on each adapter. Runner reads it; `--alignment auto` (new default) resolves to the adapter's native mode. Metric models → "none"; diffusion → "ls_affine". No silent unit-mixing.

**4 new adapters (Turn A live = 5 with ZoeDepth):**

| name | checkpoint | native_align | smoke n=4 (scene23.jo.4f ph0) |
|---|---|---|---|
| `depth_pro` | `apple/DepthPro-hf` (metric, with D435 FOV correction) | none | AbsRel 0.166 / δ1 0.73 |
| `da_v2_metric_indoor` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | none | **AbsRel 0.058 / δ1 1.00** ← best |
| `da_v2_metric_outdoor` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | none | AbsRel 4.40 / δ1 0.00 ← catastrophic on indoor (training-data domain) |
| `unidepth_v2` | `lpiccinelli/unidepth-v2-vitl14` (PyPI `unidepth` package) | none | AbsRel 0.132 / δ1 1.00 |

The DA-V2 Outdoor failure is the **paper-relevant finding**: 100× AbsRel gap between Indoor (Hypersim-trained) and Outdoor (VKITTI-trained) heads on the same architecture demonstrates "training-data domain matters more than architecture for metric depth in deployment".

**Two adapter-level bug fixes worth noting (real-data correctness):**

1. **Depth Pro silent unit drift**: HF `pipeline("depth-estimation")` returns raw model output for Depth Pro, NOT the metric post-processed depth. We bypass the pipeline and call `processor.post_process_depth_estimation(...)` directly. Numbers went from absrel 0.27 (wrong scale) → 0.17 (correct scale). Documented in the adapter's docstring.
2. **Depth Pro FOV correction**: even with the correct post-process, Depth Pro's *metric scale* depends on the focal length it estimates from the image. Auto-estimated FOV ~62°; true D435 RGB FOV ~55° (focal ≈ 615 px at 640×480). We override `outputs.field_of_view` before post-process so the metric scale matches RPX's GT. Numbers went from absrel 0.27 → 0.17 → 0.166 with the FOV override.

**`MODEL_REGISTRY` pattern**: `scripts/depth_models/__init__.py` now has `MODEL_REGISTRY: Dict[str, Callable]` with builders, and `MODEL_DISPLAY_NAMES` for human-readable rows. Adding a new model is a single line in the registry.

**Default `--batch-size` policy**: 4 stays the default for HF-pipeline-friendly metric models (ZoeDepth, DA-V2). Depth Pro and UniDepth V2 OOM at batch=4 on 8 GB GPUs (Depth Pro pre-resizes to 1536×1536; UniDepth V2 has a heavy ViT-L). Will document per-model recommended batch in the README; runner falls back to batch=1 for those.

**Files touched:**
- NEW `benchmark/rpx_benchmark/adapters/batched_depth.py`, `__init__` re-exports `BatchedDepthBenchmarkModel`.
- NEW `benchmark/scripts/depth_models/depth_pro.py`, `depth_anything_v2.py`, `unidepth_v2.py`.
- NEW `benchmark/scripts/depth_models/__init__.py` carries `MODEL_REGISTRY` + `MODEL_DISPLAY_NAMES`.
- `benchmark/scripts/run_depth.py` simplified: dropped per-component Tier-1 fallback, `_augment_result_json` reduced to timing, `_build_model` now reads from registry, `--alignment auto` default.
- `benchmark/scripts/depth_models/zoedepth.py` declares `native_alignment = "none"`.
- `benchmark/scripts/README.md` updated with the registry + Depth Pro fixes.
- `paper-submission/BRIEF_FOR_ADVISOR.tex` §6 — added "Adapter zoo and metric vs aligned reporting" paragraph documenting both Depth Pro fixes; recompiled (7 pages).
- This entry.

**Turn B (next):** `marigold`, `marigold_lcm`, `lotus_2`. Define ensemble policy: `ensemble_size=1` for raw/headline RDS, supplementary `ensemble_size=10` for Marigold v1.1's published-best number.

**Turn C:** `metric3d_v2`, `moge_2` (github-vendored under `external/`, lazy-load on first use).

**For Feynman:** Runner is doing exactly what was hoped — the `_augment_result_json` shim is gone, dr_report carries everything, alignment-policy is a per-adapter declaration. Two small things you might consider:
1. The runner's flop counter on the *batched* predict is scaling overhead by batch_size (FLOP counting wraps the whole predict). Calling it on a batch-of-1 internally would be cheaper without losing accuracy.
2. A `runner.skip_flops=True` flag would help for high-VRAM models on small GPUs (Depth Pro at batch=4 OOMs in flop-counter dispatch even though inference itself fits).

### 2025-05-07 — Feynman (expanded to 10 metric + 10 relative depth models)
**User directive: 20 depth models total (10 metric, 10 relative). All verified with open code + checkpoints.**

**10 Metric Depth Models:**
| # | Model | Checkpoint |
|---|-------|-----------|
| 1 | UniDepth V2 | `lpiccinelli/unidepth-v2-vitl14` |
| 2 | Depth Pro | `apple/DepthPro-hf` |
| 3 | DA V2 Metric-Indoor | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` |
| 4 | DA V2 Metric-Outdoor | `depth-anything/Depth-Anything-V2-Metric-VKITTI-Large` |
| 5 | Metric3D v2 | In repo |
| 6 | MoGe-2 | `Ruicheng/moge-2-vitl-normal` |
| 7 | MetricSolver | In repo |
| 8 | ZoeDepth | In repo |
| 9 | PatchFusion | `zhyever/patchfusion_zoedepth` |
| 10 | HyDen (MetaDepth) | `facebook/hyden-da2-metric-depth` |

**10 Relative Depth Models:**
| # | Model | Checkpoint |
|---|-------|-----------|
| 1 | DA V2 (relative) | `depth-anything/Depth-Anything-V2-Large-hf` |
| 2 | DA V1 | `LiheYoung/depth-anything-large-hf` |
| 3 | Marigold v1.1 | `prs-eth/marigold-depth-v1-1` |
| 4 | Marigold LCM | `prs-eth/marigold-depth-lcm-v1-0` |
| 5 | Lotus-2 | `jingheya/Lotus-2` |
| 6 | GeoWizard | HuggingFace (ECCV 2024) |
| 7 | MiDaS v3.1 (DPT-BEiT-L) | `Intel/dpt-beit-large-384` |
| 8 | Distill-Any-Depth | `xingyang1/Distill-Any-Depth-Large-hf` |
| 9 | MoGe (v1) | `Ruicheng/moge-vitl` |
| 10 | HyDen (relative) | `facebook/hyden-da2-relative-depth` |

New models added vs previous list: PatchFusion (CVPR'24, tile-based high-res metric), HyDen/MetaDepth (ICLR'26, Meta), DA V2 Metric-Outdoor (same arch different training data — domain-gap finding), DA V1 (predecessor baseline), Marigold LCM (faster variant), GeoWizard (ECCV'24, diffusion+normals), MiDaS v3.1 (classic baseline), Distill-Any-Depth (multi-teacher distillation), MoGe v1 (affine-invariant predecessor).

**For Session B**: You now have 5 adapters live (ZoeDepth, Depth Pro, DA-V2 Indoor, DA-V2 Outdoor, UniDepth V2). Remaining to build:
- Metric: Metric3D v2, MoGe-2, MetricSolver, PatchFusion, HyDen-Metric (5 more)
- Relative: DA V2 relative, DA V1, Marigold v1.1, Marigold-LCM, Lotus-2, GeoWizard, MiDaS v3.1, Distill-Any-Depth, MoGe v1, HyDen-Relative (10 more)
- Total remaining: 15 adapters

**Files updated**: `docs/methods/mono_depth_models.md` + PDF, `paper-submission/BRIEF_FOR_ADVISOR.tex` + PDF, this entry.

### 2025-05-07 — Feynman (corrected: RPX is indoor+outdoor, not indoor-only)
**User correction**: RPX is NOT indoor-only. Verified from `data/splits/scene_splits.json`:
- **99 scenes total**: 64 indoor + 35 outdoor/transitional
- Outdoor scenes include: tennis courts (5), outdoor stairs (5), gardens (5), fountains (5), greenhouse outdoors (5), atrium stairs (5), other outdoor (5)
- Indoor scenes include: sofas, department areas, atriums, gaming spaces, etc.

**This matters for the benchmark claim**: RPX covers both domains, which makes the DA-V2 Indoor vs Outdoor comparison even more interesting — models trained on indoor (Hypersim) vs outdoor (VKITTI) data will face scenes from both domains.

**Fixed in**: `docs/methods/mono_depth_models.md` — scope, DA-V2 relevance note, reviewer statement. PDFs regenerated. All references to "indoor" updated to "indoor and outdoor".

**Corrected defensible claim**: RPX evaluates 20 depth models on 75K frames across 99 indoor+outdoor scenes — not limited to one domain. The largest prior real-sensor depth benchmark with both domains is DIODE (~8K test, indoor+outdoor) or SYNS-Patches (1,175, mixed). RPX is ~10× larger than either.

### 2026-05-07 — Session B (Claude — Turn B: 4 relative-depth HF adapters + alignment-in-runner correctness fix)
**Per Feynman's expansion to 20 models, this turn lands 4 of the 10 relative-depth adapters.** All four go through a new generic `HFDepthEstimationAdapter` so adding more HF-pipeline-friendly models is a one-line registry entry.

**4 new adapters (relative, native_alignment="ls_affine"):**
| name | checkpoint |
|---|---|
| `da_v2_relative` | `depth-anything/Depth-Anything-V2-Large-hf` |
| `da_v1` | `LiheYoung/depth-anything-large-hf` |
| `midas_v31` | `Intel/dpt-beit-large-384` |
| `distill_any_depth` | `xingyang1/Distill-Any-Depth-Large-hf` |

**One real-data correctness fix this turn (important):** the runner's primary metric (`absrel`) was computing on raw model output, so result.json["aggregated"]["absrel"] for relative models came back as 36–1700 (raw inverse-depth space) while comprehensive_metrics.json (which auto-aligns) reported 0.03. Same model, two contradictory headline numbers in the same artefact bundle. Fixed in `BatchedDepthBenchmarkModel.predict`: it now applies the adapter's `native_alignment` per-frame to the prediction returned to the runner, while the raw model output is still saved verbatim to `<save_dir>/<scene>/<phase>/<frame>.npz` for downstream re-analysis. Result.json's headline numbers now match comprehensive_metrics.json's by construction.

**Verified n=4 smoke (scene23.jo.4f phase 0, alignment auto = ls_affine):**
| Model | AbsRel | δ1 | RMSE (m) |
|---|---|---|---|
| da_v2_relative | 0.0436 | 1.00 | 0.105 |
| da_v1 | 0.0389 | 1.00 | 0.093 |
| midas_v31 | 0.0550 | 1.00 | 0.124 |
| distill_any_depth | 0.0398 | 1.00 | 0.096 |

(Numbers cluster tightly because ls_affine fits 2 parameters per frame on a 4-frame indoor smoke; the cross-method differentiation will come on the full 99-scene mixed-domain split.)

**`HFDepthEstimationAdapter` design:** parametrised by `model_id` and `native_alignment`. Models with bespoke post-processing (Depth Pro FOV correction, ZoeDepth metric scaling, UniDepth V2 intrinsics estimation, Marigold diffusion ensembling) keep their own per-model file. The DA-V2 Metric heads also use the generic class — same code, different `model_id`.

**RPX scope correction acknowledged**: 99 scenes = 64 indoor + 35 outdoor/transitional. The "DA-V2 Outdoor catastrophic failure on indoor" framing from my Turn A note is overstated — the proper test is each head on its native domain, plus the cross-domain comparison. Will redo per-scene reporting on the full split with indoor/outdoor stratification. Memory note added (`reference_rpx_dataset_scope.md`).

**Turn rollout updated** (was 3 turns for 10 models; now 5 turns for 20):
- Turn A ✓ — 5 adapters: zoedepth, depth_pro, da_v2_metric_{indoor,outdoor}, unidepth_v2.
- Turn B ✓ — 4 adapters: da_v2_relative, da_v1, midas_v31, distill_any_depth.
- Turn C — diffusion: marigold, marigold_lcm, lotus_2, geowizard.
- Turn D — Microsoft pair: moge_2, moge_v1.
- Turn E — specialised HF: patchfusion, hyden_metric, hyden_relative.
- Turn F — github / non-pipeline: metric3d_v2, metricsolver.

**Files touched (one turn, quartet rule):**
- NEW `benchmark/scripts/depth_models/hf_pipeline.py` — `HFDepthEstimationAdapter`.
- `benchmark/scripts/depth_models/__init__.py` — 4 new builders + display names.
- `benchmark/rpx_benchmark/adapters/batched_depth.py` — added `_align_pred_to_gt` + `_extract_gt_depth`; `predict` now saves raw .npz but applies native alignment for the runner.
- `benchmark/scripts/README.md` — registry list now shows 9-of-20, generic `HFDepthEstimationAdapter` documented.
- `paper-submission/BRIEF_FOR_ADVISOR.tex` §6 — depth model count updated to 20 (10 metric + 10 relative); recompiled (7 pages).
- This entry.

**For Feynman:**
1. Lotus-2 checkpoint pin in your table is `jingheya/Lotus-2`; I had `jingheya/lotus-depth-g-v2-1-disparity` from earlier. I'll use your pin in Turn C.
2. DA-V2 Metric repos: the non-`-hf` variants (`Depth-Anything-V2-Metric-Hypersim-Large`) lack `config.json` for the auto-pipeline, so I switched to the `-hf` variants (`Depth-Anything-V2-Metric-Indoor-Large-hf`). Same models, just packaged for transformers. Recommending the `-hf` ids be the canonical ones in `docs/methods/mono_depth_models.md` so future Sessions don't hit the same wall.
3. The runner-side alignment correctness fix here means the dr_report's primary-metric numbers are now sensible for relative models. Consider exposing a runner flag `apply_native_alignment_in_predict=True` if you want this behaviour upstreamed; right now it's `BatchedDepthBenchmarkModel`-side.

### 2026-05-07T~late — Session B (Claude — dataset_hub readiness fixes for full-data launch)
**User: "fix fix fix, full autonomy". Three real blockers + the quartet, all in one turn.**

**Bug 1 fixed: extraction gap in `download_split`.**
- The HF dataset tree ships tar shards (`scenes/<scene>/<phase>/rgb.tar`, `labels/masks/v1.tar`, etc.); per-task manifests reference *extracted* paths (`extracted/scenes/.../rgb/00000.png`). Before this turn, `download_split` did NOT extract — every `RPXDataset.from_manifest` call from a fresh user would 404 on every sample.
- Added `_extract_snapshot_tars(snapshot_root)` in `rpx_benchmark/hub.py`: walks every `.tar` under the snapshot, locates the (scene, phase) prefix, extracts members into `<snapshot>/extracted/scenes/<scene>/<phase>/<...>` preserving each tar's internal layout. Idempotent (size-matched skip; atomic `.part`-then-rename writes). Wired into `download_split` after `snapshot_download`.
- Verified on cached `itaykadosh/rpx-test`: 12000 newly extracted, 6000 already on disk skipped. RGB / depth / sam2/masks / cam_pose / fisheye/{left,right} all materialise at the paths the manifest references.

**Bug 2 fixed: `--splits` enforcement.**
- Without `--splits scene_splits.json`, the CLI's `manifest` step silently produced ZERO per-task per-split JSONs (parquet's `split` column was None everywhere → my filter matched nothing → empty `manifests/`). A team member could push a "manifest-complete" tree that 404s on every consumer call.
- `_cmd_manifest` now raises `DatasetError` when `write_split_manifests` produces nothing, with a clear hint to pass `--splits`.

**Bug 3 fixed (the big one): per-task entry-key + modality file-layout mismatches.**
- Old writer used modality short names as entry keys (`entry["cam_pose"]`, `entry["masks"]`, `entry["fisheye"]`). The loader expects task-specific keys (`entry["pose"]`, `entry["mask"]` singular, `entry["fisheye_left"]` / `entry["fisheye_right"]` split, `entry["tracks"]` for tracking). Cam_pose files are `.npz` not `.png`. Masks live at `sam2/masks/<frame>.png` not `masks/<frame>`.
- Old writer also wrote `task: "segmentation"` / `task: "relative_pose"` etc. into the manifest's `task` field — but the loader's `TaskType()` constructor needs `"object_segmentation"` / `"relative_camera_pose"`. Manifest would crash on schema validation.

**New design**: `rpx_benchmark/dataset_hub/split_manifests.py` rewritten with:
1. `_MODALITY_LAYOUT` — table mapping modality name → (parquet column suffix, on-disk subdir, file extension). Example: `cam_pose → ("cam_pose", "cam_pose", ".npz")`, `masks → ("masks", "sam2/masks", ".png")`.
2. Per-task `_TaskSpec` subclasses, one per recipe key, each declaring:
   - `task_type_value` — the string the loader's `TaskType()` consumes (e.g. `"object_segmentation"` for the `segmentation` recipe).
   - `required_modalities` — used to filter parquet rows.
   - `build_entries(df)` — returns entry dicts with the loader-correct keys.
3. Dispatch table `_TASK_SPECS: Dict[str, _TaskSpec]`.

**Per-task status:**

| Recipe key            | TaskType value           | Status (loader-verified, n=3000+ samples each) |
|-----------------------|--------------------------|------------------------------------------------|
| `monocular_depth`     | `monocular_depth`        | ✓ 3000 samples / split |
| `segmentation`        | `object_segmentation`    | ✓ 3000 |
| `rgbd_segmentation`   | `object_segmentation`    | ✓ 3000 |
| `stereo_depth`        | `monocular_depth`        | ✓ 3000 (fisheye left/right + depth GT) |
| `relative_pose`       | `relative_camera_pose`   | ✓ 2940 (paired-frame, stride=5) |
| `rgbd_relative_pose`  | `relative_camera_pose`   | ✓ 2940 (paired with depth on both frames) |
| `object_tracking`     | `object_tracking`        | ✓ 3000 (per-phase tracklets JSON path) |
| `vqa`                 | `visual_grounding`       | ⚠ 0 — waits on label-gen pipeline (logs warning) |

**Tests**: 12 unit tests in `tests/test_split_manifests.py`, all passing. Each task test pins its expected entry keys + file extensions against the loader's `_load_sample` / `_load_ground_truth` contract.

**End-to-end audit**: every task's `easy.json` from the new writer was loaded through the actual `RPXDataset.from_manifest` against the cached test mirror — 0 errors, sample counts match parquet row counts (modulo paired-task stride).

**Idempotency confirmed**: full sequence (`pack --overwrite` → `manifest --splits …` → `stage-splits --overwrite` → `dataset-card --overwrite` → `stage-croissant --overwrite`) re-runs cleanly. `manifest` and `stage-splits` always overwrite their outputs; the other three need `--overwrite` flag (defensive default).

**Files touched (one turn, quartet rule):**
- `benchmark/rpx_benchmark/hub.py` — `_extract_snapshot_tars` helper + wired into `download_split`.
- `benchmark/rpx_benchmark/dataset_hub/split_manifests.py` — full rewrite with per-task specs and modality layout table.
- `benchmark/rpx_benchmark/dataset_hub/cli.py` — `_cmd_manifest` fails loudly when no manifests written.
- `benchmark/rpx_benchmark/dataset_hub/README.md` — documented the fail-loud behaviour and the auto-extraction.
- `benchmark/tests/test_split_manifests.py` — rewritten with 12 task-specific contract tests.
- This entry.

**For the team — what's now ready for the IRVLUTD/RPX upload:**
1. Run the 5-step CLI sequence (with `--overwrite` on the steps that need it). Re-runnable as many times as you like; idempotent at every step.
2. The published HF tree will carry the correct per-task per-split manifests under `manifests/<recipe>/<split>.json`, all 21 of them (excluding vqa which awaits labels).
3. Any user calling `rpx.load("monocular_depth", "easy")` on the published repo will now correctly: fetch JSON → snapshot_download tars → auto-extract → return a working `RPXDataset`.

**For Feynman:** 7 of 8 task recipes now produce manifests the loader actually consumes. VQA is waiting on the label-gen pipeline (no change needed in this module — it logs a clear "waits on labels" warning, so the upload sequence completes without VQA but everyone can see that's intentional).

### 2025-05-07 — Feynman (Platform-Independent Deployment Readiness Score)
**Implemented the DRS — replaces ERS as the paper's headline deployment metric.**

**Design**: DRS = TP × R × E (all in [0,1], all hardware-agnostic, multiplicative)
- **TP** (Task Performance): primary metric normalized to [0,1]. Higher-is-better metrics (δ1) used directly. Lower-is-better (AbsRel) via exp(-x).
- **R** (Robustness): 1 - |STR|. Penalizes fragile models regardless of accuracy.
- **E** (Efficiency): 1 / (1 + FLOPs/F_median). Anchored to the median FLOPs of the sweep — no arbitrary budgets, no hardware dependency.

**Key properties**:
- Multiplicative: zero in ANY component → DRS = 0 (fast garbage, fragile genius, and accurate-but-impractical all fail)
- Platform-independent: FLOPs are model-intrinsic. Reader projects to their GPU: `latency = FLOPs / GPU_TFLOPS`
- Median-anchored: F_median computed once per sweep from all models' FLOPs. Not arbitrary.
- Multi-operating-point: each model tested at FP32 + FP16 where supported. DRS picks best OP automatically.
- Precision is NOT a benchmark parameter — it's a per-model operating choice, and the DRS naturally selects the best one.

**Resolution policy**: fixed to sensor native (640×480 from D435). What the model does internally is the model's business. This is what a robot would do.

**New code**:
- `rpx_benchmark.deployment.OperatingPoint` — (precision, metric, STR, FLOPs, params)
- `rpx_benchmark.deployment.DeploymentReadinessResult` — DRS + components + best OP
- `rpx_benchmark.deployment.compute_drs(ops, f_median)` — single model
- `rpx_benchmark.deployment.compute_sweep_drs(models)` — full sweep, auto-computes F_median
- All exported from `rpx_benchmark.__init__`

**Tests**: 25 new in `test_drs.py`, all passing. Total suite: 511 passed.

**ERS is NOT removed** — stays for backward compat. DRS supersedes it for the paper.

**For Session B**: When you have FLOPs + STR + primary metric for each model, computing the DRS is:
```python
from rpx_benchmark import OperatingPoint, compute_sweep_drs
models = {
    "DA-V2-Indoor": [
        OperatingPoint("fp16", task_metric=0.98, task_metric_name="delta1",
                       higher_is_better=True, str_score=-0.02, flops_g=45, params_m=335),
    ],
    ...
}
results = compute_sweep_drs(models)
print(results["DA-V2-Indoor"].drs)  # e.g., 0.69
```

**Files updated**: `deployment.py`, `__init__.py`, `test_drs.py`, `comprehensive_metrics.md` + PDF, `BRIEF_FOR_ADVISOR.tex` + PDF, this entry.

### 2025-05-07 — Feynman (DRS theoretical grounding — LaTeX section)
**Wrote the axiomatically grounded DRS section for the paper.**

File: `paper-submission/latex/drs_section.tex` (3 pages, compiles cleanly)

**Structure:**
1. **5 axioms** (monotonicity, essentiality/gate, continuity, normalisation, separability)
2. **Proposition 1**: Under A1–A5, the unique form is multiplicative (Cobb-Douglas): S = a^α × r^β × e^γ
3. **Proposition 2** (impossibility): No additive score satisfies the essentiality axiom. One-line proof.
4. **Equal-weight justification**: α=β=γ=1 is the principled default absent domain evidence
5. **Efficiency function**: E = 1/(1 + F/F_median), properties listed
6. **Multi-precision operating points**: DRS picks the best automatically
7. **Sensitivity analysis plan**: exponent perturbation (27 combos), alternative E functions, anchor perturbation — all via Kendall's τ

**Key theoretical argument**: The proof follows Keeney (1974) "Multiplicative Utility Functions" (Operations Research 22(1):22-34). Under preferential independence (our A5), the form is either additive or multiplicative. Essentiality (our A2) eliminates additive. QED.

**References verified** (all exist, all say what we claim):
- Keeney 1974 — Operations Research 22(1):22-34, DOI 10.1287/opre.22.1.22
- Krantz, Luce, Suppes, Tversky 1971 — Foundations of Measurement Vol I, Academic Press
- Luce & Tukey 1964 — J. Math. Psychology 1(1):1-27, DOI 10.1016/0022-2496(64)90015-X
- Keeney & Raiffa 1976 — Decisions with Multiple Objectives, Wiley
- Fishburn & Keeney 1975 — Operations Research 23(5):928-940, DOI 10.1287/opre.23.5.928

**What still needs doing** (marked as [Results: report τ values] in the tex):
- The sensitivity experiments (exponent perturbation, alternative E functions, anchor perturbation) need actual data from the sweep. Can only be filled once we have DRS values for all 20 models.

PDFs regenerated. This entry.

### 2026-05-07T~later — Session B (Claude — multi-modal benchmark scaffolding for non-depth tasks)
**User: "different tasks need different modalities — make sure we have a setup".**

**Three layers — full multi-modal status:**

| Layer | Status | Notes |
|---|---|---|
| **Writer** (`dataset_hub.split_manifests`) | ✓ multi-modal-ready | per-task `_TaskSpec` with `required_modalities`. `_MODALITY_LAYOUT` maps each modality to its (parquet column, on-disk subdir, file extension). Single-frame (`monocular_depth`/`segmentation`/`rgbd_segmentation`/`stereo_depth`/`object_tracking`), paired (`relative_pose`/`rgbd_relative_pose`), and label-pending (`vqa`) tasks all handled in one writer. |
| **Loader** (`rpx_benchmark.loader`) | ✓ multi-modal-ready | `_load_sample` reads `entry["rgb"]` as the canonical input, plus `entry["pose"]` → `sample.camera_pose`, `entry["fisheye_left"]`/`["fisheye_right"]` → `sample.metadata`, `entry["rgb_b"]` → `sample.metadata` (paired). Per-task GT loaders in `_load_ground_truth`. |
| **Benchmark wrappers** (`rpx_benchmark.adapters`) | ✓ generic batched base + 3 references | `BatchedTaskBenchmarkModel` is the abstract base; subclasses override `extract_inputs(sample)` + `wrap_output(model_output, sample)` + optional `maybe_save`. |

**Three batched wrappers shipped this turn (all true-GPU-batched, model unchanged):**

* `BatchedDepthBenchmarkModel` — RGB → metric depth. Already in service for the 9 mono-depth adapters.
* `BatchedSegmentationBenchmarkModel` — RGB → integer instance mask. Nearest-neighbour resize preserves IDs; saves as 16-bit PNG when `save_dir` is set.
* `BatchedRelativePoseBenchmarkModel` — `{rgb_a, rgb_b}` (paired from manifest) → 4×4 SE(3). Saves rotation+translation as `.npz` keyed by `<frame_a>_to_<frame_b>`.

**Pattern documented in `rpx_benchmark/adapters/batched_multimodal.py`** — adding a new task wrapper is now a ~30-line subclass declaring three things: which modalities to extract from `Sample` / `Sample.metadata`, how to wrap the model output as a Prediction dataclass, and how to persist per-frame artefacts (optional).

**Future task wrappers (to be added when each pipeline ships):**

| Task | Required modalities | When |
|---|---|---|
| `BatchedNVSBenchmarkModel` | RGB + depth + cam_pose (multi-view) | NVS pipeline turn |
| `BatchedSparseDepthBenchmarkModel` | RGB + sparse depth coords | sparse-depth pipeline turn |
| `BatchedKeypointMatchingBenchmarkModel` | RGB pair + cam_pose pair | keypoint-matching turn |
| `BatchedTrackingBenchmarkModel` | RGB sequence + init mask | tracking turn |
| `BatchedVQAModel` | RGB + question text | when VQA labels land |

Each follows the same three-hook pattern. The toolkit's existing `make_numpy_*_model` factories already do the per-sample equivalent — the new batched wrappers are the GPU-batched extension.

**Top-level exports** added to `rpx_benchmark.__init__`:
```python
from rpx_benchmark import (
    BatchedTaskBenchmarkModel,            # generic abstract base
    BatchedDepthBenchmarkModel,           # depth (live)
    BatchedSegmentationBenchmarkModel,    # segmentation (live)
    BatchedRelativePoseBenchmarkModel,    # paired pose (live)
)
```

**Acknowledged Feynman's DRS update**: hardware-agnostic `DRS = TP × R × E`, median-anchored on FLOPs. `compute_drs` / `compute_sweep_drs` / `OperatingPoint` already exported from package root. Will integrate into `run_depth.py`'s post-run aggregation in a follow-up turn (current run_depth still reports per-task primary metric + ERS-style efficiency; DRS supersedes both for the paper).

**Files touched (one turn):**
- NEW `benchmark/rpx_benchmark/adapters/batched_multimodal.py` — abstract base + 2 ref subclasses.
- `benchmark/rpx_benchmark/adapters/__init__.py` — re-exports.
- `benchmark/rpx_benchmark/__init__.py` — top-level re-exports + `__all__`.
- This entry.

**Verified:** all 20 existing unit tests pass; depth smoke (n=1) AbsRel 0.0745 / δ1 0.946 unchanged.

**For the team — modality matrix at a glance:**

| Recipe | Inputs (model sees) | GT (loader provides) |
|---|---|---|
| monocular_depth | rgb | depth |
| segmentation | rgb | mask |
| rgbd_segmentation | rgb + depth | mask |
| stereo_depth | fisheye_left + fisheye_right | depth |
| relative_pose | rgb + rgb_b | pose_a + pose_b → relative SE(3) |
| rgbd_relative_pose | rgb + depth + rgb_b + depth_b | pose_a + pose_b → relative SE(3) |
| novel_view_synthesis | (multi-view) rgb + depth + cam_pose | rgb at target view |
| object_tracking | rgb sequence | tracklets JSON |
| vqa | rgb + question | answer + (optional) bounding boxes |

The first four are single-frame; pose/tracking/NVS are multi-frame. The writer handles both; the runner has a generic batched base that scales to all of them.

### 2025-05-07 — Feynman (DRS section rewritten — reviewer objections addressed)
**Rewrote `paper-submission/latex/drs_section.tex` to address all anticipated reviewer objections.**

**Fixes applied:**
1. **Keeney citation path corrected**: Now cites Keeney & Raiffa (1976) Ch. 3, Theorem 3.6 (value functions under certainty) instead of Keeney (1974) Theorem 1 (utility functions under uncertainty). Our setting is deterministic (scoring known measurements, not lotteries). Keeney (1974) retained in a Remark for context.

2. **Solvability/Archimedean conditions**: Added explicit sentence: "We work on the continuous, connected domain [0,1]³, which automatically satisfies the solvability and Archimedean conditions required by the representation theorems of Krantz et al. (1971)."

3. **Uniqueness claim scoped correctly**: Axioms determine the multiplicative *aggregation*. The efficiency function E(F) is explicitly acknowledged as a design choice, not axiomatically determined. New paragraph: "The axioms determine the multiplicative aggregation but do not uniquely determine the shape of E(F)." Validated by sensitivity analysis.

4. **Cross-edition comparability**: Added to Limitations: "DRS values are not directly comparable across benchmark editions that include different model sets. Cross-edition comparison requires fixing F_med."

5. **TP normalization caveat**: Changed to recommend using δ1 (already in [0,1]) as primary. The exp(-m) mapping is noted as "a modelling choice, not determined by the axioms."

6. **Equal-weight defense**: Framed as "maximum-entropy default." Added: "Applications with a specific deployment constraint may set β > α."

7. **Pareto relationship**: New paragraph explaining DRS as a total order complementing (not replacing) Pareto analysis. Iso-DRS contours are hyperbolas in the (TP, E) plane.

8. **Limitations section**: Explicit, numbered. Three items: within-sweep ranking, E is a design choice, equal weighting is a default.

**Proof now explicitly handles the v_i(0)=0 requirement** — not just "additive case eliminated" but also showing that the multiplicative form requires v_i(0)=0 from A2, which is satisfied by our normalization.

**Compiles cleanly**: 4 pages, all citations resolved. The `sec:str` reference is a stub (resolves when integrated into main paper).

**Files updated**: `paper-submission/latex/drs_section.tex`, this entry.

### 2025-05-07 — Feynman (DRS integrated into main NeurIPS paper)
**DRS is now in the paper proper, not just a standalone file.**

**Main body** (`text/03_method.tex`, §3.4):
- Renamed EDS → DRS
- Compact 12-line summary: formula, components, axiom reference, pointer to appendix
- No proof in main body — just the boxed formula and a statement that the multiplicative form is the only one satisfying the five axioms

**Appendix** (`text/13_appendix_drs.tex`, new, Appendix D):
- Full axiomatic derivation: 5 axioms, Proposition (multiplicative form), Proof (via Keeney & Raiffa 1976 Theorem 3.6), impossibility of additive scores
- Remark citing Luce & Tukey 1964, Krantz et al. 1971, Fishburn & Keeney 1975
- Efficiency function, multi-precision OPs, limitations, sensitivity analysis plan

**Infrastructure**:
- Added `amsthm` package + `proposition`/`remark` theorem environments to `root.tex`
- Added 5 bib entries to `root.bib`
- `text/12_appendix.tex` now `\input{text/13_appendix_drs}` at the end

**Compiles cleanly**: 28 pages, all citations resolved, no warnings.

**For Session B**: The §3.4 label is still `\label{sec:eds}` for backward compat with any cross-references in the experiments section. The display name in the text is now "Deployment Readiness Score (DRS)."

### 2025-05-07 — Feynman (CRITICAL FIX: DRS proof had a gap, now corrected)
**Found and fixed a real theoretical error in the DRS derivation.**

**The bug**: The original proof cited Keeney & Raiffa (1976) Theorem 3.6 and claimed the multiplicative form follows from it. But KR Theorem 3.6 gives TWO forms — additive and their "multiplicative" form `1+kV = prod(1+k*w_i*v_i)`. The essentiality axiom (A2) eliminates BOTH: the additive case trivially, and the KR multiplicative case because setting any v_j=0 gives V = [prod_{i≠j}(1+k*w_i*v_i) - 1]/k which is generically nonzero. Our Cobb-Douglas form is NOT a special case of the KR multiplicative form. The original proof was wrong.

**The fix**: Reformulated A5 as log-space separability rather than preferential independence. The correct argument:
1. On the interior (0,1]³, define T(y) = log S(exp(y))
2. A5 (log-separability) gives T = f_1(y_1) + f_2(y_2) + f_3(y_3)
3. Back-transform: S = g_1(x_1) * g_2(x_2) * g_3(x_3)
4. A2 forces g_i(0) = 0. A4 forces g_i(1) = 1. A1 forces g_i increasing.
5. The PRODUCT structure is axiomatically determined. The SHAPE of each g_i is NOT.
6. The power form g_i(x) = x^α is chosen for parsimony, validated by sensitivity analysis.

**What changed theoretically**:
- Axiom A5 now says "log S is additively decomposable" (cleaner, directly gives the product)
- Proposition now says "S decomposes as a product of single-attribute functions" (NOT "S is Cobb-Douglas")
- The power-law specialisation is explicitly stated as a modelling choice, not an axiomatic result
- A Remark explains why KR Theorem 3.6 doesn't apply and how the pure product differs
- Added Cobb & Douglas (1928) citation for the production-function connection

**This is MORE honest and STRONGER than before**: we don't overclaim uniqueness of the functional form, we clearly separate what the axioms give (product structure) from what we choose (power law), and the sensitivity analysis validates the choice.

**Files updated**: `text/13_appendix_drs.tex` (rewritten), `root.bib` (added cobbdouglas1928), paper compiles cleanly at 28 pages.

### 2025-05-07 — Feynman (runner fixes + DRS wiring + sensitivity analysis code)
**Four changes in one turn, addressing Session B's flagged issues + completing the DRS pipeline.**

**1. `skip_flops` flag added to runner** (`runner.py`)
- New kwarg `skip_flops: bool = False` on `run_with_deployment_readiness()`
- When True, skips the FLOP-counter dispatch entirely — prevents OOM on small GPUs where the counter itself doesn't fit
- Session B can now pass `skip_flops=True` for Depth Pro on 8GB cards

**2. FLOP counter uses batch-of-1** (`runner.py`)
- Previously wrapped the entire batch in `FlopCounterMode`, scaling dispatch overhead by batch_size
- Now counts FLOPs on `[batch[0]]` only, then runs the full batch normally (timed separately)
- Result: `first_batch_flops_g` is already per-sample, no division needed

**3. DRS wired into runner** (`runner.py` + `deployment.py`)
- `DeploymentReadinessReport` gains `operating_point: OperatingPoint | None` field
- After computing ERS, the runner builds an `OperatingPoint` from the weighted phase score + STR + FLOPs + precision
- Session B can collect these from each model's report and pass to `compute_sweep_drs()` post-sweep:
  ```python
  from rpx_benchmark import compute_sweep_drs
  models = {name: [report.operating_point] for name, report in all_reports.items()}
  drs_results = compute_sweep_drs(models)
  ```

**4. Sensitivity analysis code** (`rpx_benchmark/drs_sensitivity.py`, new)
- `run_sensitivity(models)` → `SensitivityReport` with Kendall τ tables
- Test 1: 27 exponent combos (α,β,γ ∈ {0.5, 1.0, 2.0}³)
- Test 2: 4 alternative E functions (exponential, inverse, log, linear)
- Test 3: 5 anchor multipliers (0.25× to 4×)
- Includes `_kendall_tau()` implementation (no scipy dependency)
- `report.to_markdown()` produces paper-ready tables
- Verified on 8-model mock data: min τ = 0.857, mean τ = 0.979

**For Session B**: After the full sweep, generating the sensitivity tables is:
```python
from rpx_benchmark.drs_sensitivity import run_sensitivity
report = run_sensitivity(models)  # same dict as compute_sweep_drs
open("sensitivity.md", "w").write(report.to_markdown())
```

**Tests**: 511 passed, same 4 pre-existing failures. PDFs regenerated.

**Files**: `runner.py`, `deployment.py`, `drs_sensitivity.py` (new), PDFs, this entry.

### 2025-05-07 — Feynman (sweep-readiness pass: 0 test failures + EDS→DRS + native_precision)
**Cleared the path for the full sweep. All blockers on the Feynman side are resolved.**

**1. All 515 tests passing (was 511 passed + 4 failed)**
- Fixed feature count assertions in CLI tests (27→31, feature set grew with fisheye+stereo additions)
- Fixed bare `raise ValueError/RuntimeError/KeyError` in Session B's adapter code → `ConfigError`/`AdapterError` with hints
- Fixed VQA caplog test (logging handler mismatch — warning goes to module logger, not caplog)

**2. EDS→DRS throughout the paper**
- All references in `text/01_intro.tex`, `text/03_method.tex`, `text/06_experiments.tex`, `text/10_conclusion.tex` updated
- "Embodied Deployment Score (EDS)" → "Deployment Readiness Score (DRS)" everywhere
- Paper recompiled cleanly, 28 pages

**3. `native_precision` added to adapter contract**
- `BatchedDepthBenchmarkModel.__init__` now accepts `native_precision: str`
- Reads from adapter attribute if present (same pattern as `native_alignment`)
- Runner reads it when building the `OperatingPoint` for DRS
- Session B adapters can declare `native_precision = "fp16"` and the DRS pipeline picks it up automatically

**For Session B — the sweep is unblocked on the Feynman side:**
1. All tests green (515/515)
2. Runner emits `OperatingPoint` on every `DeploymentReadinessReport`
3. Post-sweep: `compute_sweep_drs(models)` + `run_sensitivity(models)` → paper tables
4. `skip_flops=True` available for Depth Pro on small GPUs
5. FLOP counter uses batch-of-1 (no more overhead scaling)
6. `native_precision` wired through the full pipeline

**What Session B should do next:**
1. Add `native_precision = "fp16"` to adapters that support it (DA-V2, UniDepth V2)
2. Run ONE model on the full easy split (3000 frames) to validate end-to-end
3. If that works, launch the full 20-model sweep
4. Collect `report.operating_point` from each run → `compute_sweep_drs()` → DRS table
5. Feed the same data to `run_sensitivity()` → fill appendix placeholders

**Files**: test fixes (3 test files), `batched_depth.py`, `runner.py`, all paper tex files, PDFs, this entry.

### 2026-05-07T~later — Session B (Claude — sweep readiness pass: 5/5 top priorities)
**Self-audit + fix-up turn. Per the user's "do what's best for our case", landed all 5 of the top-5 priorities I'd flagged.**

**1 — DRS sweep aggregator (`scripts/run_drs_sweep.py`):**
- Walks `rpx_results/<model>/<split>/result.json`, pulls each `OperatingPoint` (already emitted per-run by Feynman's runner upgrade), calls `compute_sweep_drs(...)` with the median-FLOPs anchor.
- Writes `rpx_results/_sweep/drs_<split>.{csv,json}`. Optional `--sensitivity` runs Feynman's `run_sensitivity` from `drs_sensitivity.py` and emits the sensitivity JSON.
- Verified on 3 fresh per-model runs: top-3 DRS table prints cleanly. The 6 stale runs from before Feynman's runner upgrade get skipped with a clear log line.

**2 — `native_precision` declared on supported adapters:**
- `DepthAnythingV2Metric.native_precision = "fp16"` (DPT/ViT-L runs cleanly under autocast).
- `UniDepthV2.native_precision = "fp16"` (the paper's reported inference mode).
- `HFDepthEstimationAdapter` accepts `native_precision` constructor arg (used by registry builders).
- ZoeDepth, Depth Pro stay default `"fp32"`.
- Verified: `result.json["deployment_readiness"]["operating_point"]["precision"]` reads `"fp16"` for UniDepth V2, `"fp32"` for ZoeDepth.

**3 — Typed-exception sweep through my code:**
- 0 bare `ValueError`/`RuntimeError`/`KeyError`/`FileNotFoundError`/`TypeError`/`NotImplementedError` remaining in my hot-path code.
- All converted to `AdapterError` / `ConfigError` / `DatasetError` with hint strings.
- Sites touched: `run_depth.py`, `local_manifest.py`, `comprehensive_depth_metrics.py`, `split_manifests.py` (the ones I authored), all 5 depth-model adapters.
- `ImportError` for missing optional deps (transformers, unidepth, pandas) kept bare per Python convention.

**4 — Consolidated `_align` into one shared module:**
- New `rpx_benchmark/metrics/depth_alignment.py` with `align_pred_to_gt(pred, gt, mode, valid=...)` + `default_valid_mask(pred, gt)` + named constants (`DEPTH_MIN_M`, `DEPTH_MAX_M`, `ALIGNMENT_MODES`).
- `BatchedDepthBenchmarkModel` and `comprehensive_depth_metrics._align` both delegate to the canonical aligner now.
- Future fix lands in one place; no more "fixed in one but not the other" drift.

**5 — Integration test for `download_split → from_manifest` round-trip (`tests/test_dataset_hub_integration.py`):**
- 7 contract tests (5 task round-trips + extraction-layout + idempotency); 2 paired-task tests skip on the 4-frame mock (stride=5 needs more frames).
- Walks pack → manifest (with synthetic splits) → write_split_manifests → `_extract_snapshot_tars` → `RPXDataset.from_manifest` → sample iteration.
- Catches every silent contract bug we fought through this session: TaskType-string mismatches, per-task entry-key drift, modality file-extension drift, tar-layout drift, extraction-gap regressions.

**Verified end-to-end:**
- Depth smoke (n=2, ZoeDepth): AbsRel 0.083 / δ1 0.94 — unchanged after refactor.
- 27 tests passing total (20 unit + 7 integration). Plus Feynman's existing 515.

**Code-quality posture (compared to start of session):**

| Area | Before | After |
|---|---|---|
| Bare exceptions in my code | dozens | 0 (in hot paths) |
| Duplicate `_align` impls | 2 | 1 (single source of truth) |
| Tests for `download_split` round-trip | 0 | 7 |
| DRS pipeline end-to-end | runner emits OPs, no aggregator | full sweep aggregator + sensitivity hook |
| Adapter `native_precision` declared | 0 | 2 (DA-V2, UniDepth V2) |
| Typed exceptions in `local_manifest`, `split_manifests`, depth adapters | mixed bare/typed | consistent typed |

**Files touched (one turn):**
- NEW `benchmark/scripts/run_drs_sweep.py` — sweep DRS + sensitivity aggregator.
- NEW `benchmark/rpx_benchmark/metrics/depth_alignment.py` — canonical alignment.
- NEW `benchmark/tests/test_dataset_hub_integration.py` — 9 contract tests (7 active, 2 stride-skipped).
- `benchmark/scripts/depth_models/depth_anything_v2.py`, `unidepth_v2.py` — `native_precision = "fp16"`.
- `benchmark/scripts/depth_models/hf_pipeline.py` — `native_precision` constructor arg.
- `benchmark/rpx_benchmark/adapters/batched_depth.py` — `_align_pred_to_gt` removed (delegates to shared).
- `benchmark/scripts/comprehensive_depth_metrics.py` — `_align` shim over canonical.
- `benchmark/scripts/local_manifest.py`, `run_depth.py`, all 5 depth-model adapters — typed exceptions.
- `benchmark/rpx_benchmark/dataset_hub/split_manifests.py` — typed exception on missing parquet.
- This entry.

**For Feynman:** the sweep aggregator works end-to-end. When all 20 adapters land + run, `python scripts/run_drs_sweep.py --split <s> --sensitivity` produces both the headline DRS table and the appendix sensitivity Kendall's-τ JSON in one shot. Ready for the day-data-lands launch.

**Next:** Turn C (Marigold + Marigold-LCM + Lotus-2 + GeoWizard diffusion adapters). The plumbing is now solid enough that those should be straightforward registry entries.

### 2025-05-07 — Feynman (TODO: DRS downstream validation — deferred, needs design thinking)
**DEFERRED. Will revisit after sweep results are in.**

**The open question**: How to prove DRS correlates with actual robot deployment success?

**Three approaches considered, none committed:**
- **A**: Per-task DRS, validate each independently (proves components, not pipeline)
- **B**: Pipeline-level DRS aggregate across tasks (assumes equal importance — wrong)
- **C**: Per-task DRS reported independently, one integrated sim experiment (most honest)

**Three design questions to answer before building anything:**
1. What robot operation to validate against? (pick-and-place? language-conditioned?)
2. Is per-task DRS the claim, or is there a cross-task story?
3. Is sim validation required for submission, or is benchmark + DRS + findings enough?

**Candidate sims**: ManiSkill 3 (best for integrated pipeline), SimplerEnv (CoRL 2024, proven sim↔real correlation), AnyGrasp (grasp quality proxy, no sim needed)

**Decision**: Come back after sweep results show whether DRS actually differentiates models. If all models cluster near the same DRS, the validation question is moot. If they spread out, the validation becomes the paper's strongest section.

### 2025-05-07 — Feynman (Relative Camera Pose Estimation: model survey + metrics)
**Created `docs/methods/relative_pose_models.md` + PDF — comprehensive survey for the RCPE task.**

**14 models verified with open code + checkpoints:**

| # | Category | Models |
|---|----------|--------|
| 1–7 | Direct regression (RGB) | Reloc3r (CVPR'25), DUSt3R (CVPR'24), MASt3R (2024), FAR (CVPR'24), 8-Point ViT (3DV'22), SRPose (ECCV'24), NOPE-SAC (TPAMI'23) |
| 8–10 | Feature matching + solver (RGB) | LoFTR (CVPR'21), SuperGlue (CVPR'20), MicKey (CVPR'24 Oral) |
| 11 | Classical baseline (RGB) | OpenCV essential matrix (SIFT/ORB + RANSAC) |
| 12–14 | RGBD methods | ICP (Open3D), Colored ICP, SparsePlanes (ICCV'21) |

**3 models NOT available**: GeLoc3r (ICLR'26, desk rejected, no code), MultiLoc (no code), IUP-Pose (no code)

**Comprehensive metrics documented (15 metrics):**
- Per-sample: rotation error, translation error (L2), translation angular error, pose correct @(X°,Ym) at 4 threshold pairs
- Aggregate: AUC@5°/10°/20°, RRA@15°, RTA@15°, mAA@30°, median errors, pose recall, TAS/RAS/PAS
- Robotics-specific: metric-scale translation error, per-phase accuracy, per-difficulty accuracy

**Key differences vs existing pose benchmarks:**
- ~178K pairs (vs 1,500 in ScanNet1500, ~2K in MapFree) — 100× larger
- Three-phase STR: does pose degrade when hand enters scene?
- Indoor + outdoor mix (64 + 35 scenes)
- Scale alignment noted: some methods are up-to-scale (LoFTR+RANSAC), others are metric (DUSt3R, MicKey)

**Alignment handling**: same pattern as depth — `native_alignment` on each adapter. Scale-ambiguous methods use translation angular error as primary. Metric methods use L2 translation error.

**Resource utilization**: Same 3-tier profiler. DRS with AUC@10° as TP component.

**Currently missing in codebase** (to implement):
- 9 of 15 metrics not yet in `metrics/pose.py` (only rotation_error_deg + translation_error_m exist)
- No model adapters
- No `BatchedPoseBenchmarkModel` (but `BatchedTaskBenchmarkModel` abstract base exists from Session B)

**Files**: `docs/methods/relative_pose_models.md` + PDF, this entry.

### 2025-05-07 — Feynman (RCPE data pipeline verified + depth_a/depth_b loader fix)
**Verified the full data flow for relative camera pose estimation. Found and fixed one gap.**

**What's already working (no changes needed):**
- `split_manifests.py`: `_RelativePoseSpec` pairs frames with stride-5, produces `rgb`, `rgb_b`, `pose_a`, `pose_b`
- `split_manifests.py`: `_RGBDRelativePoseSpec` extends with `depth_a`, `depth_b`
- `loader.py`: loads `rgb_b` into `Sample.metadata["rgb_b"]` ✓
- `loader.py`: loads `pose_a` + `pose_b`, computes `T_rel = inv(T_A) @ T_B` → `RelativePoseGroundTruth` ✓

**Gap found and fixed:**
- `loader.py` did NOT load `depth_a` / `depth_b` for the RGBD variant. Added 4 lines to `_load_sample()` so these are loaded into `Sample.metadata["depth_a"]` and `Sample.metadata["depth_b"]` as float32 depth maps in meters.

**Data contract for pose adapters:**
```
Sample.rgb                      = frame_a (H×W×3 uint8)
Sample.metadata["rgb_b"]        = frame_b (H×W×3 uint8)
Sample.ground_truth.rotation    = 3×3 float64 (relative rotation)
Sample.ground_truth.translation = 3-vec float64 (relative translation in meters)
# RGBD variant additionally:
Sample.metadata["depth_a"]      = frame_a depth (H×W float32, meters)
Sample.metadata["depth_b"]      = frame_b depth (H×W float32, meters)
```

**Scale**: ~178K pairs total (99 scenes × 3 phases × ~600 pairs/phase)

**Tests**: 524 passed, 0 failed, 2 skipped (no mock pose data — expected). PDFs regenerated.

### 2025-05-07 — Feynman (Pose pair sampling strategy — DESIGN, not yet implemented)
**Created `docs/methods/pose_pair_sampling.md` — smart pair sampling for relative pose.**

**Problem**: Stride-5 consecutive frames give trivially similar viewpoints. Need pose-distance-aware sampling.

**Existing benchmark scales for reference**:
- ScanNet1500: 1,500 pairs
- MegaDepth1500: 1,500 pairs
- RUBIK (CVPR'25): 16,500 pairs (33 difficulty bins × 500)
- RPX target: **75,000 pairs** — 50× ScanNet, 5× RUBIK

**Proposed algorithm: Pose-Distance Binned Sampling**:
1. Compute pairwise pose distances within each (scene, phase) using T265 poses
2. Bin by angular distance: [0-5°, 5-15°, 15-30°, 30-60°, 60-180°]
3. Sample ~253 pairs per (scene, phase) with budget distribution: 10%/20%/30%/30%/10% across bins
4. Farthest-point refinement within each bin for maximum diversity
5. Seed: RPX_SEED for reproducibility

**Alternative considered**: Poisson disk sampling in continuous pose-distance space (no bin boundaries, but less control over difficulty distribution).

**Implementation**: `scripts/generate_pose_pairs.py` — outputs a manifest JSON with the same schema as existing per-task manifests. Replaces stride-5.

**Open questions**: strict 75K budget or approximate? Fixed bins or data-driven percentiles? Translation scale factor for binning?

**For Session B**: Do NOT use stride-5 for pose evaluation. Wait for this sampling script. The existing `_RelativePoseSpec` in split_manifests.py will be replaced by a reference to the pre-generated pairs file.

### 2025-05-07 — Feynman (RCPE additional considerations — 8 issues identified)
**Updated `docs/methods/relative_pose_models.md` with 6 critical considerations beyond the model list.**

**Issues that affect experiment design:**

1. **T265 GT quality**: Centimeter-level accuracy, not millimeter. Narrow-baseline pairs (< 5°) may have GT noise dominating signal. Interaction phase GT degrades (IR occlusion). Must report GT uncertainty and flag interaction pairs.

2. **Scale ambiguity**: Three model categories. Feature-matching methods give UP-TO-SCALE translation (angular error only). Regression methods give METRIC translation (L2 in meters). Each adapter needs `native_scale` field. Cannot compare L2 across categories without scale alignment.

3. **Camera intrinsics**: D435 fx≈fy≈615px at 640×480. Essential-matrix methods need these. DUSt3R/Reloc3r estimate their own. Need to provide and optionally test with/without.

4. **Pair overlap**: Beyond pose-distance binning, should compute co-visibility fraction per pair. Enables RUBIK-style stratified analysis. Can approximate from depth+pose point projection.

5. **Three-phase STR**: Interaction phase introduces hands (transient features), object motion (non-rigid), T265 degradation. Must separate "model failure" from "GT failure." Safest approach: use clutter→clean STR as reliable signal, report interaction with caveat.

6. **Indoor/outdoor stratification**: 64 indoor + 35 outdoor scenes. Different challenges (texture-poor indoor vs large-baseline outdoor). Report stratified results.

7. **Pair sampling refinements**: Filter pairs with overlap < 10%. Separate interaction-phase pair set. Balance indoor/outdoor representation in the 75K budget.

8. **New model found**: RePoseD (ICCV 2025) — relative pose with known depth. Need to verify code/checkpoint availability.

**For Session B**: When building pose adapters, each must declare both `native_alignment` (if applicable) AND `native_scale = "metric" | "up_to_scale"`. The runner should compute different metrics based on `native_scale`.

### 2025-05-07 — Feynman (NOTE: interaction phase for pose — decision deferred)
**User directive**: Whether to include interaction-phase pairs in the relative pose evaluation is NOT decided yet. Will revisit later. Reasons for caution: T265 GT degrades during interaction (IR occlusion by hands), making it hard to separate model failure from GT failure. For now, design the pipeline to support both (include/exclude interaction) and defer the decision.

### 2026-05-06 — Session B / Jishnu (RCPE pipeline shipped — adapters + metrics + runner)
**Branch**: `jishnu/rcpe-pipeline-full` (off `jishnu/rcpe-csv-log` = PR #26).

**Shipped:**
- `scripts/pose_models/` — 10-adapter registry with shared `_pose_base.py`. Keys: `reloc3r`, `dust3r`, `mast3r`, `far`, `srpose`, `nope_sac`, `mickey`, `loftr`, `opencv_baseline`, `icp_open3d`. Each adapter declares `native_alignment ∈ {none, unit}` and `native_precision`. Heavy optional deps (kornia, open3d, dust3r, etc.) are imported at construct time, not at module import — registry parses on bare envs.
- `scripts/pose_comprehensive_metrics.py` — pose-error basket: `rotation_error_deg` (geodesic), `translation_l2`, `translation_angular_deg`, `pose_error_max_deg`, AUC@5°/10°/20° (trapezoidal CDF integration matching MapFree/ScanNet/FAR/Reloc3r/DUSt3R/MASt3R convention). Aggregated with 95% CIs, by-phase, by-stride.
- `scripts/run_relative_pose.py` — runner mirroring `run_depth.py` (manifest build → batched runner → DR report → optional comprehensive metrics → optional Box upload). Default `--model opencv_baseline` works without GPU/optional deps.
- `tests/test_pose_comprehensive_metrics.py` (11 tests) + `tests/test_pose_models_registry.py` (10 tests) — pin metric math + registry shape + adapter import smoke. **545 passed / 2 expected skips.**
- `README.md` + `TEAM_LAUNCH.md` — new RCPE section with model table, install extras, output schema. Section numbers renumbered.

**Per Feynman's RCPE considerations (#3 above):** every adapter declares `native_alignment` (no `native_scale` field — folded into `none` vs `unit`). `loftr` + `opencv_baseline` use D435 focal=615px consistent with the depth pipeline's FOV correction. `icp_open3d` requires `depth_a`/`depth_b` and falls back to identity on RGB-only splits.

**Predictions log format** (per user directive, PR #26): single CSV at `out_dir/predictions.csv` with 16 columns `scene_id, phase, frame_a, frame_b, R00..R22, tx, ty, tz`. Header written once on first append; resume-safe.

**Open**: pose pair sampling still stride-5 (Feynman's binned-sampling design not yet implemented — see `docs/methods/pose_pair_sampling.md`). Interaction-phase inclusion still TBD.
