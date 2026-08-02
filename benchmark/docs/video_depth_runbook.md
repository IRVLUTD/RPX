# Video Depth (D1-V) Benchmarking Runbook

Operational guide for running Video Depth models against RPX. Pairs with
[`depth_smoke_runbook.md`](./depth_smoke_runbook.md) (Image Depth) — same
host + environment setup, different runner + metric set.

Design source of truth is
[`../../SESSION_HANDOFF.md`](../../SESSION_HANDOFF.md) — do not use this
runbook to argue for or against metric composition; it describes what
the pipeline **does**, not what it should be.

---

## 1. What comes out

Every run writes one `cells.parquet` per `(model, task, scene, phase[, frame_budget])`
key. Every row has every metric column below (nan where uncomputable).

All headline and diagnostic depth metrics use the same paper-valid GT
population: finite pixels satisfying `0.3 < depth < 5.0 m`. Raw predictions
must be finite and non-degenerate. Matching D1-F, raw cached predictions are
preserved while one evaluation copy is uniformly clipped to `[0.3, 5.0]`
before every spatial and temporal metric.

### K=6 primary vector (Φ MANOVA + JEDI inputs)

RGB-D-only — no external flow model, no per-frame poses required.

| Key | Direction | What |
|---|---|---|
| `metric:absrel` | ↓ | Absolute relative error `mean(|d̂ − d| / d)` |
| `metric:rmse` | ↓ | Root-mean-square error (metres) |
| `metric:delta1` | ↑ | Fraction of pixels with `max(d̂/d, d/d̂) < 1.25` |
| `metric:silog` | ↓ | Scale-invariant log error, KITTI display convention (100·√Var(log err)) |
| `metric:tgm` | ↓ | Temporal Gradient Matching — L1 of `|Δd_pred| − |Δd_gt|` in static regions |
| `metric:tgse` | ↓ | Temporal Gradient Squared Error — L2 signed variant of TGM, penalises severe flicker |

`fscore_5cm` is in D1-F's K=5 but **not** in D1-V's K=6 — it stays as a
diagnostic on the video side. Temporal metrics take the two extra
"slots" that video depth needs to be about (per SESSION_HANDOFF).

### Diagnostics (emitted, not in K)

| Category | Keys | Notes |
|---|---|---|
| Threshold accuracy | `delta2`, `delta3` | Near-saturated on SOTA |
| Robotics 3D | `fscore_5cm` | Per-frame → clip mean; expensive at 250 fps (see §4) |
| Temporal (RGB-D-safe) | `tcc`, `tmc` | SSIM-based, GT-referenced |
| Temporal (needs poses) | `tae`, `tae_near`, `tae_mid`, `tae_far` | Nan unless manifest ships `pose_filenames` |
| Temporal (needs RAFT) | `opw` | Nan unless a flow backend is wired; RGB-D-only policy keeps it out of K anyway |
| Range-stratified | `absrel_{near,mid,far}`, `rmse_*`, `delta1_*`, `tgm_*`, `tgse_*` | Depth bins: near 0.3–1.0 m, mid 1.0–2.5 m, far 2.5–5.0 m |

---

## 2. Host + environment setup

Reuse the exact `nvidia-smi`, `RPX_WORK_ROOT`, `HF_HOME`, per-model Python
environments, and `hf auth login` from
[`depth_smoke_runbook.md`](./depth_smoke_runbook.md) §§1–2. Nothing changes
for D1-V — same weights, different runner.

Adapters live in `scripts/video_depth_models/`. Every kebab-case model
key in `DEPTH_MODEL_CARDS` with `task == VIDEO_DEPTH` maps to a module
under that directory that exposes `build(device) → BenchmarkModel`.

---

## 3. First smoke — 2 clips, frame-as-video baseline

The safest starting point is `da-v2-video` — a shim that reuses the
Image Depth `DA-V2 Large` adapter you already validated in D1-F.

```bash
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da-v2-video --split easy --max-samples 2 \
    --save-predictions --resume-predictions
```

Success means `rpx_results/da-v2-video/easy/cells.parquet` exists with
these 6 columns populated (finite, non-nan):

```
metric:absrel  metric:rmse  metric:delta1  metric:silog  metric:tgm  metric:tgse
```

If those are populated, the pipeline is green — scale to `--split easy`
(full), then to true video models.

---

## 4. Frame-budget sweep — the RPX-unique diagnostic

Run the pipeline once per budget so temporal-cue-value analysis has
independent per-budget cell logs.

```bash
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da-v2-video --split easy \
    --budget-sweep 50,100,150,250
```

Behaviour:

- Model weights load **once**, reused across all budgets — no per-budget GPU warmup.
- Each budget writes to its own subdirectory:

  ```
  rpx_results/da-v2-video/easy/
  ├── budget_50/cells.parquet    (frame_budget=50 for every row)
  ├── budget_100/cells.parquet   (frame_budget=100)
  ├── budget_150/cells.parquet
  └── budget_250/cells.parquet
  ```

- Sampling defaults to `stride` (uniform with endpoint anchoring —
  equivalent to "what if the camera ran at N/T·30 FPS"). Override with
  `--sampling fps_se3` for farthest-point in SE(3) if you have poses.
- `--budget-sweep` and `--frame-budget` are mutually exclusive.
- `--max-samples` applies **per budget** — smoke with
  `--budget-sweep 50,250 --max-samples 2` = 4 quick runs (2 clips × 2 budgets).

Degradation analysis (TCV, AUDC, critical budget, Wilcoxon) is a
**post-processing** step over the per-budget cell logs — see
`rpx_benchmark.temporal_budget_sweep.degradation_analysis`. Not
computed inside the sweep so the raw cells stay authoritative.

---

## 5. Known caveats

### Nan columns that are fine and expected

| Column | Why nan | Impact |
|---|---|---|
| `metric:opw` | No RAFT backend wired in RPX today | None — OPW is diagnostic, not in K |
| `metric:tae`, `metric:tae_{near,mid,far}` | Split manifest has no `pose_filenames` | None — TAE is diagnostic under the RGB-D policy |

### F@5cm CPU cost

`fscore_5cm` computed inline for 250-frame clips × 100 scenes × 3 phases
is heavy (bidirectional cKDTree per frame). It is therefore **disabled by
default** for D1-V and emitted as `nan`; it is not part of the headline
K=6 vector. Pass `--compute-fscore` only for an explicit diagnostic run.

### Prediction persistence and resume

Use `--save-predictions --resume-predictions` for production. Each completed
scene-phase clip is atomically written to:

```
<output-dir>/predictions/<scene>/<phase>/depth.npz
```

The NPZ contains aligned raw float32 depth, the exact frame indices, and original
inference latency. Resume loads with pickle disabled, verifies the NPZ CRC,
keys, dtype, shape, frame identity, finiteness and non-degeneracy. Missing or invalid clips are inferred and atomically
replaced; valid clips perform no model forward. Counts are recorded in
`run_metadata.json`.

### Complete-case dropping in Φ

`phi_jedi_summary._group_by_scene_phase` drops any scene missing any
K-vector column across any phase. Missing a `metric:tgse` on scene X
phase 1 drops the whole X from that model's Φ. Cell logs are the
source of truth — if a row is missing, fix the run, don't paper over.

---

## 6. Post-processing flow

Raw metrics only during runs. Φ + JEDI are computed after all models
are done.

1. Run all 10 video-depth adapters × splits × (optionally) budgets.
   Cells accumulate in `rpx_results/<model>/<split>/[budget_<N>/]cells.parquet`.
2. Merge cell logs across models. Any script that reads
   `pyarrow.parquet` works; the schema is stable.
3. **Freeze empirical bounds** for the non-theoretical K-vector metrics
   (`absrel`, `rmse`, `silog`, `tgm`, `tgse`) from the observed max/min
   across the zoo. Update `rpx_benchmark/metrics/specs.py` accordingly.
   `δ₁` is theoretical [0, 1] and needs no freeze.
4. Compute Φ per (model, task) via
   `rpx_benchmark.phi.compute_phi_oneway(z, metric_names=D1V_MANOVA_METRICS)`
   — the collinearity guard names offending pairs; the bootstrap CI is
   available via `bootstrap_phi_oneway`.
5. Compute JEDI per (model, cell) via `rpx_benchmark.jedi.compute_jedi`
   with the frozen specs.
6. Fill paper tables via `scripts/fill_paper_table.py`.

Provisional bounds shipped in `specs.py` today (`absrel` worst=0.5,
`rmse` worst=2.0, `silog` worst=25.0, `tgm`/`tgse` worst=0.1) are for
diagnostic use only — **do not publish JEDI numbers computed against
them**.

---

## 7. Adapter roster (D1-V)

| Model key | Adapter kind | Runnable today? |
|---|---|---|
| `da-v2-video` | Frame-as-video shim over DA-V2 Large | Yes |
| `video-da` | Video Depth Anything | Yes (needs its package) |
| `depth-crafter` | DepthCrafter | Yes (needs its package) |
| `rolling-depth` | RollingDepth | Yes (needs its package) |
| `monst3r` | MonST3R | Yes (needs its package) |
| `vggt-omega` | VGGT-Ω (1B) | Yes (needs `vggt` package) |
| `da3-video` | DA3 Metric-L applied per clip | Yes |
| `chrono-depth` | ChronoDepth | Yes (needs its package) |
| `vigeo` | ViGeo | Yes |
| `dvd` | Official DVD v1.1 + Wan2.1, pinned Docker overlay | Relative inverse depth; per-clip `ls_disparity` alignment |
| `gem-depth` | Official source and checkpoint, pinned Docker overlay | Relative inverse depth; per-clip `ls_disparity` alignment |

Adapter contracts (module resolution, `build(device)` signature, verified
`depth_output_kind`) are exercised by
`tests/test_video_depth_skeletons.py`.

---

## 8. Cross-references

- Framework design + literature grounding: [`../../SESSION_HANDOFF.md`](../../SESSION_HANDOFF.md)
- Per-metric bounds + direction registry: [`../rpx_benchmark/metrics/specs.py`](../rpx_benchmark/metrics/specs.py)
- Φ MANOVA machinery: [`../rpx_benchmark/phi.py`](../rpx_benchmark/phi.py)
- JEDI composite: [`../rpx_benchmark/jedi.py`](../rpx_benchmark/jedi.py)
- D1-F equivalent runbook: [`./depth_smoke_runbook.md`](./depth_smoke_runbook.md)
- Team run guide: [`./team_run_guide.md`](./team_run_guide.md)
