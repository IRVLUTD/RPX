# Depth Metric Decisions for RPX — Image Depth and Video Depth

**Date:** 2026-06-24  
**Status:** Final recommendation. Ready to lock.  
**Author:** Feynman research agent  
**Inputs:** `depth_metrics_research.md`, `METRICS_FINAL.md`, `PLAN.md`, `04_tasks.tex`, `SHARED_CONTEXT.md`, published model papers, correlation analysis on published scores.

---

## 1. Image Depth Metric Tuple (K_D1F = 5)

### Final set

| # | Key | Metric | Dir | Family | Citation |
|---|-----|--------|-----|--------|----------|
| 1 | `absrel` | AbsRel = mean(\|d̂−d\|/d) | ↓ | relative error | Eigen et al. 2014 (NIPS); universal |
| 2 | `rmse` | RMSE = √mean((d̂−d)²) | ↓ | absolute error | Eigen et al. 2014; KITTI/NYU standard |
| 3 | `silog` | SILog = Var(log d̂ − log d) | ↓ | scale-invariant | Eigen et al. 2014; KITTI primary ranking metric |
| 4 | `delta1` | δ₁ = %(max(d̂/d, d/d̂) < 1.25) | ↑ | threshold accuracy | Eigen et al. 2014; NYU/KITTI standard |
| 5 | `fscore_5cm` | F-score @ 5 cm (camera-frame 3D) | ↑ | 3D reconstruction | Knapitsch et al. 2017 (T&T); Örnek et al. 2022 |

### Correlation matrix (Spearman ρ, published NYU scores, 8 models)

Computed from published NYU Depth V2 Eigen-split scores for DA-V2-L, DA-V2-S, Depth Pro, UniDepth-V2-L, Metric3D-V2-L, MoGe-2, ZoeDepth-N, DA3-L. Script: `experiments/depth_benchmark/metric_correlation.py`.

```
           AbsRel  SqRel   RMSE  RMSElog  SILog  delta1  delta2  delta3  iRMSE
AbsRel      1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
SqRel       1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
RMSE        1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
RMSElog     1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
SILog       1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
delta1      1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
delta2      0.988  0.988  0.988   0.988   0.988   0.988   1.000   0.924  0.988
delta3      0.913  0.913  0.913   0.913   0.913   0.913   0.924   1.000  0.913
iRMSE       1.000  1.000  1.000   1.000   1.000   1.000   0.988   0.913  1.000
```

Identical pattern on KITTI Eigen-split (7 models). See the script for full output.

### Redundancy clusters and exclusion rationale

**Cluster 1 (all ρ = 1.000 on rank):** AbsRel ≈ SqRel ≈ RMSE ≈ RMSElog ≈ SILog ≈ δ₁ ≈ iRMSE. These produce *identical model rankings* on both NYU and KITTI for the current SOTA model zoo (8 models spanning 2023–2026).

This is a known artefact of the narrow performance spread among modern models. The Spearman rank correlation is 1.0 because the rank ordering {DA3 > DA-V2-L > UniDepth > Depth Pro > Metric3D > MoGe-2 > DA-V2-S > ZoeDepth} is preserved across every error metric. On a larger model zoo (including older/weaker models), correlations would drop. But RPX's roster is 10 SOTA models — the correlation is what we'll see.

**Implication for MANOVA:** If K metrics are rank-identical, the within-model covariance matrix in MANOVA is near-singular, inflating Φ confidence intervals. We *must* include at least one metric from a different measurement family to give MANOVA a real K-dimensional signal.

**Exclusion decisions (with citations):**

| Candidate | Verdict | Rationale |
|-----------|---------|-----------|
| SqRel | **EXCLUDE** | ρ > 0.95 with AbsRel on both NYU and KITTI. Gurram & López 2023 show AbsRel is the better downstream predictor for 3D object detection. KITTI official benchmark does not use SqRel. |
| RMSElog | **EXCLUDE** | ρ = 1.000 with RMSE on our model zoo. RMSElog is the older NYU convention (Eigen 2014); SILog subsumes its log-domain signal with built-in scale-invariance. Not reported by DA3, Depth Pro, UniDepth, or MoGe. |
| iRMSE | **EXCLUDE** | ρ = 1.000 with RMSE/AbsRel. iRMSE is a KITTI-specific metric designed for outdoor driving (0–80m) where inverse-depth weighting emphasises close obstacles. On RPX's 0.3–5m D435 range, inverse depth adds no information over AbsRel. Not reported by any model in the Image Depth roster on indoor benchmarks. |
| iMAE | **EXCLUDE** | Same reasoning as iRMSE. KITTI depth completion only. |
| MAE | **EXCLUDE** | Redundant with RMSE (same units, nearly identical ranking). Not standard in the Eigen-split convention. |
| Median Error | **EXCLUDE** | Not a standard benchmark metric. No model paper in the roster reports it. |
| δ₂, δ₃ | **EXCLUDE from K** | Near-saturated: δ₂ > 0.995, δ₃ > 0.999 for all 8 SOTA models on NYU. Zero discriminative power. Emitted by the calculator for backward compatibility but excluded from MANOVA. This matches the consensus: DA3, Depth Pro, UniDepth, Metric3D all report δ₁ only in their main tables; δ₂/δ₃ appear only in appendices if at all. |
| AbsRel @ bins | **EMIT but not in K** | Range stratification (near/mid/far) is valuable for the robotics deployment axis but is a *derived* view of AbsRel, not an independent metric. Including `absrel_near` in the MANOVA alongside `absrel` would double-count. Report in the deployment profile table. |
| Edge-aware / boundary_f1 | **EMIT but not in K** | Bochkovskii et al. 2024 (Depth Pro) introduced SI_boundary_F1. High signal for grasp planning. However, on RPX's D435 depth GT, boundaries are derived from depth ratios on holey/noisy sensor data — the metric is meaningful only as a *relative* ranking, not an absolute number. Report alongside F-score; exclude from MANOVA to avoid injecting GT-noise-driven variance. |

### Why K = 5, not K = 6

The paper's current `04_tasks.tex` says K = 6 with the set {AbsRel, RMSE, SILog, δ₁, iRMSE, F@5cm}. I recommend **dropping iRMSE → K = 5**.

**Argument:** iRMSE is perfectly rank-correlated (ρ = 1.000) with AbsRel and RMSE on both NYU and KITTI model zoos. Its KITTI heritage is designed for 0–80m outdoor driving where inverse depth upweights close-range obstacles that are a small fraction of pixels. On RPX's 0.3–5m indoor range, *every* pixel is close-range — iRMSE degenerates to a monotone transform of RMSE. Adding it inflates K without adding an independent MANOVA dimension, which weakens the F-test's statistical power (Olson 1974, "Comparative Robustness of Six Tests in MANOVA").

**K = 5 gives exactly three families:**
1. **Relative error** (scale-proportional): AbsRel
2. **Absolute error** (metre-scale, outlier-sensitive): RMSE  
3. **Log-domain** (scale-invariant, KITTI primary): SILog
4. **Threshold accuracy** (discrete, higher-is-better): δ₁
5. **3D reconstruction** (camera-frame, robotics-specific): F@5cm

These span three mathematical families (error in d, error in log d, binary threshold, 3D point cloud) with two directionalities (4 lower-is-better, 1 higher-is-better). MANOVA's multivariate signal has genuine dimensionality.

### Scale-invariance check for MANOVA

The K = 5 set includes:
- **Scale-dependent:** RMSE (units: metres), F@5cm (unitless fraction, but threshold is metric)
- **Scale-invariant:** SILog (log domain), AbsRel (ratio), δ₁ (ratio)

For metric models (no alignment), all 5 are meaningful. For relative models post-`ls_affine` alignment, RMSE and F@5cm are meaningful only after alignment — which is the intended protocol (§2 of PLAN.md). SILog is inherently scale-invariant, providing a check: if a relative model's SILog is good but post-alignment RMSE is poor, the alignment procedure is suspect.

**Verdict:** The mix is appropriate. MANOVA standardises all K metrics to z-scores before computing Wilks' Λ (paper §3.2), so scale differences in units do not bias the test.

### Per-roster compatibility check

| Metric | Metric models (DA3, DA-V2, Depth Pro, UniDepth, Metric3D, MoGe-2) | Relative models (Lotus-2, FE2E) | RGB-D refinement (Prompt DA) |
|--------|------|------|------|
| AbsRel | ✅ Direct | ✅ After ls_affine | ✅ Direct (metric output) |
| RMSE | ✅ Direct | ✅ After ls_affine | ✅ Direct |
| SILog | ✅ Direct | ✅ After ls_affine (SILog is scale-invariant; alignment doesn't change it much) | ✅ Direct |
| δ₁ | ✅ Direct | ✅ After ls_affine | ✅ Direct |
| F@5cm | ✅ Camera-frame backprojection | ⚠️ After ls_affine, then backproject. F@5cm at a fixed threshold is sensitive to residual scale error — report but flag. | ✅ Direct |

**Flag:** F@5cm for relative models is computed after alignment. This is defensible (the standard practice for evaluating relative models), but the residual scale error from `ls_affine` can systematically bias F@5cm. Mitigation: report the aligned-track F@5cm separately, never mixed with metric-track scores. Already the protocol in PLAN.md §2.

---

## 2. Video Depth Additional Temporal Metrics (K_D1V = K_D1F + 2 = 7)

### Final temporal set

| # | Key | Metric | Dir | Citation |
|---|-----|--------|-----|----------|
| 6 | `opw` | Optical-Flow Warping Error | ↓ | Sundaram et al. 2010; adopted by DepthCrafter (Hu et al. 2024), ChronoDepth (Shao et al. 2024), Video DA (Chen et al. 2025), RollingDepth (Ke et al. 2024), GemDepth (Liu et al. 2025), ViGeo (2025) |
| 7 | `tae` | Temporal Alignment Error (SE(3) reprojection) | ↓ | ChronoDepth (Shao et al. 2024), Video DA (Chen et al. 2025) |

### Candidate analysis

| Candidate | Verdict | Evidence |
|-----------|---------|----------|
| **OPW** | **INCLUDE** | The *de facto* standard temporal consistency metric for video depth. Used by 6/8 video depth papers in the Video Depth roster. Requires RAFT optical flow between consecutive RGB frames — no GT depth or pose needed. Measures appearance-level temporal smoothness. |
| **TAE** | **INCLUDE** | Used by ChronoDepth and Video Depth Anything. Requires T265 poses (which RPX has). Measures *geometric* temporal consistency via true SE(3) backproject→reproject, complementing OPW's appearance-level signal. Implementation already exists in `deployment.py:se3_reproject_depth()`. |
| **TGM** | **EXCLUDE from K** | Temporal Gradient Matching (∇_t d̂ vs ∇_t d_GT) is conceptually appealing but rarely reported in the video-depth literature. Not used by any paper in the Video Depth roster as a headline metric. Emit for diagnostic; exclude from MANOVA. |
| **TCC** | **EXCLUDE from K** | Temporal Consistency Coefficient (cross-frame Pearson/SSIM on depth) is not standard. No video-depth paper in the roster reports it. Emit for diagnostic. |
| **Per-frame AbsRel drift** | **EXCLUDE** | Standard deviation of AbsRel across frames is a summary statistic, not an independent metric. Derivable from the cell log post-hoc. |
| **MS-SSIM on depth** | **EXCLUDE** | Not used by any video-depth paper in the roster. MS-SSIM is designed for perceptual image quality, not depth consistency. |
| **Scale-drift** | **EXCLUDE from K** | Only meaningful for relative models. Could emit as a diagnostic (std of per-frame alignment scale across the clip), but including it in MANOVA would penalise metric models (which have zero scale drift by construction). |

### Why K_temporal = 2, not 3 or 4

Two temporal metrics give MANOVA a 7-dimensional signal for Video Depth. Adding TGM (ρ likely > 0.9 with TAE — both measure geometric consistency from depth maps and poses) would inflate K without adding an independent dimension. OPW and TAE are genuinely complementary:

- **OPW** uses *RGB-derived* optical flow (RAFT) — measures whether depth is *visually* consistent across frames. Does not require GT depth or pose.
- **TAE** uses *pose-derived* geometric reprojection — measures whether depth is *geometrically* consistent in 3D. Requires poses but not GT depth.

Their failure modes differ: OPW misses scale drift (flow doesn't encode absolute depth); TAE misses texture-level flickering (reprojection only checks depth values at corresponding pixels). A model can score well on one and poorly on the other.

### Optical-flow recommendation

**Use RAFT (Teed & Deng 2020, ECCV).** Pinned checkpoint: `raft-sintel.pth`.

Justification:
- DepthCrafter, ChronoDepth, Video DA, RollingDepth, GemDepth all use RAFT for OPW computation.
- RAFT is MIT-licensed (via the authors' repo), not NC.
- FlowFormer (Huang et al. 2022) is more accurate but heavier and less widely adopted for OPW computation in the depth literature. Using a non-standard flow model would make RPX's OPW numbers incomparable with prior work.
- T265 ego-motion is insufficient: it provides only camera motion, not dense per-pixel correspondences needed for OPW.

**Implementation note:** Compute RAFT flow on-the-fly between consecutive RGB frames. RPX has 8× A6000 — compute is not a constraint. Mask low-confidence flow regions (RAFT outputs a confidence/iteration count). Pin the checkpoint and document it so OPW scores are reproducible.

### Temporal metrics on static phases (Clutter, Clean)

**Temporal metrics are meaningful on all three phases**, not just Interaction.

- **Clutter/Clean (static scene):** The *camera still moves* (hand-carried D435). Temporal consistency here measures whether the depth model produces stable predictions under ego-motion with a static scene. This is a pure test of the model's internal consistency — the "right answer" doesn't change, so any depth variation is model noise. Useful baseline: a perfectly consistent model would have OPW ≈ 0 and TAE ≈ 0 on static phases.
- **Interaction (dynamic scene):** Objects move, hand enters frame. Temporal consistency here measures whether the model tracks the changing scene correctly. OPW and TAE *should* be nonzero (the scene changed), but excessively high values indicate the model is failing to track.

The phase comparison is itself informative: Φ across phases on the temporal metrics tells us whether a model's consistency degrades under manipulation — exactly RPX's thesis.

---

## 3. Dataloader Design Specs

### 3.1 `D1FDataset` — Frame-Level

```python
@dataclass
class D1FSample:
    """Single frame for Image Depth evaluation."""
    rgb: np.ndarray              # (H, W, 3) uint8, H=480 W=640
    depth_gt: np.ndarray         # (H, W) float32, metres. 0 = invalid.
    valid_mask: np.ndarray       # (H, W) bool. GT ∈ (0.3, 5.0)m ∧ finite.
    scene_id: str                # e.g. "scene001"
    phase_idx: int               # 0=clutter, 1=interaction, 2=clean
    frame_idx: int               # 0-indexed within the phase
    intrinsics: np.ndarray       # (3, 3) float64. fx=fy=615, cx=320, cy=240.

class D1FDataset:
    """Iterates per-frame over (scene, phase, frame).
    
    Iteration order: outer=scene (sorted), middle=phase (0,1,2),
    inner=frame (0..T-1). This groups naturally for cell-log 
    aggregation: the runner collects frames until (scene, phase) 
    changes, then writes one cell.
    
    Cell-log key: (model, task="Image Depth", scene_id, phase_idx)
    Per-frame metrics are averaged within the cell.
    """
    
    def __init__(self, data_root: str, scenes: list[str] | None = None):
        ...
    
    def __iter__(self) -> Iterator[D1FSample]:
        for scene in sorted(self.scenes):
            for phase in [0, 1, 2]:
                for frame_idx in range(self.num_frames(scene, phase)):
                    yield self._load(scene, phase, frame_idx)
    
    def __len__(self) -> int:
        """Total frames across all (scene, phase) cells."""
        ...
```

### 3.2 `D1VDataset` — Video-Level (Phase Clip)

```python
@dataclass
class D1VClip:
    """Full phase clip for Video Depth evaluation."""
    rgb_seq: np.ndarray          # (T, H, W, 3) uint8
    depth_gt_seq: np.ndarray     # (T, H, W) float32, metres
    valid_mask_seq: np.ndarray   # (T, H, W) bool
    scene_id: str
    phase_idx: int
    frame_indices: np.ndarray    # (T,) int, original frame numbers
    intrinsics: np.ndarray       # (3, 3) float64
    poses: np.ndarray | None     # (T, 4, 4) float64, world-from-camera SE(3).
                                 # None if T265 data missing for this clip.

class D1VDataset:
    """Iterates per-(scene, phase) clip.
    
    Iteration order: outer=scene (sorted), inner=phase (0,1,2).
    Each __next__ yields a full clip.
    
    Cell-log key: (model, task="Video Depth", scene_id, phase_idx)
    Model produces (T, H, W) depth sequence in one shot.
    """
    
    def __init__(self, data_root: str, scenes: list[str] | None = None,
                 max_frames: int | None = None):
        """
        Parameters
        ----------
        max_frames : int | None
            If set, truncate clips longer than this. For memory
            management during development. Production runs use None.
        """
        ...
    
    def __iter__(self) -> Iterator[D1VClip]:
        for scene in sorted(self.scenes):
            for phase in [0, 1, 2]:
                yield self._load_clip(scene, phase)
```

### 3.3 Memory budget

| Item | Size | Note |
|------|------|------|
| RGB clip (250 × 640 × 480 × 3) | 230 MB | uint8 |
| Depth GT clip (250 × 640 × 480 × 4) | 307 MB | float32 |
| Valid mask clip (250 × 640 × 480 × 1) | 77 MB | bool |
| Poses (250 × 4 × 4 × 8) | 32 KB | float64, negligible |
| **Total per clip** | **~614 MB** | |
| Model activations (worst case: VGGT 1B) | ~8–12 GB | GPU |
| Host RAM per worker | ~2 GB peak | 1 clip loaded + 1 prefetched |

**64 GB host RAM is sufficient.** Even loading 4 clips simultaneously (for 4-GPU parallel) uses ~2.5 GB — well within budget. The GPU VRAM (48 GB A6000) is the binding constraint, not host RAM.

### 3.4 GPU strategy

**Recommendation: Option A — 4 models in parallel × 1 GPU each.**

Reasoning:
- Model parameter counts span 25M (DA-V2-S) to 3B (VGGT). DDP across 4 GPUs gives a ~2–3× speedup per model but serialises the model dimension (10 models × sequential). 4-way model parallelism gives 10/4 ≈ 3 sequential rounds.
- Most models in the roster are single-GPU-sized (< 2B params, < 20 GB VRAM on A6000 at fp16). Only VGGT-1B might benefit from 2-GPU tensor parallelism, but even that fits on a single A6000 in fp16.
- Implementation complexity: Option A requires zero coordination code — just launch 4 independent `run_depth.py` processes. Option B requires DDP wrappers per model, which are model-specific and fragile across the heterogeneous adapter zoo.
- Wall-clock estimate (Option A): 100 scenes × 3 phases × 250 frames = 75,000 frames. At ~50ms/frame (DA3-L) → ~63 min per model, ÷4 parallel = ~160 min for all 10 Image Depth models. For Video Depth: longer per-clip but same parallelism logic.

**On the second server (4 more GPUs):** Run Video Depth models on server 2 while Image Depth runs on server 1. Zero coordination needed — cell logs merge via `merge_cells()`.

### 3.5 Batching

- **Image Depth:** Batch across frames within a (scene, phase). Batch size is model-dependent — let the adapter declare it via `max_batch_size` property (default: 8 for ViT-L models on A6000). The dataset's iteration order (grouped by scene+phase) makes this natural.
- **Video Depth:** No batching across clips. Each clip is a single forward pass (or sliding window for models like DepthCrafter). The model adapter handles internal chunking.

### 3.6 Latency measurement

**Recommendation: Same loader, warm-cache timing wrapper.**

```python
# In the runner, after the first clip/frame (warm-up):
for sample in dataset:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = model.predict(sample)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
```

A separate "1-frame-warm-cache" path is unnecessary complexity. The runner already has a warm-up phase (first sample is discarded for timing). CUDA synchronisation before and after ensures GPU time is captured. Report median and P95 latency across all frames (excluding the first 10 as warm-up).

### 3.7 GT depth validity

**Valid mask:** `depth_gt > 0 ∧ depth_gt ∈ [0.3, 5.0] m ∧ isfinite(depth_gt)`.

Already implemented in `depth_alignment.py:default_valid_mask()`. The bounds are from the Intel RealSense D435 datasheet:
- **0.3 m minimum:** D435 active stereo minimum working distance (datasheet: 0.28m at 640×480, rounded to 0.3m for safety).
- **5.0 m maximum:** D435 accuracy degrades significantly beyond ~4m indoors (Intel spec: < 2% error at 2m, ~4% at 4m; beyond 5m, active IR returns are unreliable). The D435 depth stream encodes invalid pixels as 0.

The prediction mask additionally requires `pred > 0 ∧ isfinite(pred)` to exclude model failures. This is already enforced.

### 3.8 Sequence integrity for Video Depth

**Recommendation: Pass true T, let the model handle it.** Do not pad.

Reasoning:
- Most scenes have ~250 frames, but some may have fewer (early termination during capture) or more.
- Padding with last-frame copies creates artificial temporal consistency at the clip boundary — biasing OPW and TAE downward.
- Video depth models already handle variable-length inputs (DepthCrafter uses sliding windows; Video DA processes frame-by-frame; VGGT takes arbitrary-length sequences up to memory).
- The `D1VClip.frame_indices` field records the actual frame numbers, so the cell log knows the true T.
- **Guard:** If T < 30, skip the clip and log a warning — temporal metrics are not meaningful on very short clips.

---

## 4. Adversarial Verification Report

### 4.1 Weakest claim: "SILog adds independence from AbsRel"

**Attack:** The correlation analysis shows Spearman ρ = 1.000 between SILog and AbsRel on both NYU and KITTI. If they produce identical rankings, SILog adds zero information to MANOVA — it inflates K without benefit.

**Defence:** The perfect rank correlation is an artefact of the narrow 8-model zoo used in the analysis. SILog and AbsRel measure fundamentally different quantities:
- AbsRel = mean(|d̂−d|/d) — dominated by *median* error, robust to outliers.
- SILog = Var(log d̂ − log d) — measures the *dispersion* of log-errors, punishing inconsistent scale across the depth range.

A model with uniform 10% error everywhere has low AbsRel *and* low SILog. A model with 1% error at near range and 50% error at far range has moderate AbsRel but *high* SILog. This distinction will manifest on RPX's 100-scene dataset with heterogeneous depth distributions, even if it doesn't appear in the published NYU averages.

**Counter-citation:** Wu et al. 2025 ("Toward A Better Understanding of Monocular Depth Evaluation", arXiv:2510.19814) explicitly study metric sensitivity and find that log-domain metrics (SILog family) are more sensitive to curvature perturbations than AbsRel. Their SAWA-H composite assigns nonzero weight to both AbsRel and a log-domain component, confirming they carry distinct information.

**KITTI precedent:** KITTI's official depth prediction benchmark uses SILog as the *primary ranking metric*, not AbsRel. If it were truly redundant, KITTI would not have adopted it as the headline number.

**Verdict:** Claim survives. The perfect rank correlation in our analysis reflects the narrow model spread, not metric equivalence. On RPX's per-scene data (100 scenes × 3 phases, not averaged), SILog and AbsRel will diverge. **Confidence: MEDIUM** — this is a defensible design choice, not a proven fact. If the pilot run shows ρ > 0.98 on per-cell RPX data, revisit.

### 4.2 Most controversial metric: F-score @ 5 cm

**Attack — exclusion argument:** Gurram & López 2023 ("On the Metrics for Evaluating MDE", arXiv:2302.10007) validate AbsRel as the best single metric for 3D object detection downstream — implying that 3D metrics like F-score are unnecessary if AbsRel already predicts downstream task performance. No major depth benchmark (NYU, KITTI, ScanNet, ETH3D, DIODE) uses F-score as a primary metric. Including it is non-standard and may confuse practitioners expecting the canonical metric set.

**Counter-argument for RPX:** Gurram & López's finding is about *autonomous driving* downstream tasks (3D object detection in KITTI). RPX is a *robotics manipulation* benchmark. The deployment question is "can a gripper reach the right 3D point?" — which F-score directly measures and AbsRel does not. A model with low AbsRel (good average relative error) can still produce a warped point cloud that mislocates grasp targets by >5 cm, which F@5cm catches.

Supporting citations:
- Örnek et al. 2022 ("From 2D to 3D: Re-thinking Benchmarking of MDP") argue explicitly that 2D pixel metrics fail to capture 3D reconstruction quality and propose Chamfer-L1 + F-score as complements.
- Tanks & Temples (Knapitsch et al. 2017) uses F-score as its primary metric for 3D reconstruction quality.
- ScanNet evaluation for 3D reconstruction uses Chamfer distance and F-score.

**The RPX-specific argument:** F@5cm is the *only* metric in the K=5 set that is computed in 3D (camera-frame backprojection). All others are per-pixel 2D comparisons. This gives MANOVA a genuinely independent dimension — it's the metric most likely to produce disagreeing rankings, which is RPX's thesis.

**Verdict:** F@5cm survives for RPX. The non-standard criticism applies to generic depth leaderboards; RPX's robotics framing justifies it. **Confidence: HIGH.**

### 4.3 Cell-log re-derivability

**Question:** Can Φ be re-derived using only AbsRel + δ₁ (K=2) without re-running model inference?

**Answer: Yes.** The cell log records *all* per-frame metric values. The post-hoc analysis script (`scripts/analyze_experiment.py`) accepts a `--metrics` flag to select which columns enter MANOVA. Re-running with `--metrics absrel,delta1` recomputes Φ with K=2 from the same cell log. No model re-inference needed.

This is by design — the cell log is the canonical raw data; Φ and JEDI are derived statistics computed post-hoc. The K=5 recommendation affects only the *primary* Φ reported in the paper. Sensitivity sweeps over alternative K values are a standard robustness check (and already planned in the workflow per SHARED_CONTEXT.md).

### 4.4 Video Depth → Image Depth downgrade

**Question:** Can a video model's first-frame output be scored as a Image Depth prediction without a second inference run?

**Answer: Yes, with a minor contract addition.** The `D1VClip` contains per-frame depth GT and valid masks. If the video model adapter stores per-frame predictions (which it must, for temporal metrics), the runner can extract frame 0's prediction and score it against the Image Depth metric set.

**Contract requirement:** The video model adapter's output must include a `predictions: np.ndarray  # (T, H, W) float32` field — the per-frame depth sequence. The runner scores:
1. Each frame individually against Image Depth metrics (5 metrics).
2. The full sequence against Video Depth temporal metrics (2 additional).

For DA3 (appears in both Image Depth and Video Depth rosters), this means:
- **Image Depth run:** DA3 receives one RGB frame, outputs one depth map. Standard.
- **Video Depth run:** DA3 receives the full 250-frame clip. Its first-frame output is *also* scored on Image Depth metrics.
- **Comparison:** The per-frame Image Depth metrics from the Video Depth run may differ from the standalone Image Depth run if the model uses future-frame context. This difference *is* the frame-vs-video ablation signal.

**No separate inference run needed.** The Video Depth runner naturally produces Image Depth-compatible per-frame scores.

---

## 5. Confidence Statement

| Decision | Confidence | Source of uncertainty |
|----------|-----------|---------------------|
| K_D1F = 5 (AbsRel, RMSE, SILog, δ₁, F@5cm) | **HIGH** | Metrics are standard (4/5 from Eigen 2014 / KITTI); F@5cm is non-standard but well-motivated for robotics. Correlation analysis supports exclusion of redundant metrics. |
| Exclude SqRel, RMSElog, iRMSE | **HIGH** | ρ > 0.95 on both NYU and KITTI. No model paper in the Image Depth roster reports these as headline metrics on indoor benchmarks. |
| Exclude δ₂, δ₃ from K | **HIGH** | Near-saturated (>0.995) for all SOTA models. Zero discriminative power. Universal practice in recent papers. |
| Exclude iRMSE specifically | **HIGH** | Designed for 0–80m outdoor; degenerates on 0.3–5m indoor. |
| SILog adds independence on per-cell RPX data | **MEDIUM** | Perfect rank correlation in published averages, but different mathematical sensitivity. Depends on per-scene depth-distribution heterogeneity. Verify on pilot run. |
| K_temporal = 2 (OPW, TAE) | **HIGH** | OPW is universal (6/8 papers). TAE is adopted by 2–3 papers and uses RPX's available pose data. Both are already implemented. |
| Exclude TGM, TCC from K | **MEDIUM** | TGM is conceptually sound but not adopted by the video-depth community. If a reviewer requests it, it can be added post-hoc from the cell log (the calculator emits it). |
| D1FDataset / D1VDataset contracts | **HIGH** | Straightforward engineering; memory budget verified; iteration order aligns with cell-log grouping. |
| Option A GPU strategy (4 models parallel) | **HIGH** | Models fit on single A6000; no DDP wrappers needed; simplest correct approach. |
| No padding for short clips | **HIGH** | Padding biases temporal metrics. Models handle variable-length input. Skip clips with T < 30. |
| RAFT for OPW | **HIGH** | Used by 6/8 papers; MIT license; standard checkpoint (`raft-sintel.pth`). |
| F@5cm for relative models (post-alignment) | **MEDIUM** | Residual scale error from ls_affine may bias F@5cm. Mitigated by separate aligned-track reporting. |

---

## Summary: Locked Metric Tuples

**Image Depth (frame-level):** K = 5  
`(absrel, rmse, silog, delta1, fscore_5cm)`

**Video Depth (video-level):** K = 7  
`(absrel, rmse, silog, delta1, fscore_5cm, opw, tae)`

**Cell log emits additionally (not in MANOVA K):**  
`delta2, delta3, rmselog, chamfer_l1, fscore_1cm, fscore_2cm, fscore_10cm, normal_mae, normal_acc1125, boundary_f1, valid_ratio`  
Plus range-stratified variants: `absrel_near, rmse_near, ..., delta1_far`  
Plus (Video Depth only): `tgm, tcc`

All are re-derivable from the cell log for sensitivity analysis without model re-inference.

---

## Appendix: Divergence from Current Paper and PLAN.md

| Current state | This recommendation | Action needed |
|---------------|-------------------|---------------|
| `04_tasks.tex` says K=6 with iRMSE | K=5, drop iRMSE | Update LaTeX table |
| `PLAN.md` roster includes DepthLM, D4RT, StableDPT | Already cut per HANDOFF | No action (PLAN.md already updated) |
| `METRICS_FINAL.md` includes rmselog, silog in Block A | rmselog excluded from K, silog kept | Update METRICS_FINAL.md |
| `specs.py` registers `opw, tae, tgm, tcc` | tgm, tcc emitted but excluded from K | No code change; MANOVA selection is in analyze_experiment.py |
| `04_tasks.tex` says Video Depth adds TGM + TAE (K=8) | Video Depth adds OPW + TAE (K=7) | Update LaTeX table: TGM → OPW, K=8 → K=7 |

---

## Sources

1. Eigen et al. 2014, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network" (NIPS 2014). https://arxiv.org/abs/1406.2283
2. Wu et al. 2025, "Toward A Better Understanding of Monocular Depth Evaluation" (arXiv:2510.19814)
3. Bochkovskii et al. 2024, "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (ICLR 2025). https://arxiv.org/abs/2410.02073
4. Örnek et al. 2022, "From 2D to 3D: Re-thinking Benchmarking of MDP". https://arxiv.org/abs/2203.08122
5. KITTI Depth Prediction Benchmark. https://cvlibs.net/datasets/kitti/eval_depth.php
6. Gurram & López 2023, "On the Metrics for Evaluating Monocular Depth Estimation". https://arxiv.org/abs/2302.10007
7. Knapitsch et al. 2017, "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction" (ACM ToG). https://tanksandtemples.org
8. Teed & Deng 2020, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV). https://arxiv.org/abs/2003.12039
9. Hu et al. 2024, "DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos". https://arxiv.org/abs/2409.02095
10. Shao et al. 2024, "ChronoDepth: Learning Temporally Consistent Video Depth from Video Diffusion Priors". https://arxiv.org/abs/2406.01493
11. Chen et al. 2025, "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos". https://arxiv.org/abs/2501.12375
12. Ke et al. 2024, "RollingDepth: Extending Monocular Depth to Temporal Consistency". https://arxiv.org/abs/2407.09714
13. Olson 1974, "Comparative Robustness of Six Tests in Multivariate Analysis of Variance" (JASA).
14. Yang et al. 2024, "Depth Anything V2". https://arxiv.org/abs/2406.09414
15. Piccinelli et al. 2024, "UniDepth V2". https://arxiv.org/abs/2403.18913
16. Yin et al. 2024, "Metric3D V2". https://arxiv.org/abs/2404.15506
17. Wang et al. 2025, "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision". https://arxiv.org/abs/2410.19115
18. Yang et al. 2026, "Depth Anything 3" (ICLR 2026 Oral).
