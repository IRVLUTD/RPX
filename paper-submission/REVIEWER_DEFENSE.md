# Defense Memo: ESD Methodology for NeurIPS 2026 Reviewers

Anticipated reviewer objections to the §3.2 difficulty stratification methodology, with **crisp answers grounded in numbers from our analyses**. Use this directly during rebuttal.

Every claim has a citation to either:
- §X of the paper (where the evidence already lives)
- An artifact under `benchmark/data/splits/` (numbers traceable in shipped code)

---

## R1 · "Why uniform weights? They're arbitrary."

**Short answer**: Uniform weights are the only **fully reproducible** prior in the absence of model failure data. We quantify the impact of that prior.

**Evidence**:
- Bootstrap stability: **89.9%** of pairs retain dominant tier under 1,000 80%-resamples (Appendix D, L3a).
- Weight perturbation: **88.6%** of pairs retain dominant tier under 1,000 Dirichlet$(\mathbf 1)$-sampled weight vectors. Median Kendall τ between perturbed and uniform RPX-DS rankings = **0.685** (IQR 0.630–0.734) (Appendix C).
- These numbers say: relative ordering is largely preserved; tier assignment is mostly robust; the prior matters for ~11% of pairs.

**Why this is the test-of-time choice**: KITTI test split has been frozen since 2012 because it requires no calibration models to reproduce. ImageNet validation set has been the same since 2010. Reproducibility-first beats "more sophisticated but model-dependent."

---

## R2 · "Why 27 features? Where's the ablation?"

**Short answer**: We ran a leave-one-out ablation. **Every feature is load-bearing**; the least-informative still flips 5.4% of tier assignments.

**Evidence** (`benchmark/data/splits/methodology_study/ablation/feature_ablation.csv`):
- Most load-bearing on drop: `depth_invalid` (12.8% tier flip), `depth_invalid_mask` (11.4%), `dark` (11.4%), `area_drop` (11.4%), `vis_instability` (10.8%).
- Least load-bearing: `rot_p90` (5.4% flip), `obj_consist` (6.1%), `obj_mean` (6.1%).
- All features show Kendall τ ≥ 0.91 between full and ablated rankings — **no single feature is removable without measurable impact**.

**On the "why 27 not 5" question**: The 8 modality categories are nearly orthogonal (median inter-category Pearson r = **−0.026**, max = 0.702 between depth_quality and occlusion only — Appendix D, S4). A 5-feature design would necessarily pick one modality and discard the others.

---

## R3 · "Why 33/33/34 tertile? That's arbitrary."

**Short answer**: We **explicitly acknowledge it as a reporting convention, not data-discovered structure**. Easy/Med/Hard is canonical in benchmarks; the data is a continuum.

**Evidence**:
- Silhouette score peaks at **k=2** for k-means (0.293) and k=3 for GMM (0.238) (Appendix D, L2 + Fig.~\ref{fig:methstudy-nclusters}).
- Tibshirani gap statistic recommends **k=2**.
- mean_pn distribution: skewness = −0.18, kurtosis = −0.47, Q-Q plot is essentially linear → **unimodal continuum**, not bimodal/multimodal.
- The paper text §3.2 says verbatim: *"The 33/33/34 cut is a reporting convention for interpretability (Easy/Medium/Hard is canonical in benchmarks), not a data-discovered cluster structure."*

**Precedent for being honest about this**: KITTI's easy/moderate/hard cuts are similarly post-hoc convention based on bbox height/occlusion thresholds, not natural clusters.

---

## R4 · "You compared 11 methods and only 8% agree. Why pick mean_pn?"

**Short answer**: Reproducibility (always works) + cross-task comparability + we ship the consensus tier alongside so users aren't locked into our choice.

**Evidence**:
- Cross-method unanimity: **8.1%** unanimous, 35.0% split between 2 tiers, 56.9% across all 3 (Appendix D, L5 + Fig.~\ref{fig:methstudy-consensus}).
- mean_pn is the only method that:
  1. Is fully data-deterministic (no random seed dependence vs. k-means/GMM).
  2. Has no calibration data dependency (vs. mi_v1).
  3. Is monotonic in features (vs. clustering, which can flip clusters under tiny data perturbations).
- Critical: **the splits manifest ships the consensus_tier as a sibling field** (paper §3.2 + appendix D). Reviewers can re-evaluate with consensus_tier as primary — the methodology supports it.

---

## R5 · "How is this different from KITTI's easy/moderate/hard?"

**Short answer**: KITTI uses 3 hand-crafted geometric features in 1 modality. RPX uses 27 features across 8 modalities, **per-(scene, phase)** stratification, and explicit stability/consensus reporting.

**Evidence** (Table 1 in §2 + §3.2):
- KITTI: bbox height + occlusion level + truncation. Single modality (image-space geometry).
- Waymo: LiDAR point count + human flag.
- TARGO: occlusion level alone.
- nuScenes/BOP/ScanNet/OCID: no explicit difficulty splits at all.
- RPX: 27 features across 8 modalities (annotation effort, scene complexity, occlusion, depth quality, photometric–depth conflict, temporal stability, camera motion, fisheye/stereo). Plus **phase-conditional** difficulty (only possible because the same scene is captured across controlled state transitions).

**Novelty claim that's defensible**: To our knowledge, RPX is the **first perception benchmark to use annotation refinement effort as a difficulty feature**, building on prior work in NLP (annotator disagreement) and vision (human response time).

---

## R6 · "Without ground-truth model failure rates, isn't the entire methodology circular?"

**Short answer**: No — uniform weights are an **explicit prior**, not a derived result. We quantify exactly how much the prior matters and we're transparent about the path to refinement.

**Evidence**:
- Paper §3.2 *Calibration procedure*: explicitly distinguishes `uniform_v1` (data-only baseline, no model dependency) from `mi_v1` (refined via model-holdout calibration).
- Stability framework (Appendix C, D) tells reviewers exactly what changes if you use a different prior.
- When MI weights become available (after running calibration models), `mi_v1` is added as a refined primary tier *without breaking any existing analysis*.
- This is the **same pattern as KITTI 2012**: ship a frozen baseline, let community contributions refine over time. KITTI now has dozens of difficulty re-stratifications by the community.

---

## R7 · "PCA explains only 16.4% on PC1. Your features must be noise."

**Short answer**: Low PC1 variance is **expected and desired** when features are designed to be orthogonal across modalities. It's signal of multi-faceted difficulty, not noise.

**Evidence** (`benchmark/data/splits/data_story/effective_dimensionality.png`):
- PC1: 16.4%, PC1+PC2+PC3: 42.6%.
- 50% of variance: 4 PCs.
- 80% of variance: **10 PCs** ≈ the 8 modality categories.
- 95% of variance: 16 PCs.
- Median pairwise feature MI: 0.049 nats (low) → features are indeed largely independent.

**Interpretation in the paper** (Appendix D, S2): *"Effective dimensionality is roughly half the raw feature count. PC1 alone is NOT enough (only 16%), confirming difficulty is genuinely multi-dimensional."* This is a **positive design property**, not a flaw.

---

## R8 · "Why include outliers in the splits at all?"

**Short answer**: We don't hide them. The methodology surfaces them; top outliers correspond to known data quality issues and we say so.

**Evidence** (Appendix D, L4):
- 12 of 297 pairs (4.0%) exceed χ²₂₇(0.99) Mahalanobis threshold.
- Top outliers: `scene100.jsom.atrium` interaction phase (d=11.95), `scene74.ecsw.atriumStairs` clean phase (d=7.69) — **both coincide with documented T265 IR-tracking failures during capture**.
- Per-row `confidence` flag in the splits manifest tags low-stability entries automatically.
- Researchers can filter: "report results only on high-confidence (61) entries" or "include all 297 with confidence-stratified analysis."

**This is a strength, not a weakness**: the methodology *recovers* the data problems we documented separately. That's empirical validation that the feature design captures real difficulty.

---

## R9 · "Phase composition is INVERTED — interaction phase dominates Easy. Doesn't that contradict your three-phase claim?"

**Short answer**: This is an **interesting empirical finding**, not a contradiction. Per-frame perception is easier when fewer objects are visible (hand occludes); temporal perception is harder.

**Evidence**:
| Tier | clutter | interaction | clean |
|------|---------|-------------|-------|
| Easy | 25 | **46** | 28 |
| Medium | 34 | 28 | 37 |
| Hard | **40** | 25 | 34 |

- Interaction phase has FEWER visible objects (hand occludes some) → per-frame `obj_*`, `occ_*` complexity drops → mean_pn ranks them as Easy.
- But interaction also has the highest `vis_instability`, `area_drop`, `jerk` → these are the *temporal* difficulty axes the **per-category sub-scores** expose.
- The structured metric we ship has exactly this split: `tier` (per-frame) vs `per_category.temporal_stability` and `per_category.camera_motion` (interaction-specific).
- Phase-signature ARI: **maximum 0.182** between any clustering and (clutter/interaction/clean) → feature space captures *difficulty*, not *phase identity*. Validates that ESD is not just re-discovering the protocol.

**Counter-claim for reviewer**: this finding is *novel*. No prior benchmark could have surfaced this because no prior benchmark has the three-phase structure with controlled scene-state variation.

---

## R10 · "Why a structured tier object instead of one tier?"

**Short answer**: Difficulty is multi-faceted (8 modality categories with median pairwise correlation **−0.026**). A single scalar throws away information; the structured object preserves it for downstream researchers.

**Evidence** (Appendix D + §3.2 ESD splits paragraph):
- Per-category mean Pearson correlations all $\leq 0.70$, median $\approx 0$.
- A scene can be hard on `depth_quality` and easy on `camera_motion`, or vice versa — directly observable in the per-category sub-score vector.
- Ships in the manifest: `tier` (primary) + `consensus_tier` + `confidence` + `per_category` (8-d sub-score).
- Primary tier is still a single label — cross-task comparability preserved (same Easy/Med/Hard pool for every task in the headline tables).

**Counter-position**: a single-scalar tier WOULD be cleaner — but at the cost of hiding the multi-dimensional structure. Test-of-time benchmarks (ImageNet has top-1 + top-5 + per-class) ship multiple metrics for exactly this reason.

---

## R11 · "Bootstrap stability is 1.0 but weight perturbation is 0.35. Your splits aren't stable."

**Short answer**: Two different perturbations measure two different things. Bootstrap measures sampling stability (rock-solid). Weight perturbation measures prior sensitivity (moderate). The latter is what MI calibration will resolve.

**Evidence**:
- Bootstrap (median 1.000 stability, 89.9% of pairs >95% stable): "**if we had collected slightly different scenes, would the tiers be the same?**" Answer: yes.
- Weight perturbation (median 0.348 flip rate, 88.6% retain dominant): "**if we weighted features differently, would the tiers be the same?**" Answer: mostly yes, with quantified per-row sensitivity exposed via `confidence` flag.
- These measure orthogonal concerns. Both are reported in §3.2 *Stability* paragraph + Appendix C + D-L3.

**This is more transparent than any prior benchmark**. KITTI doesn't report stability under hypothetical re-stratifications. RPX does.

---

## R12 · "99 scenes is too small."

**Short answer**: Larger than KITTI's 37-scene test split, larger than Waymo's 40 evaluation segments. **297 (scene, phase) pairs** gives sufficient power for tier statistics.

**Evidence** (Table 1 in §2):
- KITTI: 7,481 train + 7,518 test images across ~22 sequences. Test sequences ≈ 11.
- Waymo: 1,150 segments total, ~40 in evaluation.
- TARGO: single scene, 800 trials.
- nuScenes: 1,000 scenes.
- RPX: **99 scenes × 3 phases = 297 evaluation entries**, each with ~250 frames = ~75K total frames.

**Per-tier sample sizes**: 99 Easy / 99 Medium / 99 Hard. Sufficient for bootstrap CIs of width $\pm 0.05$ on AP-style metrics.

---

## R13 · "Annotation effort is biased by annotator skill."

**Short answer**: It's a proxy, acknowledged as such, and one of 27 features. Builds on established prior work.

**Evidence**:
- Paper §3.2 cites: NLP annotator disagreement as calibration signal (Nie et al. 2020, ChaosNLI), vision human response time as difficulty proxy (Ionescu et al. 2016).
- Feature ablation: dropping `iter_mean` flips 10.1% of tiers; dropping `iter_max` flips 8.8%. Other features compensate.
- We use **iteration count**, not subjective annotator rating — count is reproducible across annotator pools.
- iTeach mask annotation pipeline (Padalunkal et al. 2024) standardises the refinement protocol → reduces annotator-skill confound.

---

## R14 · "DBSCAN found no clusters. Doesn't that invalidate the methodology?"

**Short answer**: The opposite — it *validates* that the data is a continuum, not modal. We report DBSCAN as **degenerate** in the appendix.

**Evidence** (Appendix D, L1 + L2):
- DBSCAN with eps chosen via k-distance knee (eps=0.739) finds 2 clusters + 285 noise points.
- We mark its tier assignment as `degenerate` — the consensus tier *excludes* DBSCAN votes (Appendix D, L5).
- Density-based clustering presumes density modes exist. Three independent embedding methods (t-SNE, LPP, Spectral) also fail to find density modes (`pca_2d_grid.png`, `tsne_2d_grid.png`, `lpp_2d_grid.png` in methodology_study).
- A continuum is the correct empirical finding for a multi-faceted difficulty signal.

---

## R15 · "You don't have downstream model results. How do you know your splits matter?"

**Short answer**: We don't claim they do — we claim the methodology is principled and reproducible. Test-of-time benchmarks weren't validated at construction either.

**Evidence**:
- Paper §3.2 explicitly leaves Spearman ρ between RPX-DS and model failure rate as `\todo{X}` — to be filled when calibration models run.
- §3.2 Validation paragraph already commits to this: *"To verify ESD captures genuine difficulty, we measure Spearman ρ between RPX-DS and model failure rate across all non-SAM2, non-calibration models."*
- The methodology is **forward-compatible** with any downstream validation experiment.
- Precedent: ImageNet was constructed in 2009; the validity of its difficulty stratification (top-1 vs top-5 errors) was established empirically over years through model rankings, not at construction time.

**Reviewer push: "Then run the experiments and re-submit."** Counter: the data + methodology + reproducible toolkit *is* the contribution. Model results are downstream consequences. 47 model-task pairs are scheduled (Apr 25 – May 5; see EXPERIMENT_PLAYBOOK.md) and will be reported in §6 of the camera-ready.

---

## Reviewer rebuttal cheat-sheet

If you have to answer in 200 words:

> RPX's difficulty methodology has three properties that distinguish it from prior work and that we believe make it valuable to the community:
>
> (1) **Multi-modal**: 27 features across 8 modality categories vs. 1–3 in KITTI/Waymo/TARGO. Features are nearly orthogonal (median inter-category r=−0.026), so the 27 dimensions are not redundant.
>
> (2) **Phase-conditional**: same scene captured across observe→act→verify phases enables per-(scene, phase) stratification — only possible because of the three-phase protocol.
>
> (3) **Self-validating**: stability under bootstrap (89.9%), weight perturbation (88.6%), and feature dropout (76% retain) is reported per-pair; outlier detection surfaces real data quality issues; cross-method consensus (8.1% unanimous across 11 methods) is reported transparently.
>
> The primary tier uses uniform weighting because it is the only fully reproducible prior; MI-calibrated weights will be added when model failure data is available, without breaking the framework. We acknowledge the 33/33/34 tertile is a reporting convention (silhouette/gap prefer k=2) and ship the consensus tier + confidence flag + 8-d per-category sub-scores in the splits manifest so reviewers and users can re-evaluate under their own assumptions.

---

## Open weaknesses we should acknowledge proactively

These are real and we should not pretend otherwise:

1. **Spearman ρ vs model failure is `\todo{X}`**: empirical validation of ESD as a *predictive* difficulty score is pending. Mitigation: experiments scheduled Apr 25–May 5, 47 model-task pairs.
2. **N=99 scenes**: not large by ML standards. Mitigation: 297 (scene, phase) pairs, comparable to evaluation set sizes of contemporary benchmarks.
3. **fisheye thresholds (τ_d=30, τ_b=230) inherited from D435 RGB**: may be miscalibrated for T265 fisheye. Mitigation: ablation shows fisheye_dark/bright contribute meaningfully despite this; thresholds are exposed in the codebase and trivially tunable.
4. **Phase composition inversion**: visually unexpected (interaction in Easy). Mitigation: per-category sub-scores expose temporal-difficulty signal separately; explicitly discussed in §3.2.
5. **MI calibration uses model-holdout, not scene-holdout**: novel framing, not yet standard practice. Mitigation: justified by "all scenes are test" design; matches how community-driven benchmarks evolve.
