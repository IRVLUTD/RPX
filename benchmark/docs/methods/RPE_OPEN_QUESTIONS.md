# Relative Camera Pose Estimation: Open Questions & Decision Log

**Status**: OPEN — paper-survey track to finalize the 10-model SOTA cut for the launch sweep.
**Date created**: 2026-05-07
**Companion**: [`relative_pose_models.md`](relative_pose_models.md) carries the full 14-model survey across 4 categories.

---

## 1. The single open decision

The existing survey in `relative_pose_models.md` lists **14 candidate models** spanning four categories (direct pose regression, feature-matching + solver, classical, RGBD). The launch sweep targets **10 models** matched to the depth axis (which ships 19, but the depth zoo has more SOTA churn). We need a curated 10-of-14 (or 10-of-N, picking up any 2025-2026 papers the existing survey missed).

**Selection criteria** (in priority order):
1. **Reproducibility cut** — code public *and* checkpoints publicly downloadable today (matches the depth-zoo discipline). One-URL-per-adapter.
2. **Recency** — prefer 2024+ unless a pre-2024 model is the established baseline the field still cites.
3. **Category coverage** — keep at least 1 representative per (regression, matching+solver, classical, RGBD) so reviewers can compare paradigms.
4. **Robotics relevance** — small-baseline indoor pairs at 10 Hz (paired-stride 5 = ~0.5 s); models tuned for wide-baseline / mapping may underperform regardless of accuracy.

## 2. My proposed 10-of-14 cut (paper-survey track to confirm or revise)

| # | Model | Why include | Risk |
|---|---|---|---|
| 1 | **Reloc3r** (CVPR 2025) | Current SOTA regression, 25 ms inference, public ckpt. | Fresh — minimal third-party validation. |
| 2 | **DUSt3R** (CVPR 2024) | Foundational; baseline every newer method compares to. | Slow (joint 3D pointmap regression). |
| 3 | **MASt3R** (arXiv 2024) | Matching-aware DUSt3R, better correspondences. | Same speed envelope. |
| 4 | **FAR** (CVPR 2024) | Hybrid regression + matching; Niantic MapFree benchmark. | None obvious. |
| 5 | **SRPose** (ECCV 2024) | Sparse-keypoint, robust to varying intrinsics — relevant for our D435+T265+GoPro multi-rig. | None obvious. |
| 6 | **NOPE-SAC** (TPAMI 2023) | Neural-guided RANSAC; widely cited robustness baseline. | 2023 — borderline on "recency" criterion. |
| 7 | **MicKey** (CVPR 2024 Oral) | Metric keypoints, gives metric-scale pose directly — robotics-relevant. | None obvious. |
| 8 | **LoFTR** (CVPR 2021) | Standard matching+solver baseline; field still cites. | 2021 — older, but irreplaceable as baseline. |
| 9 | **OpenCV E-matrix** (classical) | SIFT/ORB + `findEssentialMat()` + `recoverPose()`. The "no learning" floor. | Will likely lose; that's the point. |
| 10 | **ICP / Colored ICP (Open3D)** (RGBD) | Classical RGBD baseline; pairs naturally with the dataset's depth modality. | Only 1 RGBD slot — drops SparsePlanes. |

**Cut from the 14**:
- **8-Point ViT** (3DV 2022) — superseded by FAR and Reloc3r in the same regression family.
- **SuperGlue** (CVPR 2020) — superseded by LoFTR + MicKey.
- **SparsePlanes** (ICCV 2021) — narrow scene-prior bet; ICP pair is broader.
- **(swap) Colored ICP / ICP** — pick one; default to Colored ICP because RPX RGB is well-calibrated.

## 3. The actual ask: paper-survey track

**Please confirm or revise** the 10-model cut above. Specifically:

- [ ] **Did the survey miss any 2025-2026 SOTA?** Candidates I'd want sanity-checked against arXiv 2025-Q4 / 2026-Q1: VGGT (Meta, 2024-end) — does the depth-only head produce relative pose? RoMa (CVPR 2024) — dense matching, would replace LoFTR? "Pose Anything" / "Pose-Foundation" style 2025 papers if any exist with public checkpoints.
- [ ] **Any model in the 14 that should NOT be in the launch sweep?** Reasons could be: license incompatible with our redistribution, checkpoint silently moved, transitive deps too heavy.
- [ ] **Confirm the RGBD slot** — is 1 slot enough, or should we keep ICP + Colored ICP + SparsePlanes as a 3-slot RGBD subgroup so the paper has a fair RGBD-vs-RGB comparison?
- [ ] **Confirm the classical slot** — is OpenCV's `findEssentialMat()` the right "no-learning floor", or should we also include 5-point with RANSAC-MAGSAC for a stronger baseline?

Once confirmed, the implementation pass mirrors the depth zoo:
- `scripts/pose_models/<adapter>.py` per model (HF-pipeline-friendly ones share a generic class à la `HFDepthEstimationAdapter`).
- `scripts/pose_models/__init__.py` with `MODEL_REGISTRY` + `MODEL_DISPLAY_NAMES`.
- `scripts/run_relative_pose.py` analogous to `run_depth.py` (registry-driven, `--save-predictions` / `--upload-to-box`).
- Pose-specific comprehensive metrics: rotation_error_deg, translation_angular_error, AUC@5°/10°/20°, per-pair-stride breakdown, RGBD vs RGB stratification.
- DRS aggregator parameter pass: `compute_sweep_drs` is task-agnostic; the sweep script just needs `--task relative_pose`.

## 4. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-07 | Target 10 models for the launch sweep (vs 14 in the full survey) | Match the depth axis's reproducibility-cut discipline + keep the sweep tractable on team compute. |
| 2026-05-07 | Drop SuperGlue + 8-Point ViT + SparsePlanes from the launch sweep | Each is dominated by a more-recent same-category entry. |
| (open)     | Final 10-of-14 list | Awaiting paper-survey track sign-off. |

---

When this list is finalized, append a `## 5. Resolution` section here and note the merge commit that lands the corresponding adapters under `scripts/pose_models/`.
