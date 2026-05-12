# Active Perception — RPX Task #11 (proposal)

> **Status.** Methodology proposal, no code yet. Drafted 2026-05-11.
> Tracks against the no-DRS / three-axis policy in
> [`../../SHARED_CONTEXT.md`](../../SHARED_CONTEXT.md).

## Problem

Most perception evaluations score a model on static frames. For an
embodied system that controls its own camera, a second question
matters: *given what it has seen so far, where should it look next?*
The active-vision / next-best-view (NBV) literature has asked this for
two decades, mostly within method papers rather than as part of
general perception benchmarks. RPX's structural data — pose-stamped
trajectories (D435 + T265), per-frame instance masks, three-phase
scene states across 99 scenes — supports adding it as an evaluation
task without new data collection.

## Task contract

| | |
|---|---|
| **Task key** | `active_perception` |
| **Input** | `K` context views (`rgb` + `depth` + 4×4 pose), camera intrinsics, optional task-goal string |
| **Output** | Predicted next-best pose `T̂_next ∈ SE(3)` + optional utility scalar |
| **`K`** | Swept ∈ `{2, 4, 8, 16}` — matches the NVS context-density sweep |

The model is *not* given the remaining-frame pool. It must propose
`T̂_next` purely from the K context views and the task goal.

## Oracle definition

From the **remaining frames** in the same `(scene, phase)` — the
unseen poses on the trajectory — pick the pose `T*` that maximises one
of three objectives. The paper reports all three; the headline metric
uses **Coverage-NBV** because it needs no model-specific uncertainty
and is fairest across the model slate.

| Objective | What it picks | Why |
|---|---|---|
| **Coverage-NBV** | pose maximising surface-area of scene points unseen by any context view | Purely geometric; reproducible from depth + poses alone |
| **Uncertainty-NBV** | pose where a reference depth model's confidence is lowest | Tests "do I look where I'm worst?" |
| **Task-NBV** | pose with the most task-relevant pixels (e.g. cup-mask area for "find cup") | Semantic; needs masks + grounding labels |

All three are deterministic and seed-locked at scene-prep time. No
model sees them during evaluation.

## Metrics (three RPX axes)

### Axis 1 — Task Performance

- **`pose_geodesic_deg(T̂_next, T*)`** — primary. Geodesic angle in SO(3)
  + L2 translation, reported jointly (`rot_deg`, `trans_m`).
- **`PUS_active(T̂_next, T*)`** — downstream utility: render via the
  NVS slate's best model at both poses, compare depth/seg metrics.
  Ratio in `[0, 1]`. The headline embodied number — a model that
  picks a pose 30° off but where downstream depth still works gets
  partial credit.

### Axis 2 — Scene-change robustness

- **Cross-phase active perception**: context from `clutter`, oracle
  from `clean`. Tests *"can the model still pick a good view after
  the scene rearranges?"*
- **K-curve**: PUS as a function of `K ∈ {2, 4, 8, 16}`. A robust
  model improves monotonically; a fragile one plateaus or oscillates.

### Axis 3 — Compute cost

- Params, FLOPs, latency to *propose* a single pose (cheap operation
  even for large models — a few GFLOPs is the expected order).

**No composite.** As elsewhere in RPX, the three axes are reported
separately; rankings will likely disagree (e.g. a VLM baseline may
score high on Task-NBV PUS but be 100× slower on Axis 3).

## Sample generation

```
For each (scene, phase, K) over the K-sweep:
    1. Use NVSPairGenerator's context-view selection (already
       deterministic + seed-locked).
    2. Remaining frames in the phase form the oracle pool.
    3. Compute T* per objective once at scene-prep (cached).
    4. Emit ActivePerceptionSample with:
         - context_rgb_paths, context_depth_paths, context_pose_paths
         - K, intrinsics
         - oracle_pose_coverage, oracle_pose_uncertainty, oracle_pose_task
         - task_goal (optional grounding text from RPX's VG annotations)
         - difficulty (passed through from ESD split)
```

Scale estimate: 99 scenes × 3 phases × 4 K-values × 25 target poses ≈
**29 700 samples** at full scale; ~2 000 for the easy split.

## Baselines (PR-D scope)

| Baseline | What it does | Compute |
|---|---|---|
| `random_pose` | Sample uniformly from the trajectory | None (floor) |
| `farthest_point` | Pose with largest L2 translation from context centroid | None (geometric heuristic) |
| `entropy_max` | Render uncertainty at N candidate poses with a fixed depth model, pick highest entropy | One forward / candidate |
| `vlm_prompt` *(optional)* | Prompt a VLM with context tiles + task goal, parse a pose suggestion | API / large LM (Axis-3 expensive) |

Slot for **learned NBV models** from the active-vision lit follows the same
adapter pattern as `nvs_models/` and `pose_models/`.

## Why RPX is well-suited

| Asset | Used for |
|---|---|
| 200 Hz 6-DoF VIO poses tied to RGB-D | computing `pose_geodesic` against a real trajectory |
| Three-phase capture of same scene | cross-phase NBV (Axis-2 metric) |
| Per-frame instance masks | Task-NBV objective |
| Visual grounding text (post-VQA) | task-goal-conditioned NBV |
| ~75K candidate poses across 297 (scene, phase) | benchmark-scale samples |

Comparable datasets cover subsets of these but, to our knowledge, not
all four together: ScanNet has 6-DoF poses and meshes but a single
scene state per capture; Habitat-Matterport has poses + segmentation
but is synthetic / rendered; nuScenes has dense trajectories on the
driving setting but no controlled scene-state variation. RPX is a
natural fit — we are *not* claiming uniqueness, only suitability.

## Motivation (hypothesis, not prediction)

If embodied foundation models continue converging on shared VLA-style
backbones, the evaluation gap between *dense single-frame prediction*
and *attention / viewpoint selection* will widen. Adding an active
perception task to RPX is cheap insurance against that direction. If
the field stays static-frame-centric, the task is still a useful
contribution on its own merits — it just won't be the headline.

## Risks & open questions

1. **Coverage-NBV definition** — surface-area in world frame vs
   pixel-area in image plane vs ray-bundle non-overlap. Need to pick
   one and defend it (probably ray-bundle: matches what robots care
   about).
2. **K-budget reuse with NVS** — the NVS task already iterates over
   the same context-view sets. Active perception can share generation
   to keep storage / compute bounded.
3. **Trajectory bias** — T265 trajectories were captured by humans;
   the "oracle" inherits human-motion priors. Mitigated by reporting
   the *random_pose floor*: if a learned NBV beats the human-motion
   prior, the gain is real.
4. **PUS_active double-dipping** — using an NVS model from the same
   benchmark to score active perception creates a coupling. Mitigated
   by reporting PUS against **multiple** NVS adapters (best, median,
   worst on standalone NVS) and a non-NVS depth model.
5. **Adapter contract for the slate** — most active-vision papers
   ship code that needs context views as *images plus poses plus
   intrinsics in a specific frame convention*. The adapter API needs
   to expose all three uniformly.

## Implementation footprint (PR-D, estimate)

| File | LoC | Mirrors |
|---|---|---|
| `rpx_benchmark/active_perception_pairs.py` | ~400 | `nvs_pairs.py` |
| `rpx_benchmark/active_perception_metrics.py` | ~300 | `nvs_metrics.py` |
| `rpx_benchmark/tasks/active_perception.py` | ~80 | `tasks/novel_view_synthesis.py` |
| `scripts/run_active_perception.py` | ~350 | `run_nvs.py` |
| `scripts/active_perception_models/__init__.py` | ~120 | `nvs_models/__init__.py` |
| `scripts/active_perception_models/{random_pose,farthest_point,entropy_max}.py` | ~200 | one file each, baseline-scale |
| `tests/test_active_perception_*.py` | ~250 | parity with pose/nvs test surface |

Rough estimate: ~1700 LoC total, the bulk mirroring the existing NVS
pipeline structure. The actual cost is the one-time scene-prep step
that computes `T*` per (scene, phase, objective) — overnight job on
the test mirror, cached after that.

## Decision needed

Two questions for the parallel session before any code:

1. **Headline objective.** Coverage-NBV is my recommendation
   (reproducible, model-agnostic). Confirm or pick Uncertainty / Task.
2. **Paper positioning.** Likely the right framing is a small,
   well-supported claim: *"RPX adds an active-perception evaluation
   built on its existing 6-DoF + multi-phase data; here are
   baselines, headline results, and the scope on which we make no
   claim."* A bolder framing ("perception benchmarks measure the
   wrong thing") is only credible with strong empirical evidence
   that active perception actually predicts deployment outcomes —
   we don't have that yet. Lock the framing once the baselines run.
