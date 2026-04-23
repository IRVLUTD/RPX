# RPX → NeurIPS 2026 E&D Best Paper: Gap-to-Gold Plan

**Created**: 2026-04-16  
**Deadlines**: Abstract May 4 (18 days) · Full paper May 6 (20 days)  
**Track**: NeurIPS 2026 Evaluations & Datasets (formerly D&B)

---

## 1. What wins best paper at this track

The 2025 D&B best paper was **Artificial Hivemind / Infinity-Chat** (Jiang et al.).
The committee praised it for:

> "sets a new standard for datasets and benchmarks that **advance scientific understanding**
> and address **pressing societal challenges** rather than solely improving technical performance"

What made it win:
- **A surprising, named finding** ("Artificial Hivemind effect") — not just numbers in a table
- **Scale**: 26K queries, 31K human annotations, **70+ models** evaluated
- **A new taxonomy** that reframed the problem space
- **Societal implications** clearly articulated
- **Deep analysis** that went far beyond "model X > model Y"

The 2026 E&D track has been **explicitly reframed**: *"evaluation itself becomes an object of
scientific study."* Submissions must articulate *"what claims it supports, under what
assumptions, and what limitations apply."*

**Implication for RPX**: The paper must be an *evaluation science* contribution, not just
"here's a dataset with numbers." The three-phase protocol, ESD, and deployment-readiness
metrics must be framed as **methodological contributions to evaluation**, with the dataset
as the substrate that demonstrates them.

---

## 2. Current state of the manuscript

| Component | Status | Best-paper gap |
|---|---|---|
| Narrative & prose | ~90% written | Needs sharpening, too many threads |
| Formalism (ESD, TS, STR, SGC) | ✅ Complete | Good — this IS the eval-science contribution |
| Related work + comparison table | ✅ Solid | Needs 2026 E&D framing update |
| **Experimental results** | 🔴 **0% — all \todo{}** | **#1 blocker, 30+ empty cells** |
| Dataset statistics | 🔴 9 \todo{} | Need to compute from data |
| ESD weight table + validation | 🔴 7 \todo{} | Need to compute |
| Figures | 🔴 Only 1 exists | Need 3–4 analytical figures |
| Key findings | 🔴 PLACEHOLDER | Need the surprising finding |
| Appendix | 🔴 All stubs | Need extended tables + analysis |
| Downstream validation | 🔴 Not even planned | **#2 blocker for best paper** |
| Croissant metadata | 🔴 Not created | Required by track |
| HF dataset hosting | ⚠️ Unclear status | Required by track |

---

## 3. The seven upgrades from "good paper" to "best paper"

### Upgrade 1: Find the Surprising Finding (THE critical piece)

Every best paper has a named, counterintuitive result. RPX is designed to surface one.
Three candidate findings the experiments should probe:

1. **"The Robustness Inversion"**: The highest-accuracy model on standard benchmarks has
   the *worst* STR (phase-transition robustness). Would mean: leaderboard position is
   misleading for deployment.

2. **"The ESD Reversal"**: Model rankings on Easy scenes are *inverted* on Hard scenes.
   Would mean: easy-scene evaluation actively misinforms backbone selection.

3. **"The Coherence Collapse"**: SGC (mask-depth alignment) degrades catastrophically
   during interaction for all models — even those with high per-frame accuracy. Would
   mean: high accuracy ≠ geometric reliability for grasping.

**Action**: Run experiments specifically designed to test these three hypotheses.
Report whatever the data actually shows, but *design the analysis to detect it*.
Name the finding. A named finding is 10x more memorable than a table.

### Upgrade 2: Scale the model evaluation slate

Infinity-Chat evaluated 70+ models. Current RPX: ~20 across 8 tasks.

**Target**: 40–50 model evaluations across 8 tasks. Not every task needs 10 models,
but the aggregate count matters for credibility.

Priority additions:
- **Depth** (currently best-specified): Add DPT, MiDaS, Marigold, Metric3D v2, UniDepth v2,
  Prompt Depth Anything, Video Depth Anything → 9–10 total
- **Detection**: Add DINO, Co-DETR, YOLOv10 → 7–8 total
- **Segmentation**: Add SAM2 variants (tiny/small/large), Mask2Former, OneFormer → 5–6 total
- **Grounding**: Add CogVLM, Qwen-VL → 4–5 total
- **Keypoints**: Add SuperPoint+SuperGlue, DeDoDe, LightGlue → 5–6 total
- **Pose**: Add DUSt3R, MASt3R → 3–4 total
- **Tracking**: At least 2 models (SAM2 video, DEVA/Cutie)
- **NVS**: At least 2 models (3DGS, InstantSplat/Nerfacto)

### Upgrade 3: Downstream validation experiment ("The So-What Experiment")

This is what separates a good benchmark paper from a best paper.

**Design**: Pick depth (most models, most practical relevance). Take top-3 and bottom-3
depth models by RPX deployment-readiness score. Run each through a simple grasping
pipeline (e.g., GraspNet or contact-graspnet on RPX depth predictions → success rate
in simulation or on real robot). Show that RPX deployment-readiness score correlates
with grasp success rate.

Even a small-scale experiment (5–10 grasp trials per model, sim or real) would be
transformative for the paper's impact.

**Fallback if no time for grasping**: Show that RPX deployment-readiness score correlates
with downstream *policy feature quality* — e.g., run DINOv2 vs CLIP features through
a frozen linear probe on RPX scenes for affordance classification. Cheaper, still
validates the ranking.

**Minimal fallback**: At minimum, show correlation between RPX rankings and rankings from
an *independent* existing benchmark (e.g., compare RPX depth ranking vs NYU/KITTI
ranking and show divergence — this itself supports the claim that standard benchmarks
are misleading).

### Upgrade 4: Frame as evaluation science, not just a dataset

The 2026 E&D track explicitly wants *"evaluation as a scientific object."* Reframe:

**Current framing**: "We present RPX, a dataset and benchmark for robot perception."

**Best-paper framing**: "We present a methodology for evaluating foundation-model
deployment readiness under embodied conditions — instantiated through the RPX dataset
and toolkit — and discover that [SURPRISING FINDING]."

Concrete changes:
- Abstract: Lead with the evaluation methodology contribution and the finding,
  not the dataset statistics
- Intro: The core claim is about *evaluation methodology*, not about the data
- Frame ESD as a contribution to *difficulty measurement science*
- Frame TS/STR/SGC as a contribution to *deployment evaluation methodology*
- Frame the three-phase protocol as an *experimental design for controlled causal
  measurement* of scene-state effects on perception

### Upgrade 5: Analytical depth — go beyond tables

Best papers have figures that become iconic. Current paper: 1 overview figure.

**Required analytical figures** (minimum 4):

1. **The STR scatter plot** (Fig 2): x-axis = standard benchmark accuracy, y-axis = STR.
   If the correlation is weak, this IS the paper's main finding visualised.

2. **The ESD separation plot** (Fig 3): Per-task violin plots showing model performance
   distributions on Easy vs Medium vs Hard. If Hard shows wider spread + different
   ranking → ESD is validated.

3. **The phase-transition heatmap** (Fig 4): Models × Phases, colour = performance.
   Shows degradation during interaction, recovery during clean. The "heartbeat" of
   the benchmark.

4. **The deployment-readiness radar** (Fig 5): Radar/spider plot for top-5 models
   showing accuracy, TS, STR, SGC, efficiency. Different shapes = different deployment
   profiles.

5. **Optional — The ranking divergence plot**: RPX ranking vs. standard-benchmark ranking
   per model. Crossing lines = rankings diverge.

### Upgrade 6: Justify the human-as-proxy claim with data

The critique identified this as under-justified. Fix:

1. Compute Δt, ΔR, jerk statistics from T265 data (the code likely already exists)
2. Gather published motion profiles for Stretch, Spot, Unitree H1, Fetch
3. Show overlap plot: "RPX camera motion falls within the operating envelope of
   these platforms"
4. This turns a hand-wave into a validated design choice

### Upgrade 7: Track-compliance checklist

NeurIPS 2026 E&D has new requirements:

- [ ] **Double-blind submission** (or single-blind with justification for dataset)
- [ ] **Croissant metadata file** with core + RAI fields
- [ ] **Dataset on HF/Kaggle/Dataverse** with < 4GB sample
- [ ] **Code accessible and executable** on anonymous repo
- [ ] **Evaluative claims clearly articulated**: what claims, what assumptions, what limits
- [ ] Paper uses correct style: `\usepackage[eandd]{neurips_2026}` (currently `[dandb]`
  — **check if `eandd` option exists yet or if `dandb` still works**)

---

## 4. Execution timeline (20 days)

### Phase A: Compute & Experiments (Days 1–10, Apr 16–26)

**This is the rate-limiting step. Everything else can be written in parallel.**

| Day | Task | Owner |
|-----|------|-------|
| 1–2 | Compute dataset statistics (Table 2): objects/frame, occlusion, depth invalid | — |
| 1–2 | Compute camera motion statistics (Δt, ΔR, jerk) from T265 data | — |
| 2–4 | Run depth task: all 7–9 models on test split, all phases, all ESD levels | — |
| 2–4 | Run detection task: 5+ models | — |
| 3–5 | Run segmentation task: 4+ models | — |
| 4–6 | Run keypoints, pose, grounding, NVS, tracking | — |
| 5–7 | Compute deployment metrics (TS, STR, SGC) for all results | — |
| 6–8 | Compute ESD weights via mutual information procedure | — |
| 7–9 | Run ESD validation (Spearman ρ) | — |
| 8–10 | **Downstream validation experiment** (grasping or probe) | — |

### Phase B: Analysis & Figures (Days 8–14, Apr 24–30)

| Day | Task |
|-----|------|
| 8–9 | Identify the surprising finding from results |
| 9–10 | Create Fig 2 (STR scatter), Fig 3 (ESD violins) |
| 10–11 | Create Fig 4 (phase heatmap), Fig 5 (radar) |
| 11–12 | Create motion statistics comparison plot |
| 12–13 | Extended appendix tables |
| 13–14 | ESD sensitivity analysis |

### Phase C: Writing & Polish (Days 12–18, Apr 28–May 4)

| Day | Task |
|-----|------|
| 12–13 | Rewrite abstract + intro with new framing and finding |
| 13–14 | Fill all \todo{} in experiments section |
| 14–15 | Fill all \todo{} in dataset + features sections |
| 15–16 | Write downstream validation section |
| 16–17 | Rewrite conclusion with concrete findings |
| 17 | Complete appendix |
| 17–18 | Full paper review pass for coherence |

### Phase D: Submission Prep (Days 18–20, May 4–6)

| Day | Task |
|-----|------|
| 18 | **Abstract submission** (May 4 AoE) |
| 18–19 | Generate Croissant metadata |
| 19 | Verify HF dataset hosting + <4GB sample |
| 19 | Anonymise code repo |
| 19 | Final LaTeX compile + style option check |
| 20 | **Full paper submission** (May 6 AoE) |

---

## 5. What "best paper" RPX looks like vs. current RPX

| Dimension | Current paper | Best-paper version |
|---|---|---|
| **Lead claim** | "Here's a dataset + benchmark" | "Evaluation methodology reveals that standard accuracy is anti-correlated with deployment robustness" |
| **Named finding** | None | "The Robustness Inversion" (or whatever the data shows) |
| **Model scale** | ~20 across 8 tasks | 40–50 across 8 tasks |
| **Figures** | 1 overview | 5–6 analytical figures |
| **Downstream link** | None | Grasp success correlates with RPX score |
| **Framing** | Dataset paper | Evaluation science paper |
| **Human-proxy claim** | Hand-wave | Validated with motion statistics |
| **GT circularity** | Unaddressed | ESD validated on non-SAM2 models |
| **Evaluation claims** | Implicit | Explicitly stated per E&D track requirements |

---

## 6. Risk assessment

| Risk | Probability | Mitigation |
|---|---|---|
| Experiments don't finish in time | HIGH | Prioritise depth (best model slate), report partial results for other tasks with "full results in appendix/supplement" |
| No surprising finding emerges | MEDIUM | Even "deployment metrics are uncorrelated with accuracy" is a finding. Spin the analysis to characterise WHAT differs, not just that models differ |
| Downstream validation fails (no correlation) | MEDIUM | Report honestly — a negative result that RPX ranking doesn't predict grasping is itself informative. Or: redefine the downstream task |
| Track style option `[eandd]` doesn't exist yet | LOW | Verify; `[dandb]` likely still works. Check NeurIPS template zip |
| T265 pose drift too large | LOW | Bound it empirically; 250 frames at 30fps = 8s, drift should be small. Report the bound |
| Reviewer objects to human-as-proxy | HIGH | Motion statistics plot is the defense. Also cite humanoid learning-from-human literature (iTeach, HRT1) as paradigm justification |

---

## 7. The one-paragraph pitch (what the abstract should convey)

> We present a methodology for evaluating the deployment readiness of foundation-model
> perception backbones under embodied robot conditions, instantiated through RPX: a
> 75,000-frame real-world RGB-D benchmark spanning 100 scenes, ~70 object categories,
> and 8 perception tasks evaluated on identical scenes and objects. RPX's three-phase
> capture protocol (clutter → human interaction → clean) enables the first controlled
> measurement of perception degradation and recovery across scene-state transitions on
> real data. We introduce Effort-Stratified Difficulty (ESD) — a difficulty measure
> grounded in annotation effort rather than geometric heuristics — and three deployment-
> readiness metrics (Temporal Stability, State-Transition Robustness, Geometric
> Coherence) that capture properties invisible to standard accuracy metrics. Evaluating
> [N] foundation models across all tasks, we discover [THE SURPRISING FINDING]: ...
> This finding has direct implications for backbone selection in robot learning pipelines
> and cannot be observed on any existing benchmark. RPX is released as a pip-installable,
> bring-your-own-model toolkit with fixed I/O protocols and CI-verified reproducibility.

---

## 8. Immediate next actions (today)

1. **Verify dataset access**: Can we load scenes and run a model through the toolkit right now?
2. **Identify the fastest model to run**: Depth Anything V2 or ZoeDepth — get one full
   result as proof-of-concept
3. **Compute dataset statistics**: This is purely computational, no model needed
4. **Check `neurips_2026.sty`**: Verify if `[eandd]` option exists; if not, check
   if `[dandb]` is still valid
5. **Start a computation log**: Track what's been run, what's pending
