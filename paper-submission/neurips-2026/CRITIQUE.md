# RPX Paper Critique — Simulated NeurIPS E&D Review

**Paper:** Same Scene, Different Story: Ranking Robot Perception Backbones for Real-World Deployment  
**Reviewer perspective:** Senior researcher in robot learning / embodied AI, familiar with VLAs, benchmarks, and real-world deployment.

---

## Overall Assessment

**Recommendation:** The prose, framing, and methodology are strong. If the experiments deliver, this is a top-tier E&D submission. But several structural and scientific issues need fixing before submission. I list them in order of severity.

---

## 🔴 Critical Issues (must fix before submission)

### C1: The paper has zero experimental results

Every results cell is `\todo{X}`. This is acknowledged as waiting on experiments, but the paper cannot be submitted in this state. The prose quality is high enough that weak or missing results would stand out more, not less.

**Fix:** This is the known blocker. The experiment playbook is in place.

### C2: Duplicated sentence in §3

> "RPX's interaction phase captures exactly this data type: a human manipulating daily objects, viewed from ego (GoPro) and exo (D435) perspectives simultaneously."

This sentence appears **twice** in the three-phase justification paragraph, verbatim. A reviewer will notice.

**Fix:** Delete the duplicate.

### C3: Instance segmentation was dropped from the task set but remnants remain

The five tasks are D1 (Depth), D2 (Detection/Grounding), D3 (Tracking), D4 (Scene QA), D5 (Spatial QA). But:
- SGC (mask-depth boundary alignment) is defined as a deployment metric and appears in the main results table for D1 Depth — but SGC requires **predicted segmentation masks**. If there's no segmentation task, where do the masks come from?
- The ESD formulation uses mask refinement iterations — which is fine since masks exist as GT, but the connection to model evaluation is weaker without a segmentation task.
- Table 5 has TS and SGC columns for D1 Depth but "--" for everything else. If only depth has TS and no task has SGC evaluation, these metrics feel underdeveloped.

**Fix:** Either:
(a) Add instance segmentation back as a task (even with 2-3 models) so SGC is evaluated, OR
(b) Remove SGC from the paper entirely and keep only TS and STR, OR  
(c) Clarify that SGC is computed by pairing depth predictions with GT masks (not predicted masks) — but this changes the metric's meaning.

### C4: The QA ground truth is auto-generated, not human-verified

For D4 and D5, you generate Q/A pairs programmatically from questionnaire attributes and bounding-box geometry. A reviewer will ask:
- How many Q/A pairs per scene? Per phase?
- What is the false-positive/false-negative rate of auto-generated spatial QA? (e.g., "what is to the left of the bottle?" depends on viewpoint — from whose perspective?)
- Are the auto-generated Q/A pairs human-verified? If not, evaluation noise from GT errors will dominate model differences.

**Fix:** 
- State the total number of Q/A pairs generated.
- Define spatial relations precisely (e.g., "left of" = image-plane left, not world-frame left).
- Either human-verify a sample of Q/A pairs and report agreement rate, or add a caveat about GT noise.

### C5: ESD weights require model failure rates that don't exist yet

The ESD weighting procedure uses "mutual information between each feature and per-scene model failure rate on a held-out calibration set." But:
- This calibration set requires running models first — which hasn't happened.
- This creates a chicken-and-egg: you need model runs to define ESD, but ESD is used to stratify the evaluation.
- A reviewer will ask: is ESD data-dredging? Are you optimising the difficulty split to make your results look good?

**Fix:**
- Use the calibration set (val split, not test split) for ESD weight calibration.
- Report ESD validation on the test split using models NOT used for calibration.
- State this procedure explicitly to preempt the data-dredging concern.

---

## 🟡 Significant Issues (should fix)

### S1: The "five decisions" framing conflates tasks with different evaluation structures

D1 (Depth) evaluates per-frame dense prediction with continuous GT.
D2 (Detection/Grounding) evaluates per-frame localisation with box/mask GT.
D3 (Tracking) evaluates video-level consistency with tracklet GT.
D4-D5 (QA) evaluate language generation with string matching.

These are fundamentally different evaluation paradigms. Calling them all "decisions" is a rhetorical device, not a methodological unification. The scoring protocol ($S_p$, $\Delta_{int}$, $\Delta_{rec}$) works naturally for D1-D3 but is awkward for QA (what does "weighted phase score" mean for string accuracy?).

**Fix:** Acknowledge this heterogeneity. State how STR/ESD-weighted scoring applies to QA tasks specifically. Consider whether TS and SGC apply to QA (they likely don't — making the deployment metrics less "unified" than claimed).

### S2: The comparison table has 14 rows but some are strained

"Perception-level backbone ranking" row: RPX = ✓, RoboVLMs = ○. But RoboVLMs does compare backbones systematically — it just does it in sim and at the policy level. Marking it "○" is borderline unfair.

"In-context visual prompt eval" row: This is a narrow feature, not a fundamental benchmark dimension. Including it as a row alongside "Real-world capture" and "Instance masks" inflates RPX's uniqueness score.

**Fix:** Keep the table but be honest in the caption: some dimensions are dataset properties (real-world, masks, depth), others are evaluation methodology contributions (deployment metrics, in-context prompting). Don't mix them without acknowledging the difference.

### S3: The paper claims "first" too many times without qualifying

Count of "first" claims:
- "first controlled, real-world measurement of how perception degrades during manipulation"
- "first controlled comparison of text-prompted vs. image-prompted perception"
- "first benchmark where the same spatial question has different correct answers across phases"
- "first real-world benchmark to jointly evaluate the full manipulation-relevant perception stack"

A reviewer will try to find counterexamples. Are ALL of these defensible?

**Fix:** Audit each "first" claim. Replace with "to our knowledge, the first" where there's any doubt. Drop claims that are trivially true (e.g., "first benchmark where spatial QA answers change" — this is true by construction, not by effort).

### S4: 100 scenes × 15 test scenes is thin for statistical power

With 15 test scenes × 3 phases = 45 scene-phase pairs, split into 3 ESD levels = ~15 per level. With 5-7 models per task, can you detect meaningful ranking differences with n=15? 

**Fix:** Report confidence intervals or standard errors on all metrics. If the intervals overlap, acknowledge this as a limitation rather than reporting point estimates that appear precise.

### S5: No downstream validation experiment

The paper claims RPX rankings predict deployment success, but never validates this. RoboVLMs connects backbone choice to policy success (in sim). RPX does not close this loop.

**Fix:** Either:
(a) Add a small downstream experiment (best and worst depth model → grasping success in sim or real), OR
(b) Explicitly state this as a limitation and future work, and weaken the deployment-readiness claim to "deployment-readiness proxies."

---

## 🟢 Minor Issues

### M1: The \texttt{rpx-benchmark} toolkit section in the main body may be too long for a 9-page paper

The toolkit is a contribution but takes ~0.75 page. For page-constrained submission, consider moving most of it to appendix and keeping 2-3 sentences + the CLI example in main.

### M2: The figure caption for Fig 1 still says "Eight perception tasks"

Changed to "Five" in some places but the overview figure caption might still reference the old count. (Already fixed in root.tex but verify the actual image.)

### M3: The overview figure shows 8 tasks

The RPX-overview.png image shows task bubbles including NVS, Keypoint Correspondences, etc. that have been cut from the paper. This creates a visual contradiction.

**Fix:** Update the overview figure to show only the 5 tasks, or add a note that the dataset supports additional tasks shown in the figure.

### M4: "Layperson vocabulary" is stated 4 times

The FewSOL questionnaire / layperson vocabulary point is made in: abstract, intro, §3, and §5 D2. This is redundant.

**Fix:** State once prominently (§3) and reference briefly elsewhere.

### M5: Some bib entries are incomplete

`@article{lu2025gwm, author={Lu, others}}` — "others" is not a valid author. Several entries use this pattern. A reviewer may not notice, but it's sloppy.

**Fix:** Complete all author lists or use "et al." properly in the .bib file (natbib handles truncation).

### M6: VLM evaluation is standard VQA — what makes it different on RPX?

GPT-4o and Qwen2-VL are already evaluated on dozens of VQA benchmarks (VQAv2, GQA, SQA3D, etc.). A reviewer will ask: what makes RPX's QA evaluation different enough to be a contribution?

**Fix:** Emphasise the THREE unique properties: (1) same spatial question, different correct answer across phases; (2) robot-grade sensor imagery (640×480, D435 color, not internet photos); (3) layperson vocabulary on real daily objects. These three together are what no VQA benchmark has.

---

## Summary

| Category | Count | Status |
|---|---|---|
| 🔴 Critical | 5 | Must fix before submission |
| 🟡 Significant | 5 | Should fix for strong submission |
| 🟢 Minor | 6 | Nice to fix |

**The paper's thesis is strong. The methodology is novel. The citations are comprehensive (110 entries). The toolkit is a genuine community contribution. But the submission will live or die on C1 (experiments) and the structural issues (C3 segmentation/SGC, C4 QA ground truth, C5 ESD calibration). Fix those and this is a legitimate best-paper contender.**
