# Critique: "Same Scene, Different Story: Ranking Robot Perception Backbones for Real-World Deployment"

**Target:** NeurIPS 2026 Evaluations & Datasets Track  
**Reviewed by:** Feynman, 2026-04-20  
**Status:** Pre-experiment draft — methodology and framing complete, all results are `\todo{}`

---

## Overall Assessment

This is a well-motivated benchmark paper with a strong conceptual backbone. The three-phase protocol, ESD, and deployment-readiness metrics (STR, TS, SGC → EDS) form a coherent methodology that addresses a genuine gap: no existing benchmark ranks robot perception backbones across multiple tasks on the same real-world scenes under controlled scene-state variation.

The writing is sharp and the positioning against existing work is thorough (the Related Work section is one of the strongest parts). The scope is ambitious — 5 tasks, ~22 models, 100 scenes, a pip-installable toolkit.

**The core risk:** The paper has zero experimental results. Every table, figure, and finding is `\todo{}`. The methodology is well-designed on paper, but until the experiments run, the paper's central claim — that standard accuracy doesn't predict deployment robustness — is an untested hypothesis. The entire §6 (Experiments) is placeholder boxes.

**Verdict if submitted as-is:** Desk reject. The methodology is publishable; the paper is not.

---

## Strengths

1. **The three-phase protocol is a genuine contribution.** Clutter → Interaction → Clean mirrors the real manipulation cycle. Holding scene identity fixed while varying scene state is a controlled-experiment design that no existing benchmark provides. This is the paper's best idea.

2. **ESD is well-designed.** Grounding difficulty in annotation effort rather than geometric heuristics is a smart methodological choice. The calibration/validation split to avoid circularity is carefully thought through.

3. **Multi-task on shared scenes.** Evaluating depth, detection, tracking, and VLM QA on the *same* physical scenes is uniquely valuable — it enables cross-task deployment recommendations that siloed benchmarks cannot.

4. **The "five decisions" framing is reader-friendly.** Mapping each task to a concrete robot-system decision (D1: which depth model for DP3? D2: which detector for OK-Robot?) makes the benchmark immediately actionable.

5. **Positioning is strong.** The Related Work section systematically places RPX against 6 bodies of work with a 14-dimension comparison table. The "unjustified backbone choice" argument is convincing and well-cited.

6. **Limitations are honest.** The paper acknowledges sensor envelope, motion proxy, pose drift, ESD circularity, output-vs-feature evaluation, and scale. This is above average for benchmark papers.

---

## Critical Issues

### 1. No results exist (88 `\todo{}` markers)

Every experiment is a placeholder. This means:
- The headline finding ("standard accuracy is anti-correlated with deployment robustness") is a hypothesis, not a result
- All figures are red `TODO` boxes
- The abstract's key sentence is `\todo{HEADLINE FINDING}`
- The conclusion restates the methodology without any empirical contribution

**Priority:** This is the blocking issue. Nothing else matters until experiments run.

### 2. EDS multiplicative formula may be fragile

The EDS formula `EDS = S_overall · Rob · Stab · η` is multiplicative, meaning one bad factor zeros everything out. This has failure modes:
- A model with excellent accuracy, stability, and efficiency but one bad interaction drop (Rob → 0.1) gets an EDS near zero, even if it's the best available option
- The efficiency factor η penalizes latency linearly but uses a hard budget (33ms for 30fps) — many good models will be penalized for not meeting real-time requirements that may not apply to their deployment context
- No justification is given for why multiplication is better than weighted average, Pareto ranking, or lexicographic ordering

**Suggestion:** Present EDS as one reasonable default. Show that main findings are robust to the scoring formula (e.g., additive weighting, Pareto analysis). The toolkit already supports re-weighting — surface this in the main paper.

### 3. SAM2 circularity is acknowledged but not fully resolved

SAM2 is used for:
- Generating ground-truth masks (annotation)
- Evaluated as a tracker (D3)
- Generating bounding boxes for detection GT (via mask → box)

The paper acknowledges this and says human corrections make GT independent. But:
- How many keyframes were human-corrected vs. SAM2-propagated? If 95% are SAM2-propagated, the GT is still SAM2-biased
- The paper evaluates SAM2 on GT generated partly by SAM2 — the "note" mentioned in §5 D3 is not sufficient; reviewers will flag this
- ESD features include annotation-effort metrics derived from SAM2 annotation iterations

**Suggestion:** Report the fraction of human-corrected keyframes. Consider a small held-out set with fully manual annotation to validate GT quality. At minimum, report SAM2 D3 results with a clear caveat and separately from other trackers.

### 4. 100 scenes / 15 test scenes may be underpowered

- 15 test scenes × 3 phases = 45 scene-phase pairs
- Split into Easy/Medium/Hard = ~15 pairs per difficulty level
- With ~22 models, pairwise comparisons need sufficient statistical power

The paper mentions "standard deviations across scenes" but 15 scenes may not be enough to detect meaningful rank differences. A power analysis or bootstrap confidence interval on rank correlations would strengthen the claims.

### 5. The "output → feature" argument is a stretch for VLAs

The paper argues that output-level evaluation (depth maps, boxes, QA answers) informs feature-level selection for VLAs. The three arguments are:
1. TS and STR measure "representation stability" — but this assumes output instability implies feature instability, which isn't necessarily true (the head could be the unstable component)
2. D4/D5 evaluate VLMs whose encoders VLAs inherit — true but VLAs fine-tune these encoders
3. Modular systems consume raw outputs — this is the strongest argument but only applies to modular systems

**Suggestion:** Be more explicit about which claims apply to modular systems (strong) vs. VLAs (weaker proxy). The downstream validation (§6.6) is the right experiment to test this — make sure it actually runs.

### 6. The downstream validation is critical but undefined

§6.6 ("Does EDS Predict Grasp Success?") is the experiment that validates the entire scoring framework. It's currently a TODO box. If EDS doesn't correlate with grasp success, the paper's practical value collapses.

**Suggestion:** This is the most important experiment in the paper. Run it first, even with a small-scale setup (5 models × 10 trials). If the correlation is weak, the paper needs to be reframed.

---

## Minor Issues

### Writing

7. **Abstract is too long** for NeurIPS. The three contributions could be stated more concisely. The `\todo{HEADLINE FINDING}` in the abstract is a red flag — the abstract should be written last, around the actual finding.

8. **Introduction is 2+ pages.** For an 8-page NeurIPS paper (+ appendix), the intro should be ~1 page. The "Scope and assumptions" paragraph (lines 65–78 of 01_intro.tex) reads like an appendix section — move it there.

9. **"to our knowledge" appears 6 times.** Once in the abstract is fine; six times signals defensive writing. Cut to 1–2 instances.

10. **The `\finding{}` and `\todo{}` macros** should be removed before submission (obviously, but 88 TODOs is a lot of work remaining).

### Methodology

11. **ESD weight calibration uses mutual information with model failure rate** — but "failure rate" is not defined. What counts as a failure? Below-median accuracy? Below a threshold? This matters for weight stability.

12. **SGC threshold τ_sgc** is described as "calibrated to D435 depth resolution at 1m" but the actual value is `\todo{X}`. This should be derived from the D435 spec sheet (depth resolution ~1mm at 1m → gradient threshold should be ~1mm/px or similar).

13. **Fuzzy matching for QA** (normalized string similarity ≥ 0.8) may be too lenient or too strict depending on answer format. Example: "a red bottle" vs "red bottle" vs "the bottle is red" — do all match? Provide examples.

### Scope

14. **No navigation-specific perception** is covered. This is fine and stated as a limitation, but some reviewers may want at least depth-based obstacle detection.

15. **No point-cloud / 3D evaluation.** Given that DP3 and many grasp planners consume point clouds, evaluating depth models by their point-cloud quality (not just per-pixel AbsRel) would be more deployment-relevant.

16. **The model list is reasonable but could include newer models.** Depth Anything V2 is there, but what about Depth Pro V2 (if it exists), MASt3R for multi-view depth, or UniK3D? For detection, Qwen2.5-VL and Florence-2 are missing.

---

## Structural Recommendations

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | **Run experiments** — fill all 88 TODOs | Very High | Blocking |
| 2 | Run downstream validation (§6.6) first — if EDS doesn't correlate, reframe | High | Very High |
| 3 | Add power analysis or bootstrap CIs for rank comparisons | Medium | High |
| 4 | Address SAM2 circularity more rigorously | Medium | High |
| 5 | Move "Scope and assumptions" to appendix; cut intro to 1 page | Low | Medium |
| 6 | Show EDS robustness to scoring formula | Medium | Medium |
| 7 | Cut "to our knowledge" to 1–2 instances | Low | Low |

---

## Bottom Line

The methodology is the paper's real contribution — three-phase protocol, ESD, and deployment-readiness metrics are well-designed and address a genuine gap. The positioning is sharp, the related work is thorough, and the toolkit design is solid.

But a benchmark paper with zero results is not a paper — it's a proposal. The 88 `\todo{}` markers need to become real numbers. The headline finding (is standard accuracy actually anti-correlated with deployment robustness?) is the central bet. If the experiments confirm it, this is a strong NeurIPS E&D submission. If they show only weak correlation, the framing needs to change from "standard benchmarks are wrong" to "standard benchmarks are incomplete."

Run the experiments. Everything else is secondary.
