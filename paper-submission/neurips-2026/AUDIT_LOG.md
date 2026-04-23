# RPX Paper — Adversarial Audit Log

**Date**: 2026-04-17
**Scope**: Full paper pass — every section audited for reviewer/AI-review-agent vulnerabilities

---

## Issues Found & Fixed

### 🔴 Critical (would cost accept)

| # | Issue | Fix | File |
|---|-------|-----|------|
| 1 | Abstract: "the first controlled, real-world measurement" unqualified | Added "to our knowledge" | `00_abstract.tex` |
| 2 | Abstract: claims SGC applies to all tasks; SGC only applies to D1 | Changed to "Task-specific deployment-readiness metrics" | `00_abstract.tex` |
| 3 | EDS uses TS, but TS only defined for D1 → EDS undefined for D2-D5 | Added Stab factor: TS for D1, degenerates to 1 for D2-D5 | `03_method.tex` |
| 4 | EDS: $(1 - |\Delta_\text{int}|)$ can go negative | Added clamping: $\min(|\Delta_\text{int}|, 1)$ | `03_method.tex` |
| 5 | η "derived from latency" — no formula | Added explicit formula: $\eta = \min(t_\text{budget}/\text{lat}, 1)$ with configurable budget | `03_method.tex` |
| 6 | SGC threshold τ never specified | Added \todo for value + calibration note referencing D435 depth resolution | `03_method.tex` |
| 7 | SGC unclear whether GT or predicted masks | Explicitly stated "ground-truth mask boundaries" | `03_method.tex` |
| 8 | "causally attributable to scene state alone" too strong | Softened to "primarily attributable" + acknowledged camera trajectory differences | `03_method.tex` |
| 9 | ESD calibration underspecified (which models? how many? chicken-and-egg?) | Added explicit calibration procedure paragraph: val split, N_cal models, test split held out | `03_method.tex` |
| 10 | D3: SAM2 is both GT and evaluated model — circularity | Added explicit circularity acknowledgment + human-correction independence argument | `05_tasks.tex` |
| 11 | GPT-4o rows in main table had broken LaTeX | Fixed `N/A (API) \\ N/A` → `N/A & N/A & N/A` | `06_experiments.tex` |

### 🟡 Significant (would weaken score)

| # | Issue | Fix | File |
|---|-------|-----|------|
| 12 | Opening quote unattributed — looks fabricated | Removed quote format; made it a direct thesis statement | `01_intro.tex` |
| 13 | Intro: "first controlled comparison text vs image" unqualified | Added "to our knowledge" | `01_intro.tex` |
| 14 | Related: "first benchmark whose explicit purpose" unqualified | Added "To our knowledge" | `02_related.tex` |
| 15 | D5: "first benchmark where spatial question has different answers" trivially true | Reworded to emphasize phase-varying QA GT, qualified | `05_tasks.tex` |
| 16 | D4/D5: "Exact-match Acc" too noisy for VLMs | Changed to "Fuzzy-match Acc" with explicit matching procedure (normalized string sim ≥ 0.8) | `05_tasks.tex` |
| 17 | TS only in D1 Diagnostic column; looks underdeveloped for D2-D5 | Added metric applicability table (Table 3) in §3.3 with explicit justification | `03_method.tex` |
| 18 | Label `sec:d3-detection-grounding` says D3 but it's D2 | Renamed to `sec:d2-detection-grounding` | `05_tasks.tex` |
| 19 | RoboVLMs in comparison table is a methods paper, not dataset | Added $^\S$ footnote: "methods paper, included for positioning only" | `02_related.tex` |
| 20-21 | Comparison table mixes dataset properties and methodology features | Added caption text distinguishing the two row categories | `02_related.tex` |
| 22 | "layperson vocabulary" repeated 4× across sections | Kept canonical mention in §4; replaced others with "crowd-sourced natural language" + back-refs | Multiple |
| 23 | "No existing dataset provides all four" — assertion without proof | Softened to "We are not aware of any existing dataset" | `04_dataset.tex` |
| 24 | 100 scenes — indoor/outdoor split unknown | Added \todo{X indoor, X outdoor} | `04_dataset.tex` |
| 25 | "~70 categories" — instances vs categories unclear | Added \todo{X total physical instances} | `04_dataset.tex` |
| 26 | "70/15/15" — scenes or percentages? | Clarified: "70/15/15 scenes (no object or environment overlap)" | `04_dataset.tex` |
| 27 | Downstream: top-3 + bottom-3 of 6 models = all models | Changed to "all N evaluated depth models" with rank-order correlation | `06_experiments.tex` |
| 28 | Broader impact thin — no dual-use | Added surveillance acknowledgment + "adds methodology not capability" defense | `10_conclusion.tex` |
| 29 | Contribution 3 conflated with C2 | Merged into C2: "The RPX dataset and multi-task benchmark" | `01_intro.tex` |
| 30 | "Bharadhwaj" should be "Bharadhwaj et al." | Fixed | `01_intro.tex` |

### Additional hardening

| Issue | Fix | File |
|-------|-----|------|
| Limitations too short | Expanded to 6 explicit limitations with quantification | `10_conclusion.tex` |
| Statistical power concern (n=15 per ESD level) | Added to limitations; all metrics report mean ± std across scenes | `10_conclusion.tex`, `06_experiments.tex` |
| Ranking divergence validation is confounded | Added explicit caveat: "divergence does not prove standard benchmarks wrong" | `06_experiments.tex` |
| EDS decomposability | Added paragraph: EDS is not prescriptive; components reported separately; users can re-weight | `03_method.tex` |
| TS extension path | Explicit note: extending TS to detection/segmentation is a toolkit extension point | `03_method.tex`, `10_conclusion.tex` |
| Confidence intervals inconsistent | Accuracy: mean ± std across scenes; latency: 95% CI | `06_experiments.tex` |

---

## Pass 2: AI-Review-Agent Hardening (2026-04-17)

| # | Issue | Fix |
|---|-------|-----|
| 31 | Checklist `\ref{sec:benchmark}` → label doesn't exist | Changed to `\ref{sec:tasks}` |
| 32 | `Δ_rec` used but never formally defined | Added formal definition alongside `Δ_int` in STR paragraph |
| 33 | `S_ov` vs `S_overall` inconsistency | Added explicit note in table caption linking `S_ov` to `S_overall` |
| 34 | `warp()` function undefined | Added full definition: inverse depth-based warping + N_valid explained |
| 35 | Overview figure shows 8 tasks, paper says 5 | Caption now acknowledges extra tasks as "supported but not evaluated in this work" |
| 36 | Overclaiming: "no existing", "uniquely", unhedged universals | All softened with "to our knowledge" or removed |
| 37 | `others` in bib entries | Already proper `and others` format for natbib |
| 38 | ESD table shows 6/15 features without explanation | Added "of 15 total" + pointer to full weight vector in appendix |
| 39 | Empty `volume={}` fields in bib | Removed |
| 40 | `[dandb]` style option — track renamed to E&D | Verified `.sty` only has `dandb`; added comment |
| 41 | "causally attributable" in intro and conclusion | Changed to "primarily attributable" / removed causal language |
| 42 | "substantially" in scope without evidence | Removed |
| 43 | "the full manipulation-relevant perception stack" — overclaiming | Changed to "multiple manipulation-relevant perception tasks" |

---

## Remaining \todo{} items (all require experimental data)

These are legitimate data-dependent placeholders, not structural issues:

- Abstract: N models, headline finding
- Intro: 2-3 findings summary
- §3: ESD weights (6 values), Spearman ρ, Kendall τ, N_cal, τ_sgc, stability %
- §4: motion stats, drift bound, indoor/outdoor split, instance count, dataset size
- §6: all results table cells (~30+), figure contents, finding narratives
- §7: concrete findings with numbers, drift bound
- Appendix: extended tables, figures, analysis numbers

---

## Structural integrity

- ✅ All `\ref{}` resolve to defined `\label{}`
- ✅ No unqualified "first" claims
- ✅ No broken LaTeX in tables
- ✅ "Layperson vocabulary" appears once (§4 canonical)
- ✅ All contributions are distinct
- ✅ Every metric has explicit applicability scope
- ✅ EDS formula fully specified with bounds for all terms
- ✅ ESD calibration procedure explicit (val/test split, model independence)
- ✅ SAM2 circularity acknowledged with mitigation
- ✅ Causal language appropriately hedged
- ✅ Broader impact addresses dual-use
- ✅ Limitations comprehensive (6 items with quantification)
