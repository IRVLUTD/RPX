# Concrete Improvements While Experiments Run

**Date:** 2026-04-20
**Scope:** Everything that can be improved without experimental results

---

## 1. Related Work: Gaps to Fill

The related work is already strong (6 bodies of work, 14-dimension table). Three missing threads that reviewers on the E&D track will notice:

### 1a. Temporal consistency benchmarks (missing)
The paper mentions Video Depth Anything and ChronoDepth as model contributions but doesn't cite the benchmarks that measure temporal consistency. Key missing references:
- **ScanNet temporal evaluation** — several papers use ScanNet sequences for temporal depth consistency (e.g., Consistent Depth Prediction under Various Illumination Conditions, CVPR 2025)
- **DAVIS / MOSE for tracking consistency** — MOSE (Complex Video Object Segmentation, 2023) is the closest tracking benchmark to what RPX does with its interaction phase
- **Robo-VQA** (Google, 2023) — robot-relevant VQA on real images; should be compared to D4/D5

### 1b. The depth estimation benchmark landscape (sparse)
RPX evaluates 6 depth models but the related work doesn't discuss:
- **KITTI depth benchmark** and its known biases
- **NYU Depth V2** limitations (static, canonical viewpoints)
- **VOID** (Visual Odometry with Inertial-aided Depth, 2020) — real-world depth with camera motion, closest to RPX's setup
- **Spring benchmark** (Mehl et al., 2023) — high-res real-world depth/flow sequences

### 1c. The NeurIPS E&D track's own standards (missing framing)
The paper mentions the track's emphasis on "what claims does an evaluation support" but should cite:
- The **Datasheets for Datasets** framework (Gebru et al., 2021) — RPX should explicitly state its datasheet
- **Croissant** metadata standard (Akhtar et al., 2024) — mentioned in dataset section but not related work
- Prior E&D best papers as methodological precedent

### 1d. Recent real-world robot benchmarks (2025-2026)
Missing:
- **DROID** (Khazatsky et al., 2024) — cited for clutter but not compared as a benchmark
- **RoboCasa** (Nasiriany et al., 2024) — simulation but relevant for multi-task evaluation
- **ManiSkill3** (Tao et al., 2025) — if it exists
- **BEHAVIOR-Vision Suite** (Ge et al., 2024) — multi-task sim benchmark for manipulation perception

---

## 2. Introduction: Structural Fixes (no results needed)

### 2a. Cut "Scope and assumptions" (lines 65-78) → Appendix
This is 13 lines of defensive caveats in the intro. It belongs in the appendix. The intro should end with contributions, not disclaimers. Move the entire "Scope and assumptions" paragraph and the "output → feature" paragraph to Appendix §A.

### 2b. The intro is 2+ pages — cut to ~1 page
- The "Capture design" paragraph can move to §4 (Dataset)
- The "Key findings" paragraph is a TODO — write it last
- The 4-item contribution list can be tightened:
  - Contribution 1 (methodology) and Contribution 4 (toolkit) can merge — the toolkit IS the methodology made reproducible
  - This gives 3 contributions, which is cleaner

### 2c. Kill "to our knowledge" — keep at most 1
Currently 6 instances across the paper. It reads defensive. The comparison table already demonstrates novelty visually.

---

## 3. Methodology: Things to Tighten

### 3a. EDS formula — acknowledge the multiplicative fragility
Add one sentence: "We use a multiplicative formulation so that any single failing dimension prevents a high score — a conservative design choice for safety-critical deployment. The toolkit supports alternative compositions (additive, Pareto); we verify in §6 that main findings are robust to the scoring rule."

### 3b. Define "failure rate" for ESD calibration
The ESD weights use "mutual information between each feature and per-scene model failure rate" but failure rate is undefined. Add: "We define failure as per-scene accuracy below the median accuracy across all scenes for that model — a relative definition that avoids absolute threshold choices."

### 3c. SGC threshold
τ_sgc should be derived from the D435 spec: at 1m range, D435 depth resolution is ~2mm, pixel pitch at 640×480 with ~70° HFOV is ~1.4mm/px at 1m. So τ_sgc ≈ 1.5 mm/px is a reasonable default. State this derivation explicitly.

---

## 4. Tasks: Small Additions

### 4a. D2 model list — add Florence-2 and Qwen2.5-VL
These are the most capable open-weight models for grounding as of April 2026. Adding them strengthens the benchmark's coverage claim.

### 4b. D1 model list — consider adding MASt3R / DUSt3R
Multi-view depth from video is increasingly important. If RPX has sequential frames with pose, multi-view depth models are a natural addition.

### 4c. Clarify "fuzzy matching" for QA
Add 2-3 examples: 
- "a red bottle" matches "red bottle" (article removal) ✓
- "there are 3 items" matches "3 items" ✓  
- "the bottle is on the table" does NOT match "bottle" (similarity < 0.8) ✗

---

## 5. Comparison Table: Fixes

### 5a. Add DROID to the table
DROID (Khazatsky et al., 2024) is the closest comparison — real-world, multi-camera, manipulation-relevant. It should be in the table.

### 5b. Add BEHAVIOR-Vision Suite
Even though it's simulation, it's the most direct comparison for multi-task manipulation perception.

### 5c. The table has 15 columns — consider splitting
14 benchmarks in one row is hard to read. Consider splitting into two tables: (a) dataset properties (rows 1-10) and (b) evaluation methodology (rows 11-15), each with fewer columns. Or highlight RPX's row with a colored background.

---

## 6. Writing Polish

### 6a. The brand name \coolname{} (RPX with colored letters) is playful
Fine for a workshop paper. For NeurIPS best paper, consider whether the colored branding helps or hurts. Some reviewers may find it distracting. A simple \textsc{RPX} might be more professional. Low priority but worth considering.

### 6b. The abstract should be written LAST
Don't finalize the abstract until the headline finding exists. The current abstract is a template with a TODO for the key result — that's fine for now, but the final abstract should lead with the finding, not the methodology.

### 6c. Remove all \finding{} and \todo{} macros before submission
88 TODOs. Track these in a separate checklist, not in the paper.

---

## 7. What to Work on RIGHT NOW (Parallel to Experiments)

| Task | Can do now? | Impact |
|------|------------|--------|
| Fill dataset statistics (scene counts, frame counts, motion stats) | Yes — compute from data | Medium |
| Compute SGC threshold from D435 spec | Yes | Low |
| Add missing related work references | Yes | Medium |
| Move "Scope and assumptions" to appendix | Yes | Medium |
| Cut intro to 1 page | Yes | Medium |
| Add DROID + BVS to comparison table | Yes | Medium |
| Create Croissant metadata | Yes — required by track | High (track requirement) |
| Upload 5-scene sample to HF | Yes — required by track | High (track requirement) |
| Write the Appendix stubs (annotation pipeline, motion validation) | Partially — need some stats | Medium |
| Finalize ESD feature definitions in appendix | Yes | Medium |

---

## 8. The One Thing That Matters Most

The downstream validation (§6.6) — "Does EDS predict grasp success?" — is the experiment that makes or breaks the paper. If it works (ρ > 0.6), the paper has a strong practical contribution. If it doesn't, the methodology is interesting but the scoring framework is unjustified.

Run this experiment with whatever depth models you have results for first. Even 3 models × 10 grasp trials each. The correlation is the headline number.
