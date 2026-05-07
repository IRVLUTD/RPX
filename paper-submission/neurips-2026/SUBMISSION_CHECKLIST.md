# RPX NeurIPS 2026 E&D Submission Checklist

Track: Evaluations & Datasets (E&D)
Abstract deadline: May 4, 2026 (AoE)
Full paper deadline: May 6, 2026 (AoE)

---

## BLOCKING — Must complete before May 4

- [ ] Fill all `\todo{}` in paper with real numbers (see below)
- [ ] Run actual benchmark experiments (at minimum: segmentation + depth on 2–3 models)
- [ ] Upload 5-scene reviewer sample to Hugging Face (anonymous link)
- [ ] Register on OpenReview and create submission profile
- [ ] Submit abstract by May 4

---

## Paper \todo{} items to fill (in order of importance)

### §3 Dataset
- [ ] Camera motion stats: median Δt, median ΔR, p90 jerk (compute from T265 logs)
- [ ] Robot platform comparison reference for motion stats
- [ ] Object supercategory list (~70 categories)
- [ ] Number of distinct physical environments
- [ ] Table 2: frames/avg-objects/occlusion-rate/depth-invalid per phase (compute from annotations)
- [ ] Total dataset size in GB

### §4 ESD Features
- [ ] Table 3: ESD feature weights (run MI analysis against model failure rate)
- [ ] Spearman ρ: ESD vs model failure rate (requires at least one task evaluated)
- [ ] ESD validation figure caption (Fig. esd-validation)
- [ ] Sensitivity analysis results: median τ, IQR, % stable assignments (Appendix A)

### §5 Benchmark Tasks
- [ ] SGC threshold τ_sgc — verify 0.1 m/px is appropriate for D435 output
- [ ] Visual grounding model list (GroundingDINO confirmed; add GLIPv2, others)
- [ ] Depth model list (Depth Anything v2, Marigold, ZoeDepth — confirm availability)
- [ ] NVS model list (confirm 3DGS eval protocol for handheld video)

### §6 Experiments
- [ ] All Table 4 (main results) cells
- [ ] All Table 5 (phase breakdown) cells
- [ ] ESD analysis findings (Fig. esd-analysis)
- [ ] TS correlation value (r = ?)
- [ ] SGC per-phase values (SGC_C, SGC_I, SGC_L)
- [ ] "Key finding" paragraph — fill in specific model names and numbers

### Appendix
- [ ] Camera motion histogram figure (Appendix C)
- [ ] Full object taxonomy table (Appendix D)
- [ ] ESD sensitivity analysis numbers (Appendix A)

---

## Required submission artifacts

- [ ] PDF paper (≤9 pages content + unlimited refs/checklist/appendix)
- [ ] Croissant JSON file (`croissant/rpx_croissant.json`) — update SHA256 hashes
- [ ] Code repository (GitHub, public, pip-installable `rpx-benchmark`)
- [ ] Dataset sample on Hugging Face (<4GB, no login required)
- [ ] Completed NeurIPS checklist (`checklist.tex`)

---

## NeurIPS checklist answers to prepare (all currently [TODO])

1. Claims — [Yes]: intro + experiments support claims with scope limitations stated
2. Limitations — [Yes]: conclusion §Limitations; dataset card
3. Theory/proofs — [N/A]: no theoretical results
4. Reproducibility — [Yes]: pip-installable toolkit, manifests, splits released
5. Open access — [Yes]: HuggingFace + GitHub
6. Experimental details — [Yes]: sensor specs, annotation protocol, eval setup in §3/§5/§6
7. Statistical significance — [Yes]: report std across scenes in all tables
8. Compute resources — [Yes]: GPU type, eval time per model
9. Code of ethics — [Yes]: indoor daily objects, no faces, consent obtained
10. Broader impacts — [Yes]: conclusion §Broader impact
11. Safeguards — [N/A]: dataset poses no misuse risk
12. Licenses — [Yes]: CC BY 4.0 dataset, MIT code; all models properly cited
13. New assets — [Yes]: dataset card + Croissant file + toolkit docs
14. Crowdsourcing/human subjects — [Yes]: lab personnel capture, consent obtained
15. IRB — [Yes/N/A]: document institutional review status
16. LLM usage — [N/A]: LLMs not part of core method (only used for writing)

---

## Anonymisation checks before submission

- [ ] Author block replaced with "Anonymous Author(s)" ✓ (done)
- [ ] `\webpage` points to anonymous URL ✓ (done)
- [ ] No lab name (IRVLUTD) in paper body
- [ ] Self-citations use third person ("In prior work [X]..." not "In our work...")
- [ ] GitHub link is anonymous (use anonymous.4open.science or suppress)
- [ ] No acknowledgments section in submitted version ✓ (ack environment auto-hides)

---

## Post-submission (camera-ready, if accepted)

- [ ] Restore author block
- [ ] Restore real webpage URL
- [ ] Upload full dataset to Hugging Face
- [ ] Update Croissant SHA256 hashes
- [ ] Update anonymous GitHub → real GitHub URL
