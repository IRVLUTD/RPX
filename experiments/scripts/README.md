# RPX Splits Pipeline

> End-to-end pipeline that turns a captured RPX dataset into the
> **Easy / Medium / Hard scene splits** the benchmark ships.

> [!IMPORTANT]
> **Most users do not need this directory.** Splits are how the dataset
> was *labelled* into easy / medium / hard tiers. The benchmark ships
> the resulting `scene_splits.json` already in
> `benchmark/data/splits/`, and the runners read it automatically. You
> only come here if you're re-running the difficulty labelling on a
> *new* capture, or studying the labelling methodology itself.

**What it does.** Each `(scene, phase)` is scored on 27 features
(visual clutter, depth variation, mask refinement iterations, etc.).
The scores are combined into a single difficulty number, sorted, and
cut into Easy / Medium / Hard tiers — so benchmark results can be
reported stratified by difficulty rather than averaged into a single
opaque number.

```
DATA  →  Stage 1 (extract)  →  phase_esd_splits.csv
                             →  Stage 2 (split)
                             →  scene_splits.json   ← deliverable
```

---

## 1. Quick start (one-time setup)

From the repo root:

```bash
# Core install (numpy + Pillow + the rpx_benchmark library).
pip install -e benchmark

# 2-5x faster PNG decode via OpenCV (recommended for Stage 1).
pip install -e 'benchmark[esd-fast]'

# Analysis dependencies (pandas / scikit-learn / scipy / matplotlib).
pip install -e 'benchmark[analysis]'
```

Python >= 3.10. Tested on Linux (CI: 3.10 / 3.11 / 3.12). macOS expected to
work; Windows untested.

## 2. Run the pipeline

Two commands, top to bottom:

```bash
# ── Stage 1: extract 27 features per (scene, phase) ──────────────────
python experiments/scripts/build_esd_splits.py \
    --data-root /path/to/test_dataset_aggregated \
    --output    benchmark/data/splits/phase_esd_splits.json \
    --workers   $(nproc)

# ── Stage 2: turn features into Easy/Medium/Hard scene splits ────────
python experiments/scripts/build_difficulty_splits.py \
    --features benchmark/data/splits/phase_esd_splits.csv
```

Stage 1 takes ~15 min on 100 scenes × 3 phases × ~250 frames with 20
workers (CPU-bound on PNG decode). Stage 2 takes ~1 second.

## 3. What ships in `benchmark/data/splits/`

After Stage 2, the directory contains:

| File | Purpose | Audience |
|---|---|---|
| **`scene_splits.json`** ⭐ | per-scene Easy/Medium/Hard tier (33/33/34 if 100 scenes; 33/33/33 if 99) | **benchmark consumers — run models against this** |
| `phase_difficulty.json` | per-(scene, phase) structured metric: tier, consensus_tier, confidence, per-category sub-scores | analysts (per-phase reporting, STR analysis) |
| `phase_difficulty.csv` | flat, pandas-ready table with all 5 scoring methods side-by-side | analysts |
| `easy.txt` / `medium.txt` / `hard.txt` | per-(scene, phase) tier lists, one key per line | task runners filtering at the per-phase level |
| `splits_provenance.json` | input SHA-256, weights vector, git commit, generation timestamp | reproducibility audit |
| `phase_esd_splits.{json,csv}` | Stage 1 raw 27-feature table | input to Stage 2; also useful on its own for the methodology study |

### Loading scene_splits.json

```python
import json
splits = json.load(open("benchmark/data/splits/scene_splits.json"))["splits"]
# splits["easy"]   → list of scene_ids (33 entries)
# splits["medium"] → list of scene_ids (33 entries)
# splits["hard"]   → list of scene_ids (33 or 34 entries)
```

**STR guarantee**: all 3 phases (clutter / interaction / clean) of a scene
land in the same split. Required for state-transition robustness analysis.

## 4. Methodology in one paragraph

The score is **effort-stratified**:

```
RPX-DS(s, p) = α · effort_score(s, p)  +  (1 − α) · perception_score(s, p)
```

with **α = 0.25** by default. `effort_score` is the mean of the two
percentile-normalised mask-refinement features (`iter_mean`, `iter_max`);
`perception_score` is the mean of the other 25 percentile-normalised
features (8 modality categories — see paper §3.2 + appendix B).

Per-(scene, phase) tertile cut → 99 entries per tier (under N=99 scenes).
Per-scene rollup is the **mean of the 3 phase scores per scene**, then
sorted and cut into ⌊N/3⌋ Easy + ⌊N/3⌋ Medium + remainder Hard.

## 5. Tunable knobs

| Flag | Default | What it controls |
|---|---|---|
| `--effort-alpha 0.25` | `0.25` | convex weight on annotation effort. 0 = uniform; 0.25 = modest boost (default); 0.5 = effort dominates. |
| `--weighting effort_stratified` | `effort_stratified` | switch to `--weighting uniform` for the w_i = 1/27 baseline (legacy comparison) |
| `--weights-mi <PATH>` | unused | load mutual-information-derived weights from a calibration model set; activates `mi_v1` scoring (paper §3.2 calibration paragraph) |
| `--primary <method>` | `mean_pn` | primary scoring method. Alternatives: `median_pn`, `max_pn`, `pca_pc1`, `kmeans3` (used to populate `tier_*` columns; the *primary* one drives the txt + scene_splits files) |
| `--confidence-n-perturb 1000` | `1000` | Dirichlet weight samples used for the per-row confidence flag |
| `--seed 0` | `0` | RNG seed for k-means / GMM / weight perturbations |

## 6. Optional: methodology study + visualisations

After Stage 2, you can produce the artifacts referenced from paper appendix B:

```bash
# 11-method comparison + stability framework (powers paper appendix)
python experiments/scripts/difficulty_methodology_study.py \
    --features benchmark/data/splits/phase_esd_splits.csv

# Feature ablation (which features are load-bearing)
python experiments/scripts/feature_ablation_study.py \
    --features benchmark/data/splits/phase_esd_splits.csv

# 5 publication-quality story figures
python experiments/scripts/data_story_visuals.py \
    --features benchmark/data/splits/phase_esd_splits.csv

# Auto-generated data-story narrative
python experiments/scripts/data_story_analysis.py \
    --features benchmark/data/splits/phase_esd_splits.csv
```

Outputs land under `experiments/splits/` (`methodology_study/`,
`data_story/`, `story_visuals/`).

## 7. Verifying reproducibility

Every output JSON carries a `provenance` block with the input file's
SHA-256, the exact weights vector used, the git commit of the script, and
the generation timestamp. Two runs on the same data + same flags produce
byte-identical outputs:

```bash
sha256sum benchmark/data/splits/scene_splits.json
# Re-run from scratch; rerun the sha256sum; should match.
```

Both stages are seed-locked (default `--seed 0`).

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Stage 1 fails with `data_root not a directory` | Path typo or unmounted disk | Verify `--data-root` points at the directory containing `<scene>.<area>/<phase>/...` |
| Stage 1 partial-failure (`exit code 1`) | One or more scenes have corrupt files | Check `phase_esd_splits.log` for per-phase tracebacks; the JSON's `summary.failures` lists offending paths |
| Stage 2 fails with `feature columns missing from CSV` | Stage 1 CSV produced by older code | Re-run Stage 1 against the latest `rpx_benchmark` |
| Different splits across runs | Different `--seed` / `--effort-alpha` / `--weighting` | Confirm both runs use the same flags; check `splits_provenance.json` `weights` block |
| Some scenes missing from `scene_splits.json` | Stage 1 dropped them due to all-zero features | Check Stage 1 log for "dropping N rows with all-zero features" — usually means the source data had bad masks for that scene |

## 9. Tests

```bash
cd benchmark && python -m pytest tests/ -q
```

318 tests (library + CLI smoke + scoring + scene-split aggregation +
feature ablation). CI runs the suite on Python 3.10 / 3.11 / 3.12 against
every push touching `benchmark/**`.

---

## Repo layout (relevant pieces only)

```
RPX/
├── benchmark/
│   ├── rpx_benchmark/data/
│   │   ├── esd.py              # 27-feature extractor (Stage 1 library)
│   │   └── esd_scoring.py      # scoring / tertile / consensus / confidence helpers
│   ├── tests/                  # 318 tests
│   └── data/splits/            # ← OUTPUT lands here
│       ├── scene_splits.json
│       ├── phase_difficulty.json
│       └── ...
└── experiments/
    ├── scripts/
    │   ├── build_esd_splits.py             # Stage 1 CLI
    │   ├── build_difficulty_splits.py      # Stage 2 CLI
    │   ├── difficulty_methodology_study.py # 11-method comparison (optional)
    │   ├── feature_ablation_study.py       # which features are load-bearing
    │   ├── data_story_analysis.py          # auto-generated narrative
    │   ├── data_story_visuals.py           # 5 publication figures
    │   └── README.md                        # ← you are here
    └── splits/                              # study outputs (not the splits themselves)
```
