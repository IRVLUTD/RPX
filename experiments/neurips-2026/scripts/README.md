# RPX ESD Splits Pipeline

Two scripts that build and explore the **per-(scene, phase) Effort-Stratified
Difficulty (ESD)** feature table that drives RPX's Easy / Medium / Hard splits.

```
build_esd_splits.py  →  phase_esd_splits.json   ──┐
                        phase_esd_splits.csv      │
                        phase_esd_splits_feature_stats.csv
                        phase_esd_splits.log

analyze_esd.py       ←  phase_esd_splits.csv   ──┘
                     →  splits/analysis/{...}    (tables, plots, findings.txt)
```

The build script is **standalone** — only `numpy`, `Pillow`, and the local
`rpx_benchmark` library are required. The analyze script needs heavy data-
science deps (pandas / scikit-learn / scipy / matplotlib), declared as a
`pip install rpx-benchmark[analysis]` extra.

---

## 1. What this computes

**27 features per (scene, phase)** spanning seven categories:

| Category | Features | Source files |
|---|---|---|
| Annotation effort (2) | `iter_mean`, `iter_max` | `sam2/iter{1..4}_faulty.txt` |
| Scene complexity (3) | `obj_mean`, `obj_std`, `obj_consist` | `sam2/masks/`, `sam2/mask_to_object.json` |
| Occlusion (3) | `occ_mean`, `occ_p90`, `occ_heavy` | `sam2/masks/` (bbox overlap) |
| Depth quality (4) | `depth_invalid`, `depth_invalid_mask`, `depth_std`, `depth_std_mask` | `depth/` + `sam2/masks/` |
| Photometric ↔ depth (2) | `specular`, `dark` | `rgb/` + `depth/` |
| Temporal stability (3) | `area_cv`, `area_drop`, `vis_instability` | `sam2/masks/` (per-instance time series) |
| Camera motion (5) | `trans_mean`, `trans_p90`, `rot_mean`, `rot_p90`, `jerk` | `cam_pose/` |
| Fisheye / stereo (5) | `fisheye_dark`, `fisheye_bright`, `fisheye_sharpness`, `fisheye_corr`, `fisheye_texture` | `fisheye/` (T265 stereo pairs) |

Plus **descriptive stats per feature** (n, mean, std, var, min, p25, p50,
p75, max, CV, 95% CI of mean, prop_zero) — emitted in both the JSON
(`feature_stats` block) and a sibling `_feature_stats.csv`.

Formulas follow the appendix in `paper-submission/neurips-2026/overleaf/text/12_appendix.tex`,
plus 8 features carried over from the older `RPX-overleaf.bak.pdf`
(`area_drop`, `trans_p90`, `rot_p90`, and the 5 fisheye features).

---

## 2. Dataset layout expected

```
<data_root>/
  <scene_dir>/                       # e.g. "scene1.GDC.cse" or just "scene1"
    0/  1/  2/                       # phase index (clutter / interaction / clean)
      rgb/      00000.png …          # uint8 RGB
      depth/    00000.png …          # uint16 millimetres, 0 = invalid
      cam_pose/ 00000.npz            # keys: position (3,), orientation (4,) [x,y,z,w]
      fisheye/                       # any of the 3 layouts auto-detected:
          left/00000.png             #   (a) left/+right/ subdirs
          right/00000.png            #
          00000_L.png  00000_R.png   #   (b) _L/_R suffixes
          00000.png                  #   (c) single (no stereo → fisheye_corr=0)
      sam2/
        masks/    00000.png          # 1-channel, pixel value > 0 = instance ID
        mask_to_object.json          # {"<id>": "<label>", …} — optional
        iter1_faulty.txt             # frame stems still faulty after iter i
        iter2_faulty.txt             # one stem per line; tolerates ".png", paths
        iter3_faulty.txt
        iter4_faulty.txt
```

Frame indices align across modalities **by basename** (`00000.png` ↔ `00000.npz`);
frames missing from any required modality are dropped with one summary
warning per phase.

---

## 3. Install

From the repo root:

```bash
# Build script only (lightweight — colleague who just runs the extractor):
pip install -e benchmark

# +2-5× faster PNG decode (optional but recommended):
pip install -e 'benchmark[esd-fast]'

# +exploratory analysis:
pip install -e 'benchmark[analysis]'
```

Python ≥ 3.10. Tested on Linux (CI: 3.10 / 3.11 / 3.12). macOS expected to
work; Windows untested.

---

## 4. Build the feature table

```bash
python experiments/neurips-2026/scripts/build_esd_splits.py \
    --data-root /path/to/test_dataset_aggregated \
    [--output    experiments/neurips-2026/splits/phase_esd_splits.json] \
    [--workers   N]                  # default: cpu_count
    [--log-file  build.log]          # default: <output>.log
    [--verbose]                      # DEBUG-level logging
```

**Outputs** (in `--output`'s directory):

| File | Purpose |
|---|---|
| `phase_esd_splits.json` | Full record: schema_version, feature_names, summary (data_root, started_at_utc, duration_s, n_entries, n_failed, ok, failures with tracebacks), feature_stats per feature, and the 27-feature payload per (scene, phase) |
| `phase_esd_splits.csv` | Flat 1-row-per-(scene, phase) table with a `# schema_version=...` header line. Pandas/sklearn ready: `pd.read_csv(path, comment='#')` |
| `phase_esd_splits_feature_stats.csv` | One row per feature with descriptive stats (mean, std, percentiles, 95% CI, etc.) |
| `phase_esd_splits.log` | Mirrors all log records (per-phase OK/FAIL lines, warnings, errors with tracebacks) |

**Exit codes** (script-level, useful for CI / automation):

| Code | Meaning |
|---|---|
| 0 | All phases extracted successfully |
| 1 | At least one phase failed; partial JSON/CSV still written; tracebacks under `summary.failures` |
| 2 | `--data-root` is not a directory |
| 3 | Fatal error during gather (logged with traceback) |
| 4 | Output write failed (logged with traceback) |

**Performance**: streaming per-frame extractor holds ~tens of MB per worker.
Workers default to `cpu_count()`; bump down if I/O-bound. With cv2 installed
(`[esd-fast]` extra), PNG decode is 2–5× faster than the PIL fallback.

---

## 5. Explore scoring methods (analyze step)

```bash
python experiments/neurips-2026/scripts/analyze_esd.py \
    --features experiments/neurips-2026/splits/phase_esd_splits.csv \
    [--out-dir experiments/neurips-2026/splits/analysis] \
    [--seed    0]                    # k-means / GMM seed
```

Loads the CSV, percentile-normalises each feature to [0,1] (matching paper
§3.2's `f̃ᵢ`), then runs **7 candidate scoring methods** and writes the
following analysis artefacts to `--out-dir`:

### Scoring methods compared

| Method | Family | What it asserts about difficulty |
|---|---|---|
| `mean_pn` | percentile aggregator | "all 27 features matter equally" — uniform-weighted RPX-DS |
| `median_pn` | | robust: half the features matter; outlier features ignored |
| `max_pn` | | weakest-link: the single hardest feature defines difficulty |
| `top3_mean_pn` | | the 3 hardest features per row matter; rest is noise |
| `pca_pc1` | dim-reduction | first principal component of the percentile matrix |
| `kmeans3` | clustering (k=3) | natural groupings on first 5 PCs; clusters ordered by centroid |
| `gmm3` | clustering (k=3) | Gaussian-mixture on first 5 PCs |

Each method produces a continuous score AND an Easy / Medium / Hard
tertile label per (scene, phase).

### Outputs

| File | Contents |
|---|---|
| `feature_stats_by_phase.csv` | per (feature × phase): n, mean, std, min, p25, median, p75, IQR, max, MAD, 95% CI, prop_zero |
| `phase_difference_tests.csv` | Kruskal–Wallis across all 3 phases + pairwise Mann–Whitney U + Cohen's d |
| `feature_correlations_pearson.csv`<br>`feature_correlations_spearman.csv` | per-feature pairwise correlations (Pearson on percentile-normalised, Spearman on raw ranks) |
| `outliers.csv` | univariate (\|z\| > 3) and multivariate (Mahalanobis with χ²₂₇ 99% threshold) outlier flags |
| `tertile_assignments.csv` | one row per (scene, phase) with each method's continuous score and Easy/Medium/Hard label |
| `method_agreement_kendall_tau.csv` | Kendall τ between methods' continuous scores |
| `method_agreement_ari.csv` | Adjusted Rand Index between methods' tertile labels |
| `phase_composition.csv` | for each (method, tertile, phase) — what fraction of the tertile is that phase (does interaction dominate "Hard"?) |
| `correlation_heatmap.png` | feature × feature correlation matrix |
| `pca_scree.png`<br>`pca_loadings.png` | explained-variance ratio per PC; top-2 PC loadings on each feature |
| `feature_distributions.png` | per-feature box plots by phase (clutter / interaction / clean) |
| `findings.txt` | auto-generated headline: top-correlated pairs, constant features, PC1 explained variance, full agreement matrices, one-line interpretation |

### Interpreting agreement

- **Median Kendall τ across methods > 0.85** → the natural difficulty axis
  is robust; any scoring will rank scenes similarly. Pick `mean_pn` (paper-
  compatible) and move on.
- **0.6 < τ < 0.85** → moderate agreement; ship `mean_pn` for the paper but
  flag the sensitivity in the appendix.
- **τ < 0.6** → genuinely multi-dimensional difficulty signal; the
  MI-weighted RPX-DS (computed once calibration models exist) will do real
  work, and uniform/PCA scorings are unreliable. Wait for calibration.

---

## 6. Important: the constraint splits MUST respect

**All three phases of the same scene must land in the same train/val/test
split** (otherwise STR — state-transition robustness — is undefined).

Both scripts preserve `scene_id` per row so any downstream splitter can
`groupby(scene_id)` and assign at the scene level. Tertile assignments
within Easy/Med/Hard are per-(scene, phase) by design — a scene's clutter
phase can be Easy while its interaction phase is Hard, which is exactly
the signal we want.

---

## 7. Common operational notes

**Safe to re-run.** Output JSON/CSV/log are overwritten each run; no
incremental state. For 100 scenes a clean re-run is a few minutes.

**One warning per directory, not per file.** Multi-channel masks (saved as
RGB by mistake), missing fisheye pairs, and modality misalignments each
emit a single summary warning per phase — log volume is bounded.

**Scene rename is transparent.** `scene_id` is the dir basename; renaming
`scene1.GDC.cse/` → `scene1/` collapses keys from `scene1.GDC.cse.phase0`
to `scene1.phase0` automatically with no code change.

**Hardcoded thresholds** (per paper spec; if you need to sweep them, edit
`benchmark/rpx_benchmark/data/esd.py`):

| Constant | Value | Used by |
|---|---|---|
| `_OCC_HEAVY_THRESHOLD` | 0.3 | `occ_heavy` |
| `_AREA_DROP_THRESHOLD` | 0.5 | `area_drop` |
| `_SPECULAR_LUMA_MIN` | 230 | `specular`, `fisheye_bright` |
| `_DARK_LUMA_MAX` | 30 | `dark`, `fisheye_dark` |

---

## 8. Running the tests

```bash
cd benchmark && python -m pytest tests/ -q
```

CI runs the full suite on Python 3.10 / 3.11 / 3.12 against every push
touching `benchmark/**`. Coverage:

- `tests/test_esd.py` — 35 tests, one per feature plus edge cases
  (multi-channel mask dedupe, missing modalities, both fisheye layouts,
  area-drop spikes, p90 motion spikes, malformed JSON fallback)
- `tests/test_build_esd_splits_cli.py` — 3 CLI smoke tests
  (clean run, missing data-root → exit 2, per-phase failure → exit 1
  with traceback in `summary.failures`)
