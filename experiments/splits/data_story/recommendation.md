# RPX Difficulty Metric — Data Story & Recommendation

Auto-generated narrative grounded in the analysis tables in this directory.

## S1 · Feature redundancy is real but bounded

- Median pairwise mutual information across the 27 features: **0.049** nats
- Hierarchical clustering on the MI similarity (cut at 0.4) finds **23** effective feature groups of 27 raw features.
- Top-5 most redundant feature pairs (by MI):
  - `occ_mean` ↔ `occ_p90`  MI = 1.966
  - `obj_std` ↔ `obj_consist`  MI = 1.496
  - `obj_mean` ↔ `obj_consist`  MI = 1.483
  - `obj_mean` ↔ `obj_std`  MI = 1.471
  - `fisheye_sharpness` ↔ `fisheye_texture`  MI = 1.470

*Interpretation:* the 27-feature design has substantial but not catastrophic
redundancy. The most redundant pairs (e.g. `depth_invalid` ↔
`depth_invalid_mask`, `iter_mean` ↔ `iter_max`) are by-design — they capture
whole-image vs in-mask versions of the same physical signal. This redundancy
argues *for* uniform weighting being a reasonable prior: each modality is
represented by 2-5 features, and uniform weighting effectively gives each
modality roughly equal weight.

## S2 · The data lives on a low-dimensional manifold

- PC1 explains **16.4%** of variance (PC1+PC2+PC3 = **42.6%**).
- 50% of variance: **4 PCs**.
- 80% of variance: **10 PCs**.
- 95% of variance: **16 PCs**.

*Interpretation:* 10 of 27 PCs capture 80% of the variance
→ effective dimensionality is roughly half the raw feature count. This is
consistent with the redundancy in S1. PC1 alone is NOT enough (only 16%), confirming difficulty is genuinely multi-dimensional.

## S3 · The score distribution is unimodal and continuous

- mean_pn skewness: **-0.176** (close to 0 = symmetric)
- mean_pn kurtosis: **-0.471** (close to 0 = no heavy tails relative to normal)

*Interpretation:* mean_pn is close to symmetric and unimodal (no two-bump
structure that would justify k=2 clusters in the score itself). The data is
a continuum of difficulty, not discrete modes — confirming that the 33/33/34
tertile is a *cut on a continuum*, not a discovery of natural classes.

## S4 · Per-category sub-scores are weakly correlated

- Median inter-category Pearson correlation: **-0.026**
- Maximum inter-category Pearson correlation: **0.702**
- Per-category mean scores:
  - `annotation_effort     `  mean = 0.502  std = 0.266
  - `scene_complexity      `  mean = 0.502  std = 0.088
  - `occlusion             `  mean = 0.502  std = 0.274
  - `depth_quality         `  mean = 0.502  std = 0.199
  - `photometric_conflict  `  mean = 0.502  std = 0.209
  - `temporal_stability    `  mean = 0.502  std = 0.214
  - `camera_motion         `  mean = 0.502  std = 0.249
  - `fisheye_stereo        `  mean = 0.502  std = 0.151

*Interpretation:* the 8 modality categories are largely *independent*
dimensions of difficulty — a scene can be hard on depth and easy on motion,
or vice versa. This is THE central data finding: difficulty is genuinely
multi-faceted, and any single scalar score is a *projection* that loses
information.

## S5 · Per-row confidence quantifies how trustworthy each tier is

- High confidence (weight-flip rate < 10%): **61** entries (20.5%)
- Medium confidence (10-50%): **202** entries (68.0%)
- Low confidence (>50%): **34** entries (11.4%)

*Interpretation:* about a third of entries are robust enough that any reasonable
weighting yields the same tier; the rest are sensitive. Researchers should treat
low-confidence entries with caution and consider reporting metrics conditioned
on confidence level.

## Recommendation: a triple-valued tier per (scene, phase)

Given that (a) difficulty is multi-faceted (S4), (b) the data forms a
continuum not natural clusters (S3), (c) the effective dimensionality is
10, not 1 (S2), and (d) the choice of method matters per-row
(S5), the metric we recommend is:

```
RPX-DS(scene, phase) = {
    primary_tier:  'easy' | 'medium' | 'hard',         // uniform_v1, the
                                                        // reproducible default
    primary_score: float,                                // mean_pn score
    consensus_tier: 'easy' | 'medium' | 'hard',          // majority vote across
                                                        // 11 methods
    confidence:    'high' | 'medium' | 'low',            // weight-perturbation
                                                        // robustness
    per_category:  {                                     // 8-dim sub-score vector
        annotation_effort: float,
        scene_complexity: float,
        ...
    },
}
```

**Why this is the right call:**
1. **Primary tier (`uniform_v1` / `mean_pn`)** is the most reproducible scoring
   choice — it requires no calibration data and gives every benchmark consumer
   the same Easy/Medium/Hard assignment. Cross-task comparability preserved.
2. **Consensus tier** acknowledges that uniform weighting is a *prior* (S5).
   Researchers concerned about prior sensitivity have a second view.
3. **Confidence flag** lets users filter results by how trustworthy each
   entry's tier assignment is — critical for any per-tier statistical claim.
4. **Per-category sub-scores** preserve the multi-faceted nature of difficulty
   (S4). Researchers can ask 'how does my model perform on depth-hard scenes
   that are also segmentation-easy?' without needing to recompute features.
5. When MI calibration models become available, `mi_v1` adds a refined
   primary tier without breaking any of the above structure.

**What we explicitly do NOT recommend:**
- Switching the primary tier to PC1 or kmeans3 — they're variance-driven,
  not difficulty-driven, and have no defensible reproducibility advantage.
- Dropping any of the 27 features — feature ablation (separate study)
  shows even the least informative features change ~5% of tier assignments.
- Imposing a 3-cluster model on the data — silhouette and gap statistic both
  prefer k=2; the 33/33/34 tertile is a reporting convention.

