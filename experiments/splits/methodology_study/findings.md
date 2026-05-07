# Difficulty Methodology Study — Findings

Generated automatically from 297 (scene, phase) entries.
All numbers below come from the analysis tables in this directory.

## L1. Method comparison

- 11 methods compared.
- Median pairwise Kendall τ across continuous scores: **0.363**
- Median pairwise ARI across categorical tiers: **0.083**

Full matrices: agreement/kendall_tau.csv, agreement/ari.csv

## L2. Is k=3 the right number of clusters?

- K-means silhouette is maximised at **k=2** (silhouette=0.293)
- GMM silhouette is maximised at **k=3** (silhouette=0.238)
- Silhouette at the chosen k=3: kmeans=0.261, gmm=0.238
- Gap statistic recommends **k=2** (smallest k satisfying Tibshirani's rule)

If best-silhouette-k != 3, the data does NOT prefer 3 natural clusters; the
tertile design is a *reporting convention* (33/33/34), not a discovery of
natural structure. This is honest to acknowledge in the paper.

## L3. Stability under perturbations

- **Bootstrap (80% resampling)**: median per-row tier-stability = **1.000** (267 / 297 entries with >95% stability)
- **Weight perturbation (Dirichlet samples)**: median tier-flip rate = **0.348** (33 entries flip >50% of the time)
- **Feature dropout (drop 1/3 random features)**: median tier-flip rate = **0.275** (11 entries flip >50% of the time)

Interpretation:
- High bootstrap stability + low weight-perturbation flip → the tertile assignment
  is robust to the specific weighting choice.
- Low feature-dropout flip → the tertile is robust to small feature-set changes.

## L4. Outliers

- **Multivariate outliers** (Mahalanobis > χ²_F(0.99) threshold): 12
- **Cluster-consensus outliers** (flagged by ≥3 clustering methods): 2

Top-5 by Mahalanobis distance:
  - scene100.jsom.atrium phase=1  d=11.95
  - scene18.su.CHECKERBOARD phase=0  d=10.86
  - scene36.jo.4f phase=2  d=7.86
  - scene42.gh.out phase=2  d=7.69
  - scene74.ecsw.atriumStairs phase=2  d=7.69

## L5. Cross-method consensus

- **24 entries (8.1%)** assigned to 1 distinct tier(s) across all methods
- **104 entries (35.0%)** assigned to 2 distinct tier(s) across all methods
- **169 entries (56.9%)** assigned to 3 distinct tier(s) across all methods

Full-agreement rate: **8.1%**.
Higher = methods converge; lower = the choice of method matters and the
paper should report at least the consensus tier alongside the chosen method.

## L6. Do the phases (clutter/interaction/clean) form distinct clusters?

 method  ari_vs_phase  n_clusters
kmeans3      0.176646           3
kmeans5      0.090299           5
kmeans8      0.181676           8
   gmm3      0.035939           3
  ward3      0.003093           3
 dbscan      0.003463           2

Maximum ARI between any clustering and the phase index: **0.182**.
→ Features have weak phase signal. Most of the variance is genuine difficulty.

## Methodological recommendation (data-driven)

- **Primary method**: `mean_pn` (uniform-weighted percentile mean) remains the
  most reproducible choice given absence of model failure data. The stability
  numbers above quantify how much this matters.
- **Always reported alongside**: `consensus_tier` (majority vote across all
  methods). When `mean_pn` and `consensus_tier` disagree, the paper should
  flag the entry rather than hide the disagreement.
- **k=3 vs natural-k**: if silhouette and gap recommend k≠3, the paper
  acknowledges the 33/33/34 tertile is a *reporting convention*, justified
  by interpretability (Easy/Medium/Hard is canonical in benchmarks) rather
  than by data-driven cluster discovery.

