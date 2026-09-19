# RPX Relative Camera Pose Estimation (RCPE) Benchmark

## Supervisor handover report

**Report date:** 4 September 2026  
**Dataset:** `IRVLUTD/RPX`  
**Pinned revision:** `2e2a387f7f93e98c177b2e039c141eacda94e5fc`  
**Models evaluated:** VGGT-Ω, DA3, CUT3R, MASt3R, MUSt3R, DUSt3R, Reloc3R, Pi3X, Fast3R and MonST3R

## 1. Current completion status

| Model | Easy | Medium | Hard | Current status |
|---|---:|---:|---:|---|
| VGGT-Ω | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| DA3 | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| CUT3R | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| MASt3R | 6676/6676 | Running | Running | Easy complete; Medium and Hard in progress |
| MUSt3R | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| DUSt3R | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| Reloc3R | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| Pi3X | 6676/6676 | 6715/6715 | 6879/6879 | Complete; Medium CSV repaired after restart duplication |
| Fast3R | 6676/6676 | 6715/6715 | 6879/6879 | Complete |
| MonST3R | 6676/6676 | 6715/6715 | 6879/6879 | Complete |

At the time of this report, **28 of 30 model/split cells are complete**. Pi3X Medium originally contained 11,720 rows after an interrupted run appended predictions to the existing CSV. It was deduplicated to 6,715 unique predictions and its comprehensive metrics were regenerated successfully using the correctly mounted ground-truth pose repository.

## 2. Metric definitions

| Metric | Meaning | Direction |
|---|---|---|
| Rotation mean / median | Geodesic angular difference between predicted and ground-truth relative rotations on SO(3) | Lower is better |
| Translation angle | Angle between predicted and ground-truth translation directions; invariant to translation magnitude | Lower is better |
| Translation error (m) | Euclidean translation error in metres; meaningful only for adapters that provide metric-scale translation | Lower is better |
| Pose max | Per-pair `max(rotation_error_deg, translation_angular_deg)`, averaged over the split | Lower is better |
| AUC@5/10/20 | Normalized area under the empirical pose-accuracy curve up to 5°, 10° or 20° | Higher is better |
| Rotation drift | Accumulated rotational inconsistency along temporal chains | Lower is better |
| JEDI | Direction-aware geometric mean of normalized metric desirabilities | Higher is better |
| Φ | Multivariate phase-effect statistic; values nearer 1 indicate a smaller phase effect | Nearer 1 means more phase-stable |

The primary combined pose error is

\[
e_{\mathrm{pose}} = \max(e_{\mathrm{rotation}}, e_{\mathrm{translation\ direction}}).
\]

AUC is computed from the distribution of this per-pair pose error. It is not simply the percentage of samples below the named threshold.

## 3. Main split-wise RCPE results

`Trans. m` is shown only when the run declared metric translation. A dash means that metric-scale translation was unavailable under that run's protocol.

| Model | Split | Pairs | Rot. mean° | Rot. median° | Trans. angle° | Trans. m | Pose max° ↓ | AUC@5 ↑ | AUC@10 ↑ | AUC@20 ↑ | Rot. drift° ↓ | Chains | Existing latency ms* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUT3R | Easy | 6676 | 0.946 | 0.846 | 11.190 | 0.0296 | 11.199 | 0.137 | 0.335 | 0.569 | 4.714 | 330 | 475.8 |
| CUT3R | Medium | 6715 | 0.990 | 0.881 | 11.065 | 0.0339 | 11.075 | 0.136 | 0.345 | 0.576 | 5.211 | 330 | 507.0 |
| CUT3R | Hard | 6879 | 1.008 | 0.927 | 9.738 | 0.0454 | 9.749 | 0.151 | 0.375 | 0.621 | 6.141 | 340 | 673.0 |
| DA3 | Easy | 6676 | **0.732** | **0.644** | **6.613** | — | **6.625** | **0.225** | **0.475** | **0.697** | **2.577** | 330 | **162.7** |
| DA3 | Medium | 6715 | **0.756** | **0.673** | **7.115** | — | **7.124** | **0.220** | **0.465** | **0.687** | **2.636** | 330 | **163.4** |
| DA3 | Hard | 6879 | **0.765** | 0.710 | **7.370** | — | **7.380** | **0.225** | **0.472** | **0.696** | **2.933** | 340 | **148.1** |
| DUSt3R | Easy | 6676 | 1.207 | 0.814 | 17.044 | — | 17.264 | 0.077 | 0.213 | 0.424 | 5.451 | 330 | 779.6 |
| DUSt3R | Medium | 6715 | 1.038 | 0.853 | 17.550 | — | 17.665 | 0.066 | 0.198 | 0.405 | 4.836 | 330 | 1129.5 |
| DUSt3R | Hard | 6879 | 1.061 | 0.849 | 16.730 | — | 16.814 | 0.068 | 0.194 | 0.402 | 5.607 | 340 | 845.1 |
| Fast3R | Easy | 6676 | 1.057 | 0.912 | 11.031 | — | 11.041 | 0.125 | 0.320 | 0.554 | 5.041 | 330 | 1176.0 |
| Fast3R | Medium | 6715 | 1.065 | 0.939 | 11.552 | — | 11.568 | 0.119 | 0.319 | 0.562 | 5.132 | 330 | 764.2 |
| Fast3R | Hard | 6879 | 1.026 | 0.933 | 10.190 | — | 10.200 | 0.120 | 0.330 | 0.592 | 5.179 | 340 | 498.0 |
| MASt3R | Easy | 6676 | 0.831 | 0.725 | 10.826 | — | 10.837 | 0.167 | 0.373 | 0.581 | 3.229 | 330 | 7566.5 |
| MASt3R | Medium | — | — | — | — | — | Running | — | — | — | — | — | — |
| MASt3R | Hard | — | — | — | — | — | Running | — | — | — | — | — | — |
| MonST3R | Easy | 6676 | 1.290 | 1.006 | 28.151 | — | 28.183 | 0.043 | 0.120 | 0.268 | 6.597 | 330 | 1390.8 |
| MonST3R | Medium | 6715 | 1.301 | 1.067 | 29.845 | — | 29.880 | 0.031 | 0.096 | 0.228 | 6.096 | 330 | 832.4 |
| MonST3R | Hard | 6879 | 1.296 | 1.014 | 26.698 | — | 26.742 | 0.034 | 0.099 | 0.236 | 6.504 | 340 | 839.5 |
| MUSt3R | Easy | 6676 | 0.846 | 0.755 | 11.658 | — | 11.665 | 0.108 | 0.291 | 0.527 | 3.425 | 330 | 459.1 |
| MUSt3R | Medium | 6715 | 0.851 | 0.760 | 11.565 | — | 11.572 | 0.124 | 0.315 | 0.544 | 3.345 | 330 | 537.3 |
| MUSt3R | Hard | 6879 | 0.825 | 0.758 | 10.677 | — | 10.682 | 0.125 | 0.323 | 0.568 | 3.187 | 340 | 654.2 |
| Pi3X | Easy | 6676 | 0.761 | 0.681 | 7.987 | 0.0333 | 7.997 | 0.192 | 0.417 | 0.643 | 2.954 | 330 | 224.6 |
| Pi3X | Medium | 6715 | 0.777 | 0.687 | 8.034 | 0.0428 | 8.043 | 0.191 | 0.420 | 0.647 | 3.069 | 330 | 245.3 |
| Pi3X | Hard | 6879 | 0.771 | **0.702** | 7.706 | 0.0541 | 7.718 | 0.209 | 0.452 | 0.680 | 3.093 | 340 | 312.5 |
| Reloc3R | Easy | 6676 | 0.786 | 0.689 | 7.323 | — | 7.332 | 0.212 | 0.454 | 0.675 | 3.208 | 330 | 515.5 |
| Reloc3R | Medium | 6715 | 0.807 | 0.714 | 7.855 | — | 7.864 | 0.194 | 0.438 | 0.662 | 3.330 | 330 | 571.7 |
| Reloc3R | Hard | 6879 | 0.806 | 0.720 | 7.581 | — | 7.594 | 0.217 | 0.462 | 0.689 | 3.532 | 340 | 560.5 |
| VGGT-Ω | Easy | 6676 | 0.765 | 0.691 | 7.852 | — | 7.862 | 0.193 | 0.420 | 0.645 | 3.205 | 330 | 351.3 |
| VGGT-Ω | Medium | 6715 | 0.779 | 0.703 | 8.398 | — | 8.406 | 0.183 | 0.420 | 0.646 | 2.960 | 330 | 435.1 |
| VGGT-Ω | Hard | 6879 | 0.773 | 0.706 | 8.106 | — | 8.116 | 0.199 | 0.439 | 0.668 | 3.071 | 340 | 372.5 |

\* Existing latency values were collected during the production runs and are **provisional**, because other models were running concurrently. A controlled latency experiment is still required.

## 4. Three-split aggregate ranking

MASt3R is excluded until Medium and Hard finish. Pi3X reflects the repaired Medium result.

| Rank by mean pose error | Model | Mean pose error° ↓ | Mean AUC@20 ↑ | Production-run mean latency ms* |
|---:|---|---:|---:|---:|
| 1 | DA3 | **7.043** | **0.693** | **158.1** |
| 2 | Reloc3R | 7.597 | 0.676 | 549.2 |
| 3 | Pi3X | 7.919 | 0.657 | 260.8 |
| 4 | VGGT-Ω | 8.127 | 0.653 | 386.3 |
| 5 | CUT3R | 10.675 | 0.589 | 551.9 |
| 6 | Fast3R | 10.936 | 0.570 | 812.7 |
| 7 | MUSt3R | 11.307 | 0.547 | 550.2 |
| 8 | DUSt3R | 17.248 | 0.410 | 918.1 |
| 9 | MonST3R | 28.268 | 0.244 | 1020.9 |

DA3 is the strongest model on every completed difficulty split. Reloc3R is consistently second. Pi3X and VGGT-Ω form the next performance group. Translation-direction error, rather than rotation error, dominates the combined pose error for every model.

## 5. Phase-conditioned robustness: clutter (phase 0) to clean (phase 2)

Direct phase-0-to-phase-2 relative-pose pairs are **not valid** in this dataset revision because each capture phase has an unrelated T265 local world frame. The runner correctly requires `--cross-pairs-per-bin 0`. Instead, phase robustness is measured by comparing within-phase RCPE performance between clutter and clean captures. Within-phase relative pose is valid because the arbitrary world-frame transform cancels.

For error deltas, negative is improvement in the clean phase. For AUC deltas, positive is improvement.

| Model | Split | P0 pairs | P2 pairs | ΔRot.° | ΔTrans. angle° | P0 pose° | P2 pose° | ΔPose° | P0 AUC@20 | P2 AUC@20 | ΔAUC@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUT3R | Easy | 3353 | 3323 | -0.086679 | -0.729936 | 10.511583 | 9.780323 | -0.731260 | 0.567486 | 0.612594 | 0.045108 |
| CUT3R | Hard | 3461 | 3418 | -0.058600 | -2.662213 | 10.138013 | 7.480161 | -2.657852 | 0.605899 | 0.666761 | 0.060863 |
| CUT3R | Medium | 3361 | 3354 | -0.128303 | -0.571957 | 10.367806 | 9.795855 | -0.571951 | 0.572337 | 0.617537 | 0.045200 |
| DA3 | Easy | 3353 | 3323 | -0.055268 | -0.968364 | 6.745395 | 5.781128 | -0.964267 | 0.686145 | 0.730537 | 0.044392 |
| DA3 | Hard | 3461 | 3418 | -0.040426 | -2.663214 | 8.179879 | 5.521514 | -2.658365 | 0.681062 | 0.737586 | 0.056524 |
| DA3 | Medium | 3361 | 3354 | -0.061329 | -0.377899 | 6.702085 | 6.329293 | -0.372791 | 0.688425 | 0.718177 | 0.029752 |
| DUSt3R | Easy | 3353 | 3323 | -0.138290 | -1.945416 | 17.139650 | 15.148091 | -1.991559 | 0.415131 | 0.460059 | 0.044927 |
| DUSt3R | Hard | 3461 | 3418 | -0.013010 | -2.013841 | 16.988283 | 14.986968 | -2.001314 | 0.395188 | 0.427933 | 0.032745 |
| DUSt3R | Medium | 3361 | 3354 | -0.063718 | -0.560314 | 16.518920 | 15.978698 | -0.540222 | 0.420213 | 0.425142 | 0.004929 |
| Fast3R | Easy | 3353 | 3323 | -0.006533 | 0.248382 | 10.235633 | 10.486474 | 0.250841 | 0.560101 | 0.589358 | 0.029256 |
| Fast3R | Hard | 3461 | 3418 | -0.026496 | -2.007090 | 10.443930 | 8.435191 | -2.008739 | 0.594789 | 0.614810 | 0.020021 |
| Fast3R | Medium | 3361 | 3354 | -0.122832 | 0.138037 | 10.406096 | 10.541436 | 0.135340 | 0.566541 | 0.593272 | 0.026731 |
| MASt3R | Easy | 3353 | 3323 | -0.040695 | -1.995375 | 11.309301 | 9.318660 | -1.990642 | 0.567680 | 0.622493 | 0.054814 |
| MonST3R | Easy | 3353 | 3323 | -0.055841 | -5.039312 | 29.342669 | 24.324790 | -5.017879 | 0.249723 | 0.311423 | 0.061701 |
| MonST3R | Hard | 3461 | 3418 | -0.047301 | -5.028109 | 27.850894 | 22.823248 | -5.027647 | 0.219444 | 0.283642 | 0.064199 |
| MonST3R | Medium | 3361 | 3354 | -0.199034 | -5.431537 | 31.038928 | 25.571927 | -5.467001 | 0.198048 | 0.275720 | 0.077672 |
| MUSt3R | Easy | 3353 | 3323 | -0.064582 | -1.383651 | 11.662194 | 10.279103 | -1.383090 | 0.523024 | 0.563184 | 0.040160 |
| MUSt3R | Hard | 3461 | 3418 | -0.027741 | -3.026659 | 11.575044 | 8.550934 | -3.024110 | 0.545488 | 0.612880 | 0.067392 |
| MUSt3R | Medium | 3361 | 3354 | -0.056834 | -0.776388 | 11.094696 | 10.320515 | -0.774181 | 0.540324 | 0.581169 | 0.040844 |
| Pi3X | Easy | 3353 | 3323 | -0.048389 | -0.997092 | 7.963453 | 6.964843 | -0.998610 | 0.634198 | 0.683956 | 0.049758 |
| Pi3X | Hard | 3461 | 3418 | -0.054884 | -2.620496 | 8.541984 | 5.920884 | -2.621100 | 0.663473 | 0.719071 | 0.055598 |
| Pi3X | Medium | 3361 | 3354 | -0.102606 | -0.848943 | 7.920726 | 7.074743 | -0.845982 | 0.637398 | 0.684574 | 0.047176 |
| Reloc3R | Easy | 3353 | 3323 | -0.044545 | -0.490999 | 7.118224 | 6.625598 | -0.492627 | 0.670389 | 0.706259 | 0.035870 |
| Reloc3R | Hard | 3461 | 3418 | -0.029081 | -2.431942 | 8.212157 | 5.782354 | -2.429803 | 0.679307 | 0.729149 | 0.049842 |
| Reloc3R | Medium | 3361 | 3354 | -0.060316 | -0.211820 | 7.316338 | 7.106251 | -0.210088 | 0.665564 | 0.692706 | 0.027142 |
| VGGT-Ω | Easy | 3353 | 3323 | -0.057132 | -1.435640 | 8.257834 | 6.826348 | -1.431486 | 0.627135 | 0.684437 | 0.057302 |
| VGGT-Ω | Hard | 3461 | 3418 | -0.059587 | -3.211038 | 9.241246 | 6.033189 | -3.208057 | 0.640636 | 0.716358 | 0.075722 |
| VGGT-Ω | Medium | 3361 | 3354 | -0.085587 | -0.509309 | 7.977185 | 7.468893 | -0.508292 | 0.639049 | 0.682013 | 0.042963 |

### Phase-robustness observations

- Clean-phase AUC@20 is higher in all 28 completed cells.
- Pose error improves in 26 of 28 cells.
- Fast3R is the exception: its Easy and Medium pose errors rise slightly in phase 2, although AUC@20 still improves.
- The largest clean-phase pose improvements occur for MonST3R, especially Medium (-5.467°), despite MonST3R remaining the weakest absolute performer.
- Hard splits frequently show larger clutter-to-clean improvement than Medium splits.

## 6. Φ and JEDI phase analysis

| Model | Split | Φ | p-value | Effect | JEDI | JEDI clutter | JEDI clean | ΔJEDI | AUC@20 clutter | AUC@20 clean | ΔAUC@20 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CUT3R | Easy | 0.953571 | 0.705056 | Small | 0.293928 | 0.269640 | 0.318216 | 0.048576 | 0.547796 | 0.593584 | 0.045788 |
| CUT3R | Hard | 0.924921 | 0.497435 | Medium | 0.324894 | 0.292001 | 0.357787 | 0.065786 | 0.589839 | 0.659201 | 0.069362 |
| CUT3R | Medium | 0.856439 | 0.206166 | Large | 0.297698 | 0.265488 | 0.329909 | 0.064422 | 0.555795 | 0.603350 | 0.047555 |
| DA3 | Easy | 0.869558 | 0.248649 | Medium | 0.419231 | 0.387070 | 0.451391 | 0.064320 | 0.678067 | 0.718538 | 0.040471 |
| DA3 | Hard | 0.862730 | 0.212155 | Medium | 0.418544 | 0.380383 | 0.456704 | 0.076321 | 0.667005 | 0.729834 | 0.062830 |
| DA3 | Medium | 0.888579 | 0.322910 | Medium | 0.411866 | 0.386617 | 0.437116 | 0.050499 | 0.675498 | 0.704683 | 0.029185 |
| DUSt3R | Easy | 0.950779 | 0.684927 | Small | 0.186483 | 0.170029 | 0.202937 | 0.032908 | 0.400670 | 0.447218 | 0.046548 |
| DUSt3R | Hard | 0.961763 | 0.755696 | Small | 0.171822 | 0.158478 | 0.185166 | 0.026688 | 0.387130 | 0.421561 | 0.034431 |
| DUSt3R | Medium | 0.985630 | 0.934636 | Small | 0.171657 | 0.166766 | 0.176547 | 0.009782 | 0.398706 | 0.413571 | 0.014865 |
| Fast3R | Easy | 0.950360 | 0.681924 | Small | 0.278234 | 0.262723 | 0.293745 | 0.031022 | 0.542701 | 0.569867 | 0.027166 |
| Fast3R | Hard | 0.950872 | 0.673981 | Small | 0.285291 | 0.270413 | 0.300169 | 0.029756 | 0.579871 | 0.607925 | 0.028054 |
| Fast3R | Medium | 0.950261 | 0.681211 | Small | 0.275999 | 0.259655 | 0.292344 | 0.032689 | 0.548333 | 0.582815 | 0.034483 |
| MASt3R | Easy | 0.974616 | 0.859413 | Small | 0.326654 | 0.303900 | 0.349389 | 0.045489 | 0.555397 | 0.608507 | 0.053110 |
| MonST3R | Easy | 0.900409 | 0.377430 | Medium | 0.107749 | 0.096910 | 0.118589 | 0.021679 | 0.245461 | 0.296263 | 0.050801 |
| MonST3R | Hard | 0.861490 | 0.208281 | Medium | 0.089069 | 0.078889 | 0.099248 | 0.020359 | 0.206089 | 0.268429 | 0.062341 |
| MonST3R | Medium | 0.805320 | 0.094363 | Large | 0.084273 | 0.057832 | 0.110713 | 0.052881 | 0.187644 | 0.273211 | 0.085567 |
| MUSt3R | Easy | 0.910748 | 0.430641 | Medium | 0.249359 | 0.230754 | 0.267964 | 0.037210 | 0.506026 | 0.551445 | 0.045419 |
| MUSt3R | Hard | 0.921210 | 0.474892 | Medium | 0.280744 | 0.248422 | 0.313066 | 0.064643 | 0.533426 | 0.609102 | 0.075676 |
| MUSt3R | Medium | 0.928163 | 0.532264 | Medium | 0.274377 | 0.246005 | 0.302750 | 0.056745 | 0.523310 | 0.572821 | 0.049511 |
| Pi3X | Easy | 0.931660 | 0.554444 | Medium | 0.368566 | 0.335578 | 0.401555 | 0.065977 | 0.617928 | 0.670791 | 0.052863 |
| Pi3X | Hard | 0.923095 | 0.486254 | Medium | 0.399606 | 0.369369 | 0.429842 | 0.060473 | 0.652034 | 0.714085 | 0.062050 |
| Pi3X | Medium | 0.926787 | 0.523693 | Medium | 0.371077 | 0.343130 | 0.399025 | 0.055894 | 0.627004 | 0.672598 | 0.045593 |
| Reloc3R | Easy | 0.918050 | 0.471429 | Medium | 0.399805 | 0.376106 | 0.423505 | 0.047398 | 0.659488 | 0.692181 | 0.032694 |
| Reloc3R | Hard | 0.913165 | 0.428528 | Medium | 0.409804 | 0.378548 | 0.441059 | 0.062511 | 0.663673 | 0.720967 | 0.057293 |
| Reloc3R | Medium | 0.893469 | 0.344640 | Medium | 0.382195 | 0.354974 | 0.409416 | 0.054442 | 0.648446 | 0.682297 | 0.033851 |
| VGGT-Ω | Easy | 0.955230 | 0.717114 | Small | 0.371077 | 0.340681 | 0.401473 | 0.060792 | 0.616404 | 0.677508 | 0.061104 |
| VGGT-Ω | Hard | 0.845306 | 0.162980 | Large | 0.386875 | 0.346351 | 0.427400 | 0.081049 | 0.630166 | 0.711461 | 0.081295 |
| VGGT-Ω | Medium | 0.936848 | 0.588378 | Medium | 0.366737 | 0.345330 | 0.388143 | 0.042814 | 0.631213 | 0.666485 | 0.035272 |

### Statistical interpretation

- All 28 reported p-values are above 0.05; none of the per-model/per-split multivariate phase effects reaches conventional statistical significance.
- Effect-size labels nevertheless range from Small to Large. The largest effects occur for CUT3R Medium, MonST3R Medium and VGGT-Ω Hard.
- JEDI is higher in clean phase 2 for every completed cell.
- DA3 has the highest JEDI on all three completed difficulty splits: Easy 0.419, Medium 0.412 and Hard 0.419.
- Effect magnitude and statistical significance must be reported separately: a Large label with `p > 0.05` is not evidence of a statistically significant phase effect.

## 7. Efficiency metadata

The following values were recovered from the Easy runs. Parameters and estimated memory traffic are hardware-agnostic. Latency percentiles are provisional production-run measurements. The reported 47.4 GB system-card value is GPU capacity, not measured peak allocation.

| Model | Parameters (M) | Estimated memory traffic (GB) | Latency p50 (ms)* | Latency p95 (ms)* |
|---|---:|---:|---:|---:|
| VGGT-Ω | 1256.536 | 15.0784 | 351.433 | 405.737 |
| DA3 | 1355.674 | 16.2681 | 161.673 | 196.964 |
| CUT3R | 463.439 | 5.5613 | 458.847 | 643.648 |
| MASt3R | 688.639 | 8.2637 | 7896.389 | 9021.746 |
| MUSt3R | 423.430 | 5.0812 | 447.788 | 599.271 |
| DUSt3R | 571.171 | 6.8541 | 768.378 | 994.450 |
| Reloc3R | 426.524 | 5.1183 | 511.315 | 678.720 |
| Pi3X | 992.465 | 11.9096 | 212.095 | 312.586 |
| Fast3R | 647.551 | 7.7706 | 800.149 | 3581.831 |
| MonST3R | 268.074 | 3.2169 | 1013.502 | 3632.522 |

\* These latency values must not be used as the final controlled hardware comparison.

## 8. Protocol and data-quality notes

1. **Direct cross-phase RCPE is unavailable.** Phase 0 and phase 2 are separate captures with unrelated T265 local coordinate systems. A direct cross-phase relative pose would require a validated per-scene inter-phase registration transform. The current runner correctly refuses nonzero `--cross-pairs-per-bin`.
2. **Phase-conditioned analysis is valid.** All 28 completed cells contain both phase 0 and phase 2 measurements. Their within-phase relative poses do not depend on a common global origin.
3. **Difficulty is not monotonic in the reported scores.** Several models score better on Hard than Easy. The difficulty labels should therefore be described as dataset strata, not assumed to be a strictly increasing scalar difficulty scale.
4. **Translation direction dominates pose error.** Rotation errors remain around 0.7°–1.3°, while translation-direction errors range much more widely.
5. **Metric AUC status.** The joint rotation-plus-metric-translation AUC was marked `skipped_by_protocol`; the reported AUC@5/10/20 values are the standard angular pose AUCs.
6. **Archive integrity.** The downloaded summary archive matched its SHA-256 sidecar but failed `gzip -t` with `unexpected end of file`. This means the checksummed source archive itself was truncated. The summary TSVs were recoverable, but a new final archive must be created after all runs finish.
7. **Pi3X Medium repair.** Restart duplication was removed by retaining one prediction per unique sample identifier. Comprehensive metrics were regenerated only after mounting the RPX snapshot at the path expected by the container.
8. **Restart behavior.** The RCPE CSV writer appends across process restarts. Any restarted split must be deduplicated and validated before metrics are regenerated or uploaded.

## 9. Remaining work before publication

- [ ] Allow MASt3R Medium and Hard to finish.
- [ ] Deduplicate MASt3R Medium and Hard after completion if their runs appended to partial CSVs.
- [ ] Confirm exactly 6,715 and 6,879 unique MASt3R predictions, respectively.
- [ ] Regenerate MASt3R `pose_comprehensive_metrics.json` after deduplication.
- [ ] Rerun the main, phase and Φ/JEDI extraction after all 30 cells validate.
- [ ] Conduct controlled latency runs: one model at a time, same physical GPU, batch size 1, identical manifest and resolution, no concurrent GPU workload, and preferably three repetitions.
- [ ] Report p50 and p95 latency together with hardware, CUDA, PyTorch, precision and model-image revision.
- [ ] Rebuild and test the final results archive with `gzip -t` and `tar -tzf` before generating its SHA-256 file.
- [ ] Upload only validated outputs to Box. Avoid simultaneous tracking and RCPE upload processes sharing the same OAuth token cache.

## 10. Interim conclusion

The completed benchmark cells consistently identify **DA3 as the strongest RCPE model**, followed by **Reloc3R**, with **Pi3X and VGGT-Ω forming the next tier**. Clean-phase performance is generally better than clutter-phase performance: AUC@20 and JEDI improve in every completed cell, while combined pose error improves in 26 of 28 cells. None of the individual Φ tests is significant at `p < 0.05`, so the observed phase changes should be described as effect-size trends rather than statistically established phase effects.

The task-performance results are suitable as an interim supervisor update. Final paper-ready reporting remains contingent on completing and validating MASt3R, standardizing latency measurement, rerunning the consolidated tables, and producing a valid non-truncated archive.

## 11. Evidence used for this handover

- Recovered `rcpe-paper-metrics.tsv`, `rcpe-all-scalar-metrics.tsv` and validation table from the locally downloaded RCPE summary archive.
- Server-generated phase-robustness table shown in `Screenshot_2026-09-04_15-20-06.png`.
- Server-generated Φ/JEDI table shown in `Screenshot_2026-09-04_15-20-25.png`.
- Pi3X Medium post-repair validation: 6,715 unique predictions and 6,715 reconstructed comprehensive-metric pairs.
