# RPX SOS AR-tag versus camera-pose audit: metric explanations

## What this audit does

The AR-tag board is physically stationary during each single-object (SOS)
sequence. For every usable RGB frame, the audit:

1. detects the known markers on the board;
2. estimates the board pose relative to the D435 RGB camera;
3. reads the synchronized T265 camera-to-world pose;
4. converts the raw T265 axes to the OpenCV convention; and
5. composes the two transforms:

   ```text
   world_from_board = world_from_camera × camera_from_board
   ```

If all measurements, calibration and synchronization were perfect, the
reconstructed board pose would remain constant. Its apparent movement is the
end-to-end inconsistency measured by this audit.

Dataset: `IRVLUTD/RPX`  
Revision: `2e2a387f7f93e98c177b2e039c141eacda94e5fc`

## What is a valid AR-tag pose?

A frame has a **valid AR-tag board pose** when:

- at least two known markers from the 18-marker board are detected;
- RANSAC finds a geometrically consistent pose from their corners; and
- the estimated pose has at most 2 pixels reprojection RMSE.

Frames that fail these checks are excluded from the primary pose comparison.
They are still recorded in `per_frame.csv` so that failure coverage is visible.

| Coverage measure | Result | Meaning |
|---|---:|---|
| Total frames inspected | 35,000 | 70 SOS sequences × 500 frames |
| Any raw tag pose | 16,731 (47.8%) | At least one marker-based pose was available |
| Strict valid board pose | 11,317 (32.3%) | Passed the two-marker and reprojection gates |
| Sequences with detections | 70/70 | Every SOS sequence contributed measurements |

Low coverage does not directly mean that the T265 pose is bad. The board may be
outside the image, occluded, blurred, too oblique or too small.

## Main metrics

### Translation jitter

Translation jitter is the Euclidean distance between a frame's reconstructed
world-space board position and the robust central board position for that SOS
sequence. The central position is the coordinate-wise median.

| Statistic | Result | Interpretation |
|---|---:|---|
| Median | 0.0468 m (4.68 cm) | Typical accepted-frame displacement |
| P95 | 0.1181 m (11.81 cm) | 95% of accepted frames lie within this distance |
| RMSE | 0.0762 m (7.62 cm) | Quadratic average, emphasizing large residuals |
| Robust-inlier P95 | 0.0880 m (8.80 cm) | P95 after automatic residual-outlier rejection |

### Rotation jitter

Rotation jitter is the geodesic angular difference between a frame's
reconstructed board orientation and the sequence's rotation medoid.

| Statistic | Result | Interpretation |
|---|---:|---|
| Median | 1.526° | Typical accepted-frame orientation disagreement |
| P95 | 5.886° | 95% of accepted frames lie within this angle |
| RMSE | 11.657° | Strongly increased by rare large PnP orientation branches |
| Robust-inlier P95 | 3.465° | P95 after automatic residual-outlier rejection |

The large difference between rotation RMSE and median/P95 indicates a
heavy-tailed error distribution. A small number of planar-PnP orientation
failures have a disproportionate effect on RMSE.

### Correlation

Correlation measures whether the AR-derived motion and the stored T265 motion
follow the same pattern. It ranges from `-1` to `1`:

- `1`: extremely similar variation;
- `0`: no linear relationship;
- `-1`: consistently opposite variation.

| Metric | Result | Interpretation |
|---|---:|---|
| Translation correlation | 0.9792 | AR-derived and stored translations follow very similar trajectories |
| Translation-magnitude correlation | 0.9549 | Their overall displacement magnitudes also agree strongly |
| Rotation-vector correlation | 0.9724 | Their orientation changes follow similar patterns |
| Rotation-magnitude correlation | 0.9952 | Their angular-motion magnitudes agree extremely strongly |

High correlation does **not** mean that the metric poses are identical. Two
tracks may be highly correlated while retaining a scale, timing, calibration
or constant-transform error.

### Outlier rejection

An outlier is an accepted frame whose translation or rotation residual is more
than 3.5 scaled median absolute deviations above its sequence median. This is a
robust rule that does not assume Gaussian errors.

| Metric | Result |
|---|---:|
| Robust residual outlier rate | 8.32% |
| Translation P95 before/after rejection | 11.81 cm / 8.80 cm |
| Rotation P95 before/after rejection | 5.89° / 3.47° |

Outliers remain in `per_frame.csv` with
`static_board_robust_outlier=True`; they are not silently deleted. Likely
causes include planar-PnP ambiguity, motion blur, poor corners, partial
occlusion, synchronization mismatch, calibration error or a T265 pose jump.

## Supporting metrics

### Reprojection RMSE

The detected 2D marker corners are compared with the corners obtained by
projecting the known 3D board geometry through the estimated pose.

| Metric | Result |
|---|---:|
| Mean reprojection RMSE | 0.8567 px |
| Strict acceptance threshold | 2.0 px |

Lower is better. A low reprojection error means the PnP solution fits the
observed image corners, but it does not guarantee the correct 3D orientation
because a planar board can admit ambiguous pose branches.

### Mean marker count

| Metric | Result |
|---|---:|
| Mean inlier markers per valid pose | 3.14 |

Using multiple markers constrains the board pose better than relying on one
marker.

### Anchor-relative error (ATE-style)

The first valid frame anchors both motion tracks. Later AR-derived and stored
poses are expressed relative to this anchor and compared.

| Metric | Result |
|---|---:|
| Anchor-relative translation RMSE | 0.1705 m |
| Anchor-relative rotation RMSE | 10.119° |

These measure longer-term trajectory disagreement, but they are sensitive to a
bad first observation and accumulated error. They are described as ATE-style
because no best-fit sensor transform is learned during evaluation.

### Static-board drift RMSE

This directly tests how constant the composed world-space board pose remains
relative to its first accepted observation.

| Metric | Result |
|---|---:|
| Translation drift RMSE | 0.0933 m |
| Rotation drift RMSE | 10.119° |

Zero would represent perfect end-to-end consistency.

### Step error (RPE-style)

The audit independently computes motion between consecutive valid AR
observations and between their corresponding stored T265 poses, then compares
the two relative transforms.

| Metric | Result |
|---|---:|
| Step translation RMSE | 0.1397 m |
| Step rotation RMSE | 9.366° |
| Gap-normalized translation RMSE | 0.1144 m/frame |
| Gap-normalized rotation RMSE | 7.862°/frame |

Valid AR detections are not always adjacent source frames. The gap-normalized
version divides each measurement by its source-frame gap. These values remain
sensitive to occasional PnP failures and should not be presented as pure
smooth-frame T265 noise.

### Drift slope

A least-squares trend is fitted through the reconstructed board residual over
source-frame index.

| Metric | Result |
|---|---:|
| Translation drift slope | 0.000230 m/frame |
| Rotation drift slope | 0.016576°/frame |

This detects systematic growth in disagreement. It is an aggregate across
sequences and is not a claim that every sequence drifts at this exact rate.

## Overall conclusion

The high translation and rotation correlations show that the AR-derived motion
and stored T265 poses follow the same underlying camera trajectory. The
non-zero board jitter shows measurable end-to-end disagreement.

The measured jitter must **not** be called pure T265 error. It contains:

```text
T265 pose error
+ D435/T265 synchronization error
+ AR-tag corner and PnP uncertainty
+ approximate D435 intrinsics/distortion error
+ any unmodelled D435-to-T265 rigid transform
```

The released RPX data does not include the capture device's exact distortion
coefficients or a numerical D435-to-T265 extrinsic. Consequently, the correct
claim is:

> The stored T265 poses and the independent AR-board trajectory are strongly
> correlated. On strictly accepted frames, the reconstructed stationary board
> has 4.68 cm / 1.53° median end-to-end inconsistency, with robust-inlier P95
> values of 8.80 cm / 3.47°. These are upper bounds on camera-pose error, not
> isolated T265 accuracy measurements.

## What other RCPE and trajectory benchmarks report

The literature uses two related but different evaluation families.

### Relative camera-pose estimation (RCPE)

RCPE benchmarks generally evaluate image pairs. Common measurements are:

| Literature metric | Definition | RPX audit equivalent/status |
|---|---|---|
| Relative rotation error (RRE) | Geodesic angular error between predicted and reference relative rotations | Implemented as step/anchor rotation error |
| Relative translation angular error (RTAE) | Angle between predicted and reference translation directions | Not the primary RPX measure because both RPX sources provide metric translation |
| Metric translation error | Euclidean translation error when scale is available | Implemented as step/anchor translation error |
| Mean and median error | Central error over all evaluated pairs | RMSE and robust median/P95 are reported |
| RRA@θ | Percentage of pairs with rotation error below an angular threshold | Can be derived automatically from `per_frame.csv`; not currently used as the headline |
| RTA@θ | Percentage of pairs with translation-direction error below a threshold | Appropriate if RPX is evaluated as direction-only RCPE |
| mAA/AUC@θ | Area under an accuracy-versus-error-threshold curve | Recommended when comparing future RCPE models |
| Estimate/detection rate | Fraction of inputs for which the method returns a valid pose | Implemented as raw and strict AR-pose coverage |
| Reprojection error/VCRE | Image-plane consequence of pose error | Marker reprojection RMSE is implemented; VCRE is useful for NVS/AR model evaluation |

Examples from recent RCPE work:

- **Wide-Baseline Relative Camera Pose Estimation with Directional Learning**
  reports mean and median geodesic errors separately for rotation and
  translation direction.
- **Reloc3r** reports relative rotation accuracy at 15 degrees (`RRA@15`),
  relative translation accuracy at 15 degrees (`RTA@15`) and mean average
  accuracy/AUC up to 30 degrees (`mAA@30` / `AUC@30`).
- **MicKey / Map-free relative pose evaluation** reports virtual-correspondence
  reprojection-error AUC and precision, median translation in metres, median
  rotation in degrees, and the percentage of inputs producing an estimate.

### Visual odometry and trajectory datasets

Trajectory benchmarks evaluate a sequence rather than independent image pairs:

| Literature metric | Definition | RPX audit equivalent |
|---|---|---|
| Absolute Trajectory Error (ATE) | Global trajectory inconsistency after accounting for coordinate-frame alignment | Anchor-relative and static-board drift metrics |
| Relative Pose Error (RPE) | Difference between relative motions over a fixed frame/time interval | Step translation and rotation error |
| Drift per frame/second/metre | RPE normalized by temporal or travelled-distance interval | Gap-normalized step error and drift slope |
| RMSE/mean/median/std/max | Distribution summaries of pose residuals | Implemented, including robust P95 |

The TUM RGB-D benchmark recommends RPE for visual odometry and ATE for global
trajectory consistency. EuRoC additionally emphasizes that camera intrinsics,
camera-to-sensor extrinsics and temporal alignment must be released or known
for a defensible ground-truth comparison.

### Fiducial-marker studies

Fiducial-marker comparisons usually report:

- pose accuracy;
- repeatability or standard deviation while the target is stationary;
- detection/success rate;
- image reprojection or corner error;
- translation and rotation error versus viewing distance and angle;
- robustness under blur, occlusion, shadows and oblique views; and
- sometimes runtime.

This is why the RPX audit reports fixed-board translation/rotation dispersion,
coverage, marker count, reprojection error and outlier rate together. Reporting
only the camera-pose residual would incorrectly treat the AR solver as perfect
ground truth.

## Recommended automated RPX protocol

For the current **AR-board versus stored camera-pose validation**, report these
as the primary table:

1. strict valid-pose coverage;
2. median, RMSE and P95 static-board translation error in centimetres;
3. median, RMSE and P95 static-board rotation error in degrees;
4. RPE translation and rotation at fixed frame gaps;
5. robust outlier rate;
6. marker reprojection RMSE and marker count; and
7. results stratified by board distance, viewing angle and SOS sequence.

Use correlation only as secondary evidence that both sources follow the same
trajectory. It is not a standard RCPE accuracy measurement and can remain high
despite metric bias.

When actual RCPE models are benchmarked on RPX image pairs, add:

- rotation-error median and `RRA@5/10/15°`;
- translation-direction-error median and `RTA@5/10/15°`;
- `mAA/AUC@30°` over the maximum of rotation and translation-direction error;
- metric translation error when the model predicts scale; and
- valid-estimate rate and runtime.

This provides an automated protocol consistent with both RCPE papers and
trajectory-estimation benchmarks, rather than relying on visual inspection.

### References

- TUM RGB-D trajectory evaluation tools and ATE/RPE:
  https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools
- Sturm, Burgard and Cremers, *Evaluating Egomotion and
  Structure-from-Motion Approaches Using the TUM RGB-D Benchmark*:
  https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros_ws.pdf
- Chen et al., *Wide-Baseline Relative Camera Pose Estimation with
  Directional Learning*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Wide-Baseline_Relative_Camera_Pose_Estimation_With_Directional_Learning_CVPR_2021_paper.pdf
- Dong et al., *Reloc3r*, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.pdf
- Barroso-Laguna et al., *Matching 2D Images in 3D: Metric Relative Pose from
  Metric Correspondences*, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Barroso-Laguna_Matching_2D_Images_in_3D_Metric_Relative_Pose_from_Metric_CVPR_2024_paper.pdf
- Kalaitzakis et al., *Fiducial Markers for Pose Estimation*, 2021:
  https://doi.org/10.1007/s10846-020-01307-9
- EuRoC MAV dataset and calibration/ground-truth specification:
  https://projects.asl.ethz.ch/datasets/euroc-mav/

## Result files

- `report.md`: concise results.
- `summary.json`: complete machine-readable metrics.
- `per_sequence.csv`: object-wise metrics.
- `per_frame.csv`: detection decisions and frame-wise residuals.
- `overview.pdf`: aggregate visualization.
- `per_sequence_diagnostics.pdf`: one diagnostic page per SOS sequence.
