# RPX Benchmark: Comprehensive Metrics per Task

**Date**: 2025-05-07  
**Purpose**: Exhaustive list of all standard evaluation metrics for each task in the RPX benchmark. Compute everything; select later for publication.

---

## 1. Monocular Depth Estimation

### Currently implemented (rpx_benchmark/metrics/depth.py)
- AbsRel, RMSE, δ1, δ2, δ3

### Full metric set to implement

#### Error metrics (lower is better)

| Metric | Formula | Used by |
|--------|---------|---------|
| **AbsRel** | (1/N) Σ \|d̂ᵢ - dᵢ\| / dᵢ | All papers |
| **SqRel** | (1/N) Σ (d̂ᵢ - dᵢ)² / dᵢ | Eigen et al. 2014, KITTI, DA-V2, Metric3D |
| **RMSE** | √((1/N) Σ (d̂ᵢ - dᵢ)²) | All papers |
| **RMSElog** | √((1/N) Σ (log d̂ᵢ - log dᵢ)²) | Eigen et al. 2014, NYU, DA-V2 |
| **SIlog** (scale-invariant log) | √((1/N) Σ eᵢ² - (1/N² (Σ eᵢ)²)) × 100, where eᵢ = log d̂ᵢ - log dᵢ | KITTI benchmark (primary ranking metric) |
| **log10** | (1/N) Σ \|log₁₀ d̂ᵢ - log₁₀ dᵢ\| | NYU v2, some indoor papers |
| **iRMSE** | RMSE computed on inverse depth (1/d) | KITTI depth prediction benchmark |
| **iMAE** | MAE computed on inverse depth (1/d) | KITTI depth prediction benchmark |
| **MAE** | (1/N) Σ \|d̂ᵢ - dᵢ\| | MDEC challenge, BenchDepth |

#### Accuracy / threshold metrics (higher is better)

| Metric | Formula | Used by |
|--------|---------|---------|
| **δ1** (δ < 1.25) | % pixels where max(d̂/d, d/d̂) < 1.25 | All papers |
| **δ2** (δ < 1.25²) | % pixels where max(d̂/d, d/d̂) < 1.5625 | All papers |
| **δ3** (δ < 1.25³) | % pixels where max(d̂/d, d/d̂) < 1.953125 | All papers |

#### Boundary / edge metrics (from MDEC and Depth Pro)

| Metric | Formula / Description | Used by |
|--------|----------------------|---------|
| **F-score** | Harmonic mean of accuracy and completeness at edge pixels | MDEC challenge (4th edition) |
| **Accuracy (edges)** | % predicted edges within threshold of GT edges | MDEC |
| **Completeness (edges)** | % GT edges that have a predicted edge nearby | MDEC |
| **ORD** (ordinal error) | % pixel pairs where depth ordering is wrong | Some ordinal-focused papers |

#### Alignment variants

For relative-depth models (Marigold, Lotus, DA-V2 base), metrics must be computed after alignment:

| Alignment | Description | When to use |
|-----------|-------------|-------------|
| **None** | Directly compare predicted vs GT (requires metric output) | UniDepth V2, Depth Pro, DA-V2 Metric, Metric3D v2, MoGe-2, MetricSolver, ZoeDepth |
| **Median scaling** | pred_aligned = pred × median(GT) / median(pred) | Legacy protocol |
| **Least-squares (affine)** | pred_aligned = a × pred + b (fit a,b by least squares) | MDEC 4th edition (new standard) |
| **Least-squares in disparity** | Same but fit on 1/d | Some papers (Lotus, Marigold) |

#### Per-region / stratified metrics

| Stratification | Description |
|----------------|-------------|
| **Near / Mid / Far** | Split valid pixels by depth range (e.g., 0–1m, 1–3m, 3m+) |
| **Object vs Background** | Use segmentation masks to separate |
| **Edge vs Interior** | Canny/Sobel on GT depth to separate boundary pixels |
| **Per-phase** (RPX-specific) | Clutter / Interaction / Clean phase breakdown |

#### Robotics-specific metrics

| Metric | Description | Why it matters |
|--------|-------------|----------------|
| **Planarity error** | RMSE of plane fit residuals on known-flat surfaces (table) | Grasp planning assumes flat support |
| **Object-relative depth ordering** | % object pairs with correct relative depth | Pick sequence planning |
| **Depth discontinuity recall** | % object boundaries where depth discontinuity is detected | Segmentation from depth |

**Reference for metric definitions**: Eigen et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS 2014 (arXiv:1406.2283)

---

## 2. Instance Segmentation

### Currently implemented (rpx_benchmark/metrics/segmentation.py)
- mIoU

### Full metric set to implement

| Metric | Formula / Description | Used by |
|--------|----------------------|---------|
| **mIoU** | Mean IoU across GT classes | Standard |
| **AP** (Average Precision) | Area under PR curve at IoU=0.5 | COCO |
| **AP50** | AP at IoU threshold 0.50 | COCO, VOC |
| **AP75** | AP at IoU threshold 0.75 | COCO |
| **AP (COCO)** | Mean AP at IoU thresholds [0.50:0.05:0.95] | COCO primary |
| **APˢ / APᴹ / APᴸ** | AP for small / medium / large objects | COCO |
| **PQ** (Panoptic Quality) | Recognition quality × segmentation quality | Panoptic segmentation |
| **SQ** (Segmentation Quality) | Mean IoU of matched segments | Panoptic |
| **RQ** (Recognition Quality) | F1 of matched segments | Panoptic |
| **Boundary IoU** | IoU computed only at mask boundaries (within d pixels of edge) | Boundary-sensitive eval |
| **Mask AP** | AP using mask IoU rather than box IoU | COCO instance seg |
| **F1 per class** | Per-category F1 score | Balanced eval |
| **FWIoU** (Frequency Weighted IoU) | IoU weighted by class pixel frequency | Semantic seg |
| **Pixel Accuracy** | % correctly classified pixels | Semantic seg |
| **Mean Accuracy** | Mean per-class pixel accuracy | Semantic seg |

#### RPX-specific stratification
- Per-phase (clutter / interaction / clean)
- Per-object-size (small < 32² px, medium < 96² px, large ≥ 96² px following COCO)
- Occluded vs non-occluded instances
- Known vs novel categories (for open-vocab setting)

---

## 3. Object Detection

### Currently implemented (rpx_benchmark/metrics/detection.py)
- Precision, Recall, F1 at IoU=0.5

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **mAP** | Mean AP across classes (IoU=0.5) | VOC |
| **mAP@[.5:.95]** | COCO-style mean AP at 10 IoU thresholds | COCO primary |
| **AP50** | AP at IoU=0.50 | Standard |
| **AP75** | AP at IoU=0.75 | COCO |
| **APˢ / APᴹ / APᴸ** | AP by object size | COCO |
| **AR₁ / AR₁₀ / AR₁₀₀** | Average recall given 1 / 10 / 100 detections per image | COCO |
| **Precision@IoU=0.5** | Already implemented | |
| **Recall@IoU=0.5** | Already implemented | |
| **F1@IoU=0.5** | Already implemented | |
| **mAP per category** | Per-class AP breakdown | All |
| **Localization error** | Mean IoU of TP detections (how tight are the boxes) | Diagnosis |
| **Classification error rate** | % TPs with wrong label | Diagnosis |
| **Miss rate vs FPPI** | At various false-positives-per-image rates | Pedestrian detection |

---

## 4. Relative Camera Pose Estimation

### Currently implemented (rpx_benchmark/metrics/pose.py)
- Rotation geodesic error (degrees), Translation L2 error (meters)

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **Rotation error (deg)** | Geodesic: arccos((tr(R_pred^T R_gt) - 1) / 2) | All pose papers |
| **Translation error (m)** | L2 norm of translation difference | All pose papers |
| **Translation angular error (deg)** | Angle between predicted and GT translation directions | MapFree, 7Scenes |
| **AUC@5°** | Area under recall curve at 5° rotation threshold | FAR, RelPose |
| **AUC@10°** | Area under recall curve at 10° rotation threshold | Standard |
| **AUC@20°** | Area under recall curve at 20° rotation threshold | Standard |
| **Median rotation error** | Median across samples (robust to outliers) | Visual localization |
| **Median translation error** | Median across samples | Visual localization |
| **Pose recall@(X°, Ym)** | % pairs with rot < X° AND trans < Ym | ScanNet, MapFree |
| **Relative pose AUC** | AUC of the cumulative recall vs threshold curve | 8-point paper |
| **Angular translation error** | arccos(t_pred · t_gt / (\|t_pred\| \|t_gt\|)) | When scale is ambiguous |

#### Thresholded recall at standard cutoffs
- (5°, 5cm), (5°, 10cm), (10°, 25cm), (15°, 50cm) — indoor
- (5°, 2m), (10°, 5m) — outdoor

---

## 5. Keypoint Matching

### Currently implemented (rpx_benchmark/metrics/keypoints.py)
- Keypoint accuracy (% within px_threshold), Mean match error

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **Match accuracy @3px** | % correspondences within 3px of GT | ScanNet protocol |
| **Match accuracy @5px** | % within 5px | LoFTR, SuperGlue |
| **Match accuracy @10px** | % within 10px | Relaxed threshold |
| **Mean matching error (px)** | Already implemented | |
| **Precision** | TP / (TP + FP) for matches | SuperGlue |
| **Recall** | TP / (TP + FN) for matches | SuperGlue |
| **Repeatability** | % keypoints detected in both views | Feature detectors |
| **Homography estimation AUC** | AUC of corner reprojection error under estimated H | HPatches |
| **Pose estimation AUC@5/10/20°** | From estimated E/F matrix | MegaDepth, ScanNet |
| **Number of inliers** | After RANSAC | Practical measure |
| **Inlier ratio** | Inliers / total matches | Practical measure |

---

## 6. Visual Grounding

### Currently implemented (rpx_benchmark/metrics/grounding.py)
- Grounding IoU, Grounding accuracy@0.5

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **Acc@IoU=0.5** | Already implemented | RefCOCO standard |
| **Acc@IoU=0.25** | Relaxed threshold | Some papers |
| **Acc@IoU=0.75** | Strict threshold | |
| **Mean IoU** | Average IoU of top prediction with GT | Already implemented |
| **Top-5 accuracy** | Correct box in top-5 predictions | Multi-box outputs |
| **Pointing accuracy** | Center of predicted box within GT box | Pointing game |
| **GIoU** (Generalized IoU) | IoU that accounts for non-overlapping boxes | Detection literature |
| **Per-query-type accuracy** | Breakdown by query complexity | Compositional |
| **Unique vs Non-unique accuracy** | How well it handles ambiguous references | RefCOCO+ |

---

## 7. Object Tracking

### Currently implemented (rpx_benchmark/metrics/tracking.py)
- MOTA, IDF1, FP, FN, IDsw

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **MOTA** | 1 - (FN + FP + IDsw) / GT | MOT standard |
| **MOTP** | Mean IoU of matched detections | MOT standard |
| **IDF1** | ID F1-score (identity-aware) | MOT standard |
| **HOTA** | Higher Order Tracking Accuracy (geometric mean of detection + association) | MOT20/TAO |
| **DetA** | Detection accuracy component of HOTA | MOT20 |
| **AssA** | Association accuracy component of HOTA | MOT20 |
| **MT** (Mostly Tracked) | % GT tracks tracked for ≥80% of lifetime | MOT |
| **ML** (Mostly Lost) | % GT tracks tracked for ≤20% of lifetime | MOT |
| **FP** | Total false positives | Already output |
| **FN** | Total false negatives | Already output |
| **IDsw** | Total identity switches | Already output |
| **Frag** | Total fragmentations | MOT |
| **Track mAP** | AP treating tracking as detection + linking | TAO |

---

## 8. Novel View Synthesis

### Currently implemented (rpx_benchmark/evaluators.py)
- PSNR, SSIM (simplified global)

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **PSNR** | Peak signal-to-noise ratio | All NVS papers |
| **SSIM** (structural, windowed) | Wang et al. 2004 — use sliding 11×11 window, NOT global | Standard; current impl is simplified |
| **LPIPS** | Learned perceptual similarity (AlexNet or VGG backbone) | All modern NVS papers |
| **MS-SSIM** | Multi-scale SSIM | Some papers |
| **FID** | Fréchet Inception Distance (needs many samples) | Generative quality |
| **DISTS** | Deep image structure and texture similarity | Some papers |
| **FLIP** | HDR-VDP inspired perceptual metric | NVIDIA |
| **AbsRel depth** | If NVS also predicts depth, evaluate depth accuracy | NeRF-based methods |
| **Masked PSNR/SSIM** | Only on foreground (object) regions | Object-centric NVS |

---

## 9. Sparse Depth

### Currently implemented (rpx_benchmark/metrics/sparse_depth.py)
- Sparse AbsRel, Sparse RMSE

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **AbsRel** (at sparse points) | Already implemented | |
| **RMSE** (at sparse points) | Already implemented | |
| **MAE** (at sparse points) | Mean absolute error at GT locations | Depth completion |
| **iMAE** | MAE on inverse depth | KITTI completion |
| **iRMSE** | RMSE on inverse depth | KITTI completion |
| **δ1 at sparse points** | Threshold accuracy at sparse locations | |

---

## 10. Robot VQA (rpx_benchmark/tasks/robot_vqa.py)

### Currently implemented
- MCQ accuracy, Per-task-type accuracy, Per-phase accuracy, Per-robot-operation accuracy

### Full metric set to implement

| Metric | Description | Used by |
|--------|-------------|---------|
| **Overall accuracy** | Already implemented | |
| **Per task type (T1–T6)** | Already implemented | |
| **Per phase** | Already implemented | |
| **Per robot operation** | Already implemented | |
| **Confidence calibration** | If model outputs logits: ECE (Expected Calibration Error) | VQA reliability |
| **Abstention rate** | % "cannot determine" answers | Safety-critical |
| **STR (Scene Transition Robustness)** | Accuracy change across phases | RPX-specific |
| **Consistency** | Same question across viewpoints → same answer? | 360° walkaround |
| **Reference grounding accuracy** | T1 specifically: separate TP/FP/TN/FN rates | Object registration |
| **Spatial reasoning accuracy** | T3 specifically: per-direction (L/R/above/below) | Spatial grounding |

---

## 11. Cross-Task / Deployment Metrics

These apply to ALL tasks and are computed by the runner/profiler.

### Three-Tier Resource Reporting (implemented in `rpx_benchmark.profiler`)

**Tier 1 — Hardware-agnostic model properties** (identical on any hardware):

| Metric | Description | Status |
|--------|-------------|--------|
| **Parameters (M)** | Trainable parameter count | ✅ Implemented |
| **FLOPs (G)** | Forward-pass floating-point ops at batch=1, FP32 | ✅ Implemented |
| **MACs (G)** | Multiply-accumulate ops (= FLOPs / 2) | ✅ Implemented |
| **Memory Traffic (GB)** | Estimated DRAM read+write bytes | ✅ Implemented |
| **Arithmetic Intensity (FLOP/Byte)** | FLOPs / Traffic — determines compute- vs memory-bound | ✅ Implemented |

**Tier 2 — Roofline latency bounds** (hardware-parametric, anyone can recompute):

| Metric | Description | Status |
|--------|-------------|--------|
| **Compute-bound time (ms)** | FLOPs / GPU peak FLOP/s | ✅ Implemented |
| **Memory-bound time (ms)** | Traffic / GPU peak bandwidth | ✅ Implemented |
| **Roofline latency (ms)** | max(compute, memory) — theoretical lower bound | ✅ Implemented |
| **Bottleneck** | "compute" or "memory" | ✅ Implemented |

Published for three reference GPUs: A100-80GB, RTX 4090, Jetson Orin 64GB. Custom GPUs supported via `GPUSpec`.

**Tier 3 — Measured (hardware-specific, with SystemCard):**

| Metric | Description | Status |
|--------|-------------|--------|
| **Latency (p50, p95, p99, mean)** | Per-sample inference time | ✅ Implemented |
| **Peak memory (CPU, CUDA, MPS)** | Max memory usage | ✅ Implemented |
| **SystemCard** | GPU name, VRAM, CPU, RAM, CUDA version, PyTorch version, OS, precision | ✅ Implemented |
| **Throughput (FPS)** | Frames per second at batch=1 | Derivable from latency |
| **Batch throughput** | FPS at batch=4, 8, 16 | Add to profiler |

### Composite + Deployment Metrics

> **Policy (2026-05-11).** RPX reports each model on **three
> independent axes** — task performance, scene-change robustness,
> compute cost — and never collapses them into a single composite
> score. The paper's central finding is that *rankings disagree across
> axes*. The historical composite scores (DRS, ERS) **have been
> removed** from the toolkit; the per-axis components they used to
> combine remain the canonical reporting surface. See
> [`../../SHARED_CONTEXT.md`](../../SHARED_CONTEXT.md) for the policy
> log.

| Metric | Description | Status |
|--------|-------------|--------|
| **Deployment Readiness Score (DRS)** | *Historical:* `TP × R × E`. TP = task performance [0,1]; R = 1-\|STR\| [0,1]; E = 1/(1 + FLOPs/F_median) [0,1]. Multiplicative composite. | ❌ **Removed** — hid axis-level disagreement. |
| **Embodied Readiness Score (ERS)** | *Historical:* weighted composite of accuracy + robustness + latency + memory + compute. | ❌ **Removed** — same reasoning. |
| **Temporal Stability** | Consistency across consecutive frames | ✅ Implemented (depth, seg) — part of the scene-change-robustness axis. |
| **Geometric Coherence (SGC)** | Depth <-> segmentation consistency | ✅ Implemented (seg) — part of the scene-change-robustness axis. |
| **TensorRT/ONNX compatibility** | Binary: can the model be exported? | Manual flag |
| **Warm-up frames** | How many frames until stable predictions | Temporal models |

---

## Implementation Priority

**Phase 1** (immediate — needed for depth benchmark): Extend `depth.py` with SqRel, RMSElog, SIlog, log10, MAE, iRMSE, iMAE, alignment variants, and per-region stratification.

**Phase 2** (before submission): Full COCO-style detection/segmentation AP, windowed SSIM, LPIPS for NVS, HOTA for tracking, full pose recall tables.

**Phase 3** (nice-to-have): Boundary metrics (F-score, Boundary IoU), planarity error, FID, calibration metrics.

---

## References

1. Eigen et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." NeurIPS 2014. arXiv:1406.2283
2. KITTI Depth Prediction Benchmark: https://cvlibs.net/datasets/kitti/eval_depth.php
3. "How Should One Evaluate Monocular Depth Estimation?" arXiv:2510.19814
4. 4th MDEC Challenge. arXiv:2504.17787
5. BenchDepth. arXiv:2507.15321
6. COCO Evaluation Metrics: https://cocodataset.org/#detection-eval
7. MOT Metrics: https://motchallenge.net/
8. HOTA: Luiten et al. "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking." IJCV 2021.
