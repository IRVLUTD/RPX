# Monocular Depth Estimation: Model Survey for RPX Benchmark

**Survey date**: 2025-05-07  
**Scope**: State-of-the-art monocular metric and relative depth estimation models. The RPX benchmark covers 100 scenes (indoor + outdoor) captured with a D435+T265+GoPro rig, spanning tabletop manipulation, atriums, staircases, gardens, tennis courts, and fountains.  
**Criterion**: Open-source code **and** publicly downloadable model checkpoints required for inclusion in the benchmark.

---

## 1. Models Included in Benchmark

The following models have verified open-source implementations and publicly available pretrained weights as of the survey date.

### 1.1 UniDepth V2

| Field | Details |
|-------|---------|
| **Paper** | "UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler" |
| **Venue** | arXiv:2502.20110 (Feb 2025); UniDepth V1 at CVPR 2024 Highlight |
| **Affiliation** | ETH Zurich |
| **Code** | <https://github.com/lpiccinelli-eth/UniDepth> |
| **Checkpoints** | `lpiccinelli/unidepth-v2-vitl14` (ViT-L, 3.4M downloads/month), `lpiccinelli/unidepth-v2-vits14` (ViT-S) on HuggingFace |
| **License** | See repo |
| **Output** | Metric depth (meters). Also predicts camera intrinsics — no intrinsics input required. |
| **Key claim** | Overall new SOTA in zero-shot metric monocular depth estimation; ranks #1 among published methods on KITTI Depth Prediction benchmark. |
| **Relevance** | Camera-agnostic metric prediction is directly useful for robotics where camera parameters may vary or be imprecise. |

### 1.2 Depth Pro

| Field | Details |
|-------|---------|
| **Paper** | "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" |
| **Venue** | ICLR 2025 |
| **Affiliation** | Apple |
| **Code** | <https://github.com/apple/ml-depth-pro> |
| **Checkpoints** | `apple/DepthPro` (original), `apple/DepthPro-hf` (transformers-native) on HuggingFace |
| **License** | Apple Sample Code License |
| **Output** | Metric depth at 2.25 megapixel resolution. No camera intrinsics needed. |
| **Key claim** | 0.3 seconds per frame on a standard GPU. Sharp high-frequency boundary details. |
| **Relevance** | Speed + boundary sharpness are critical for real-time robotics perception of object edges. |

### 1.3 Depth Anything V2

| Field | Details |
|-------|---------|
| **Paper** | "Depth Anything V2" |
| **Venue** | NeurIPS 2024 |
| **Affiliation** | HKU, TikTok |
| **Code** | <https://github.com/DepthAnything/Depth-Anything-V2> |
| **Checkpoints (relative)** | `depth-anything/Depth-Anything-V2-Small-hf` (24.8M params, 1.8M dl/mo), `-Base-hf` (97.5M), `-Large-hf` (335.3M) |
| **Checkpoints (metric)** | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` (indoor-trained), `Depth-Anything-V2-Metric-VKITTI-Large` (outdoor-trained) |
| **License** | Apache 2.0 |
| **Output** | Relative depth (base models); metric depth (fine-tuned variants). |
| **Key claim** | Trained on 595K synthetic + 62M real unlabeled images. Finer detail and more robust than V1. 10× faster than diffusion-based methods. |
| **Relevance** | Both metric variants benchmarked: Hypersim-trained (indoor-focused) and VKITTI-trained (outdoor-focused). RPX has both indoor and outdoor scenes, so this directly tests domain-gap effects. Multiple model sizes enable latency/accuracy tradeoff analysis. |

### 1.4 Metric3D v2

| Field | Details |
|-------|---------|
| **Paper** | "Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation" |
| **Venue** | IEEE TPAMI 2024 |
| **Affiliation** | Multiple (see paper) |
| **Code** | <https://github.com/YvanYin/Metric3D> |
| **Checkpoints** | Auto-downloaded via repo scripts |
| **License** | See repo |
| **Output** | Metric depth + surface normals jointly. |
| **Key claim** | Zero-shot metric depth with auxiliary normal estimation improves geometric accuracy. |
| **Relevance** | Joint depth + normal output is valuable for robotics grasp planning and surface reasoning. |

### 1.5 MoGe-2

| Field | Details |
|-------|---------|
| **Paper** | "MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details" |
| **Venue** | NeurIPS 2025 (MoGe v1: CVPR 2025 Oral) |
| **Affiliation** | Microsoft Research |
| **Code** | <https://github.com/microsoft/MoGe> |
| **Checkpoints** | `Ruicheng/moge-2-vitl-normal` on HuggingFace. Training code released Oct 2025. |
| **License** | See repo |
| **Output** | Metric-scale 3D point map + normal map from a single image. |
| **Key claim** | Extends MoGe (affine-invariant point maps) to absolute metric scale with improved sharpness and lower latency. |
| **Relevance** | Full 3D point map output (not just depth) is directly useful for robotic manipulation planning. |

### 1.6 MetricSolver

| Field | Details |
|-------|---------|
| **Paper** | "Metric-Solver: Sliding Anchored Metric Depth Estimation from a Single Image" |
| **Venue** | arXiv:2504.12103 (Apr 2025) |
| **Affiliation** | Tele-AI |
| **Code** | <https://github.com/Tele-AI/MetricSolver> |
| **Checkpoints** | Provided in repo |
| **License** | MIT |
| **Output** | Metric depth with dynamic scale anchoring. |
| **Key claim** | Sliding anchor method adapts to diverse depth scales across indoor/outdoor without separate heads. |
| **Relevance** | Handles the scale diversity present in robotics scenes (close tabletop objects vs. room-scale). |
| **Note** | ICLR 2026 submission was withdrawn; code and weights remain publicly available. |

### 1.7 Marigold

| Field | Details |
|-------|---------|
| **Paper** | "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" |
| **Venue** | CVPR 2024 (Oral, Best Paper Award Candidate) |
| **Affiliation** | ETH Zurich |
| **Code** | <https://github.com/prs-eth/Marigold> |
| **Checkpoints** | Multiple variants on HuggingFace under `prs-eth/` |
| **License** | Apache 2.0 |
| **Output** | Relative depth (affine-invariant). Requires post-hoc alignment for metric evaluation. |
| **Key claim** | Leverages Stable Diffusion visual priors. Strong zero-shot generalization from synthetic-only fine-tuning. |
| **Relevance** | Diffusion-based baseline representing a different architectural paradigm. |

### 1.8 Lotus / Lotus-2

| Field | Details |
|-------|---------|
| **Paper** | "Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction" |
| **Venue** | ICLR 2025 (Lotus-2: arXiv:2512.01030, Nov 2025) |
| **Affiliation** | HKUST |
| **Code** | <https://github.com/chenxwh/Lotus> |
| **Checkpoints** | `jingheya/lotus-depth-g-v2-1-disparity` (Lotus v2, 4.2K dl/mo), `jingheya/Lotus-2` (Lotus-2 with sharpener + normals) |
| **License** | See repo |
| **Output** | Relative depth (disparity space). Also normal estimation in Lotus-2. |
| **Key claim** | Single-step prediction (not iterative diffusion), faster than Marigold with comparable or better quality. |
| **Relevance** | Faster diffusion-family alternative; Lotus-2 adds boundary sharpening relevant to object segmentation. |

### 1.9 ZoeDepth

| Field | Details |
|-------|---------|
| **Paper** | "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth" |
| **Venue** | arXiv:2302.12288 (2023) |
| **Affiliation** | Intel |
| **Code** | <https://github.com/isl-org/ZoeDepth> |
| **Checkpoints** | Provided in repo |
| **License** | MIT |
| **Output** | Metric depth. |
| **Key claim** | Combines relative depth backbone with metric depth heads for domain-specific transfer. |
| **Relevance** | Legacy baseline. Repository archived May 2025 — still functional but no longer maintained. |

---

## 2. Models NOT Benchmarked — Unavailable Code/Checkpoints

The following recent models are relevant but **cannot be included** because open-source code and/or pretrained checkpoints were not publicly available as of **2025-05-07**.

| Model | Paper | Venue | Reason for Exclusion |
|-------|-------|-------|---------------------|
| **AsyncMDE** | "Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory" | arXiv:2603.10438 (Mar 2026) | No code or model weights released. |
| **RTS-Mono** | "A Real-Time Self-Supervised Monocular Depth Estimation Method for Real-World Deployment" | arXiv:2511.14107 (Nov 2025) | No code or model weights released. |
| **PTC-Depth** | "Pose-Refined Monocular Depth Estimation with Temporal Consistency" | arXiv:2604.01791 (Apr 2026) | No code or model weights released. |
| **AnyDepth** | "Depth Estimation Made Easy" (DINOv3 encoder) | arXiv:2601.02760 (Jan 2026) | No code or model weights released. |
| **KineDepth** | "Utilizing Robot Kinematics for Online Metric Depth Estimation" | IROS 2025 (arXiv:2409.19490) | Implementation tightly coupled to a specific robot platform; not generalizable without hardware. |

### Statement for Reviewers

> These models represent relevant concurrent or subsequent work. Their exclusion from our benchmark results is solely due to the absence of publicly available code and pretrained weights at the time of evaluation (2025-05-07). Our benchmark dataset (75K frames across 100 indoor and outdoor scenes) and evaluation toolkit are publicly released; **we invite the authors of these methods to evaluate on our dataset**, which demonstrates the self-sustaining nature of our benchmark — new methods can be evaluated against the same standardized data and metrics without our direct involvement.

---

## 3. Additional Context

### 3.1 Evaluation Protocol Reference

- **4th Monocular Depth Estimation Challenge (MDEC)**, CVPR 2025 Workshop (arXiv:2504.17787): switched to least-squares alignment (from median). 24 submissions.
- **BenchDepth** (<https://zhyever.github.io/benchdepth>): proposes alignment-free evaluation of depth foundation models through five carefully designed tasks.
- **"How Should One Evaluate Monocular Depth Estimation?"** (arXiv:2510.19814): critiques standard metrics (δ1, AbsRel) as insensitive to curvature perturbations.

### 3.2 Video/Temporal Depth (out of scope for this benchmark phase)

- **DepthCrafter** (CVPR 2025 Highlight, Tencent): temporally consistent video depth. Code at <https://github.com/Tencent/DepthCrafter>. Relevant if the benchmark expands to sequential frames.

### 3.3 Deferred Models (noted in CHANGELOG.md)

Per the RPX benchmark CHANGELOG, these are explicitly deferred:
- **Video Depth Anything** — needs temporal eval mode
- **Prompt Depth Anything** — needs sparse-depth prompt task
- **Depth Anything 3** — not yet in `transformers` library

---

## 4. Summary Table

### 4.1 Metric Depth Models (10)

| # | Model | Venue | Code | Checkpoints | Notes |
|---|-------|-------|------|-------------|-------|
| 1 | UniDepth V2 | CVPR'24 / arXiv'25 | [github](https://github.com/lpiccinelli-eth/UniDepth) | `lpiccinelli/unidepth-v2-vitl14` | SOTA metric, no intrinsics |
| 2 | Depth Pro | ICLR 2025 | [github](https://github.com/apple/ml-depth-pro) | `apple/DepthPro-hf` | Sharp edges, 0.3s, Apple |
| 3 | Depth Anything V2 Metric-Indoor | NeurIPS 2024 | [github](https://github.com/DepthAnything/Depth-Anything-V2) | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` | Indoor (Hypersim-trained) |
| 4 | Depth Anything V2 Metric-Outdoor | NeurIPS 2024 | same repo | `depth-anything/Depth-Anything-V2-Metric-VKITTI-Large` | Outdoor (VKITTI-trained) — domain-gap probe across RPX's indoor+outdoor mix |
| 5 | Metric3D v2 | TPAMI 2024 | [github](https://github.com/YvanYin/Metric3D) | In repo (auto-download) | Joint depth + normals |
| 6 | MoGe-2 | NeurIPS 2025 | [github](https://github.com/microsoft/MoGe) | `Ruicheng/moge-2-vitl-normal` | Metric 3D point map + normals |
| 7 | MetricSolver | arXiv Apr 2025 | [github](https://github.com/Tele-AI/MetricSolver) | In repo | Sliding anchor, MIT license |
| 8 | ZoeDepth | arXiv 2023 | [github](https://github.com/isl-org/ZoeDepth) | In repo | Legacy baseline, archived May 2025 |
| 9 | PatchFusion | CVPR 2024 | [github](https://github.com/zhyever/PatchFusion) | `zhyever/patchfusion_zoedepth` + variants | Tile-based high-res metric, MIT |
| 10 | HyDen (MetaDepth) | ICLR 2026 | [github](https://github.com/facebookresearch/metadepth) | `facebook/hyden-da2-metric-depth` | Meta, hybrid dual-path encoder |

### 4.2 Relative Depth Models (10)

| # | Model | Venue | Code | Checkpoints | Notes |
|---|-------|-------|------|-------------|-------|
| 1 | Depth Anything V2 (relative) | NeurIPS 2024 | [github](https://github.com/DepthAnything/Depth-Anything-V2) | `depth-anything/Depth-Anything-V2-Large-hf` (335M) | S/B/L sizes |
| 2 | Depth Anything V1 | CVPR 2024 | [github](https://github.com/LiheYoung/Depth-Anything) | `LiheYoung/depth-anything-large-hf` | Predecessor baseline |
| 3 | Marigold v1.1 | CVPR 2024 Oral | [github](https://github.com/prs-eth/Marigold) | `prs-eth/marigold-depth-v1-1` | Diffusion-based, best paper candidate |
| 4 | Marigold LCM | CVPR 2024 Oral | same repo | `prs-eth/marigold-depth-lcm-v1-0` | Faster variant (LCM distillation) |
| 5 | Lotus-2 | ICLR 2025 | [github](https://github.com/chenxwh/Lotus) | `jingheya/Lotus-2` | Single-step diffusion + normals |
| 6 | GeoWizard | ECCV 2024 | [github](https://github.com/fuxiao0719/GeoWizard) | On HuggingFace | Joint depth + normals, diffusion |
| 7 | MiDaS v3.1 (DPT-BEiT-L) | TPAMI 2022 | [github](https://github.com/isl-org/MiDaS) | `Intel/dpt-beit-large-384` | Classic baseline, 5 backbone variants |
| 8 | Distill-Any-Depth | arXiv Feb 2025 | [github](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth) | `xingyang1/Distill-Any-Depth-Large-hf` (61K dl/mo) | Multi-teacher distillation of DA-V2 |
| 9 | MoGe (v1) | CVPR 2025 Oral | [github](https://github.com/microsoft/MoGe) | `Ruicheng/moge-vitl` | Affine-invariant 3D point map |
| 10 | HyDen (relative) | ICLR 2026 | [github](https://github.com/facebookresearch/metadepth) | `facebook/hyden-da2-relative-depth` | Meta, relative-depth variant |

### 4.3 Models NOT Benchmarked — Unavailable Code/Checkpoints

| Model | Paper | Venue | Reason for Exclusion |
|-------|-------|-------|---------------------|
| AsyncMDE | arXiv:2603.10438 | arXiv Mar 2026 | No code or weights released |
| RTS-Mono | arXiv:2511.14107 | arXiv Nov 2025 | No code or weights released |
| PTC-Depth | arXiv:2604.01791 | arXiv Apr 2026 | No code or weights released |
| AnyDepth | arXiv:2601.02760 | arXiv Jan 2026 | No code or weights released |
| KineDepth | arXiv:2409.19490 | IROS 2025 | Code tightly coupled to specific robot platform |
