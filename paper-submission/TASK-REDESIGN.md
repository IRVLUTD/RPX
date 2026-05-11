# RPX Task Redesign: What to Benchmark

## Available reliable signals

| Signal | Source | Quality | Notes |
|--------|--------|---------|-------|
| RGB video | D435 | ✓ Good | 640×480, 30fps, all 3 phases |
| Metric depth | D435 | ✓ Good | Structured light, invalid pixels on reflective/transparent |
| Instance masks | SAM2 + human | ✓ Good | Per-frame, with refinement counts |
| Tracklets | SAM2 video + human keyframes | ✓ Good | Across frames within phase |
| Object descriptions | AMT crowdsourced | ✓ Good | Category, color, material, shape |
| Bounding boxes | Derived from masks | ✓ Good | Tight boxes |
| GoPro ego video | GoPro | ✓ Good | Interaction phase only |
| T265 fisheye stereo | T265 | ✓ Available | Not yet used |
| Camera pose | T265 VIO | ✗ Noisy | Degrades during interaction |

## Task Design Principles

1. **No task should depend on noisy camera pose** — every metric must be computable from per-frame or phase-level data
2. **Every task should map to a real robot system decision** — "which model should I plug into X?"
3. **The three-phase protocol should be a differentiator** — tasks should exploit clutter→interaction→clean
4. **GT must be derivable from existing annotations** — no new data collection
5. **Tasks should span the perception→planning pipeline** — not just "see" but "see well enough to act"

---

## Recommended Tasks (8 total — 5 current + 3 new)

### Tier 1: Keep (already designed, GT exists)

**T1: Monocular Metric Depth Estimation**
- Input: single RGB frame
- Output: dense metric depth map
- GT: D435 depth (with validity mask)
- Metrics: AbsRel, RMSE, δ<1.25, **SGC** (depth-edge alignment with mask boundaries)
- STR: accuracy change across phases
- Robot relevance: DP3, AnyGrasp, TesserAct, world models consume depth → point cloud
- Models: Depth Anything V2, Video Depth Anything, Depth Pro, UniDepth V2, Metric3D V2, ZoeDepth

**T2: Open-Vocabulary Detection & Grounding**
- Input: RGB frame + text prompt OR reference image
- Output: bounding boxes
- GT: boxes derived from instance masks
- Metrics: AP50, Acc@0.5
- STR: detection accuracy change across phases
- Special: text vs visual prompt comparison (same objects in isolated + cluttered scenes)
- Robot relevance: OK-Robot, modular grasp pipelines need "find the X"
- Models: GroundingDINO, OWL-ViT v2, YOLO-World, Florence-2, Qwen2.5-VL (text); DE-ViT, NIDS-Net (visual)

**T3: Multi-Object Tracking**
- Input: RGB video + first-frame GT masks
- Output: per-frame instance tracks
- GT: SAM2 video tracklets + human keyframes
- Metrics: MOTA, IDF1, ID switches per phase
- STR: tracking quality change across phases (interaction = hand occlusion stress test)
- Robot relevance: pick-and-place needs "keep tracking the target through my hand entering the scene"
- Models: SAM2 video, Cutie, DEVA, XMem2

**T4: Scene Understanding VLM QA**
- Input: RGB frame + question
- Output: text answer
- GT: template-instantiated answers from annotations (phase-varying)
- Metrics: fuzzy-match accuracy
- STR: QA accuracy change across phases
- Robot relevance: VLAs (π₀, OpenVLA) inherit VLM encoders; if VLM fails on D435 imagery, VLA fails
- Models: GPT-4o, Qwen2-VL, Qwen2.5-VL, LLaVA-OneVision, Molmo 2, PaliGemma, InternVL2.5

**T5: Spatial Reasoning VLM QA**
- Input: RGB frame + spatial question ("what is to the left of X?")
- Output: text answer
- GT: spatial relations from bounding box geometry (phase-varying — objects move!)
- Metrics: fuzzy-match accuracy
- STR: spatial QA accuracy change across phases
- Robot relevance: spatial instructions ("pick up the object next to the bottle") require spatial reasoning
- Models: same as T4

### Tier 2: Add (high value, GT derivable from existing data)

**T6: Instance Segmentation**
- Input: RGB frame + text/visual prompt
- Output: instance masks
- GT: already have instance masks
- Metrics: mIoU, boundary F-score, mask AP
- STR: segmentation quality change across phases
- Robot relevance: grasp planning (Contact-GraspNet, AnyGrasp) segments point cloud by instance; SAM2/Grounded-SAM is the standard preprocessing step in every modular robot system
- Models: SAM2 (auto + prompted), Grounded-SAM2, Florence-2, SEEM, Qwen2.5-VL (segmentation mode)
- Why add: segmentation is arguably MORE important than detection for manipulation — you grasp a segment, not a box

**T7: Scene Change Detection (THREE-PHASE UNIQUE)**
- Input: RGB frame from phase A + RGB frame from phase B (same scene)
- Output: what changed? (which objects moved, appeared, disappeared)
- GT: derivable from per-phase instance masks — compare mask sets across phases
- Metrics: change detection accuracy (precision, recall of changed objects), change description accuracy (for VLMs)
- Why this is special: ONLY possible with three-phase protocol — no other benchmark has this
- Robot relevance: post-manipulation verification ("did I actually pick up the right thing?"), anomaly detection, scene state tracking for long-horizon planning
- Models: VLMs (GPT-4o, Qwen2.5-VL) given before/after pairs; dedicated change detection models; simple mask-overlap baselines
- Futuristic: world models should predict what changes when you act — this tests that

**T8: Depth Completion / Inpainting**
- Input: D435 raw depth (with invalid pixels on reflective/transparent surfaces)
- Output: completed depth map
- GT: while there's no perfect GT for invalid regions, you can:
  (a) artificially mask valid regions and evaluate reconstruction
  (b) evaluate downstream: does completion improve point cloud quality for grasping?
  (c) cross-validate with multi-view consistency (different frames see different valid pixels)
- Metrics: RMSE on artificially masked regions, downstream grasp quality
- Robot relevance: transparent objects (glasses, bottles) and reflective surfaces (metal tools) are the #1 depth failure mode in real kitchens. Every robot system encounters this. Currently no benchmark evaluates this on real robot-relevant scenes.
- Models: depth completion models (NLSPN, CompletionFormer), depth inpainting via diffusion (Marigold), monocular depth models run on RGB then merged with D435 valid pixels
- Why add: this is a genuine unsolved problem in robot perception — transparent/reflective object handling

---

## Tasks Considered but NOT Recommended

| Task | Why not |
|------|---------|
| Camera pose estimation | No reliable GT; different research area (SLAM) |
| Surface normal estimation | Can derive GT from D435 depth, but noisy; low robot system demand vs depth |
| Object pose estimation (6-DoF) | Needs CAD models or manual pose annotation; different benchmark (BOP) |
| Navigation / obstacle avoidance | Out of scope for manipulation-focused benchmark |
| Ego-exo correspondence | Interesting but needs calibration between GoPro and D435; future work |
| Action prediction | Policy-level, not perception-level |
| Stereo depth (T265) | T265 fisheye is very different from standard stereo; niche |

---

## Final Recommended Task Set (8 tasks)

| # | Task | Pose needed? | GT from existing data? | Three-phase unique? | Robot system it serves |
|---|------|-------------|----------------------|--------------------|-----------------------|
| T1 | Metric depth | No | ✓ D435 | STR across phases | DP3, AnyGrasp, TesserAct |
| T2 | Detection/grounding | No | ✓ Masks→boxes | STR + text vs visual | OK-Robot, modular pipelines |
| T3 | Tracking | No | ✓ Tracklets | Interaction stress test | Pick-and-place, re-grasp |
| T4 | Scene QA | No | ✓ Templates | Phase-varying answers | VLA backbone evaluation |
| T5 | Spatial QA | No | ✓ BBox geometry | Phase-varying spatial | Spatial instruction following |
| T6 | Instance segmentation | No | ✓ Already have masks | STR across phases | Grasp planning, point cloud |
| T7 | Scene change detection | No | ✓ Cross-phase mask diff | **Only possible here** | Post-manipulation verify |
| T8 | Depth completion | No | ✓ D435 invalid pixels | Transparent/reflective | Real-world depth robustness |

## Why this set lasts

1. **T1-T5** cover the current perception stack (2024-2026)
2. **T6** (segmentation) is becoming more important as SAM2 becomes standard — but nobody benchmarks it on robot scenes under manipulation
3. **T7** (change detection) is forward-looking — as robots do long-horizon tasks, they need to verify what changed. This task doesn't exist in any benchmark. The three-phase protocol makes it trivially derivable.
4. **T8** (depth completion) targets the single biggest unsolved perception problem in real kitchens — transparent and reflective objects. This will matter for years.
5. **All 8 tasks share the same scenes, same objects, same three-phase protocol** — the composition is the moat. Nobody else can provide cross-task deployment recommendations on shared real-world manipulation scenes.
