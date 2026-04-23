# RPX Final Task Design

## What the evidence says

### The perception stack in 2024-2026 robot learning:

```
                        ┌─────────────────────────┐
                        │   VLA Policies           │
                        │   (π₀, OpenVLA, GR00T)   │
                        │   Consume: RGB features   │
                        │   Encoder: SigLIP/DINOv2  │
                        └────────┬────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                     │
   ┌────────▼────────┐  ┌───────▼────────┐   ┌───────▼────────┐
   │  Modular Systems │  │  World Models   │   │  Reward Models  │
   │  (OK-Robot, DP3) │  │  (TesserAct)    │   │  (VIP, ReWiND)  │
   │  Consume:        │  │  Consume: RGB   │   │  Consume: RGB   │
   │  depth, masks,   │  │  Generate: RGB  │   │  + language      │
   │  detections,     │  │  + depth        │   │                  │
   │  point clouds    │  │  + normals      │   │                  │
   └────────┬─────────┘  └───────┬─────────┘   └────────────────┘
            │                    │
   ┌────────▼────────────────────▼──────────┐
   │         Perception Models               │
   │  Depth: DA-V2, ZoeDepth, Depth Pro     │
   │  Detect: GroundingDINO, OWL-ViT        │
   │  Segment: SAM2, Grounded-SAM2          │
   │  Track: SAM2 video, Cutie             │
   │  VLM: GPT-4o, Qwen2-VL, PaliGemma    │
   └─────────────────────────────────────────┘
```

### Key insight from the evidence:

1. **SigLIP is the dominant VLA encoder** — π₀, OpenVLA, GR00T N1 all use it. Testing VLMs built on SigLIP (PaliGemma) directly tests VLA perception quality.

2. **Most VLAs are RGB-only** — they don't consume depth. But the MODULAR systems that VLAs are compared against (OK-Robot, DP3) DO consume depth. Both pathways need evaluation.

3. **Depth matters for geometry-focused manipulation** — AnyGrasp, Contact-GraspNet, DP3 all consume point clouds. TesserAct generates depth as output. Depth quality directly affects grasp success.

4. **Instance segmentation is consumed everywhere** — SAM2 is in OK-Robot, in demonstration pipelines, in annotation tools. But nobody benchmarks it on robot scenes under manipulation.

5. **The next generation needs temporal perception** — world models predict future frames, reward models verify progress, long-horizon planning tracks state changes. ALL of these need perception that works across time.

---

## Final Task Set: 7 Tasks

### The Design Rule:
Every task must satisfy ALL of these:
- ✓ No camera pose needed
- ✓ GT derivable from existing RPX annotations (RGB, D435 depth, instance masks, object descriptions, three-phase structure)
- ✓ Maps to a specific deployed robot system
- ✓ Exploits the three-phase protocol as a differentiator
- ✓ Will still matter in 2028

---

### T1: Monocular Metric Depth Estimation
**What:** Predict dense metric depth from single RGB  
**GT:** D435 depth with validity mask  
**Metrics:** AbsRel, RMSE, δ<1.25, SGC (depth-edge alignment)  
**Phase diagnostic:** STR — does depth quality degrade during interaction?  
**Serves:** DP3, AnyGrasp, Contact-GraspNet, TesserAct pseudo-GT  
**Models:** Depth Anything V2 (S/B/L), Video Depth Anything, Depth Pro, UniDepth V2, Metric3D V2, ZoeDepth  
**Why it lasts:** Every manipulation system needs metric 3D. Even as VLAs go RGB-only, modular systems and world models still need depth.

### T2: Open-Vocabulary Detection & Grounding
**What:** Localize objects from text prompt OR reference image  
**GT:** Bounding boxes derived from instance masks  
**Metrics:** AP₅₀, Acc@0.5  
**Phase diagnostic:** STR — detection accuracy across phases. Text vs visual prompt comparison (same objects in isolated + cluttered scenes)  
**Serves:** OK-Robot target finding, modular grasp pipelines  
**Models (text):** GroundingDINO, OWL-ViT v2, YOLO-World, Florence-2, Qwen2.5-VL  
**Models (visual):** DE-ViT, NIDS-Net  
**Why it lasts:** Open-vocabulary detection is the entry point for any instruction-following system.

### T3: Instance Segmentation
**What:** Segment individual object instances from text/visual prompt or in auto mode  
**GT:** Already have instance masks  
**Metrics:** mIoU, boundary F-score, mask AP  
**Phase diagnostic:** STR — segmentation quality across phases  
**Serves:** Grasp planning (segment point cloud → per-object grasps), demonstration annotation, VLA training data  
**Models:** SAM2 (auto + prompted), Grounded-SAM2, Florence-2, SEEM  
**Why it lasts:** SAM2 is in EVERY robot pipeline now. Nobody benchmarks it on robot-relevant scenes under manipulation clutter. This is the most underserved evaluation gap.

### T4: Multi-Object Tracking
**What:** Maintain consistent object identities across frames  
**GT:** SAM2 video tracklets + human-corrected keyframes, initialized with GT first-frame masks  
**Metrics:** MOTA, IDF1, ID switches per phase  
**Phase diagnostic:** Interaction phase = tracking stress test (hand occlusion → ID switches)  
**Serves:** Pick-and-place (track target through grasp), re-grasping, demonstration data quality  
**Models:** SAM2 video, Cutie, DEVA, XMem2  
**Why it lasts:** Any multi-step manipulation needs tracking. Hand occlusion during grasping is the hardest tracking scenario and nobody benchmarks it.

### T5: VLM Scene Understanding (General + Spatial QA)
**What:** Answer questions about the scene (general: "what objects are on the table?" and spatial: "what is to the left of the bottle?")  
**GT:** Template-instantiated answers from annotations — answers CHANGE across phases  
**Metrics:** Fuzzy-match accuracy  
**Phase diagnostic:** STR on QA accuracy — does the VLM lose understanding during interaction?  
**Serves:** VLA backbone evaluation (π₀ uses PaliGemma, OpenVLA uses Prismatic/SigLIP — test the VLM these are built on). Spatial instruction following.  
**Models:** GPT-4o, Qwen2-VL, Qwen2.5-VL, LLaVA-OneVision, Molmo 2, PaliGemma, InternVL2.5  
**Why it lasts:** As VLAs scale, the VLM backbone becomes the bottleneck. Testing VLM understanding on real robot imagery with phase-varying GT is something no VQA benchmark provides.

### T6: Scene Change Detection ⭐ (THREE-PHASE UNIQUE)
**What:** Given two frames from different phases of the same scene, identify what changed (which objects moved, appeared, disappeared)  
**GT:** Compare instance mask sets across phases — automatically derivable  
**Metrics:** Object-level change detection precision/recall; for VLMs, change description accuracy  
**Phase diagnostic:** The ENTIRE task IS the phase diagnostic — it measures perception of state transitions  
**Serves:**
- Post-manipulation verification ("did I pick up the right object?")
- World model validation ("does the model predict the correct state change?")  
- Long-horizon planning ("what has changed since my last observation?")
- Reward model grounding ("did the task succeed?")
**Models:** VLMs given before/after image pairs; dedicated change detection models; mask-overlap baselines  
**Why it lasts:** This is the ONLY benchmark that can test scene change detection with controlled ground truth. As robots do multi-step tasks, state change verification becomes critical. World models (TesserAct, GWM) predict state changes — this benchmarks whether perception can verify them. No other benchmark has this because no other benchmark has the three-phase protocol.

### T7: Depth Completion on Failure Regions
**What:** Complete/inpaint depth on D435 invalid regions (transparent objects, reflective surfaces, out-of-range)  
**GT:** Two approaches:  
  (a) Artificial masking: randomly mask valid depth regions, evaluate reconstruction  
  (b) Cross-frame validation: different frames see different valid pixels (viewpoint variation reveals occluded depth); consensus from multiple frames provides pseudo-GT for invalid regions  
**Metrics:** RMSE on masked regions, completion consistency across frames  
**Phase diagnostic:** STR — do invalid regions change across phases? (interaction may reveal/hide transparent objects)  
**Serves:** EVERY system that uses D435 depth. Transparent bottles, glasses, metal tools are in every kitchen. D435 returns zero depth. This is the #1 real-world depth failure.  
**Models:** Monocular depth models (run on RGB, merge with valid D435 pixels), depth completion models (NLSPN, CompletionFormer), diffusion-based inpainting (Marigold)  
**Why it lasts:** Transparent and reflective object handling is unsolved and will matter for years. No existing benchmark evaluates this on real robot-relevant scenes with actual sensor failure patterns.

---

## Summary Table

| Task | Pose? | New GT? | Phase-unique? | Contemporary system | Future system |
|------|-------|---------|---------------|--------------------|----|
| T1: Depth | No | No | STR | DP3, AnyGrasp | World models |
| T2: Detection | No | No | STR + text/visual | OK-Robot | Instruction-following |
| T3: Segmentation | No | No | STR | SAM2 pipelines | VLA data |
| T4: Tracking | No | No | Interaction stress | Pick-and-place | Multi-step manip |
| T5: VLM QA | No | No | Phase-varying GT | π₀, OpenVLA VLM backbone | Spatial reasoning |
| T6: Change detection | No | No (cross-phase diff) | **The task IS the phase** | Post-manip verify | World model eval |
| T7: Depth completion | No | Partial (artificial mask) | STR on invalid regions | Every D435 system | Transparent objects |

## What was dropped vs the earlier 8-task proposal

- **D4 (Scene QA) and D5 (Spatial QA) merged into T5** — they use the same models, same GT pipeline, and splitting them into separate "decisions" was artificial. Report general and spatial accuracy as sub-metrics within one task.
- **T8 (Depth completion) renumbered to T7**

## The moat

No other benchmark provides ALL of:
1. Same objects across 7 tasks
2. Three-phase structure enabling T6 (change detection) and STR for all tasks
3. Real robot-relevant scenes (not simulation, not internet images)
4. Paired isolated/cluttered objects for visual prompting
5. Phase-varying QA ground truth
6. Depth sensor failure regions (transparent/reflective) for T7

This combination will be hard to replicate and will remain useful as models improve — the tasks are defined by deployment needs, not by current model capabilities.
