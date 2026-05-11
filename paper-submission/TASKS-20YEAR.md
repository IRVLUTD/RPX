# RPX Tasks: 1–20 Year Horizon

## Given data: RGB + metric depth (D435) + instance masks + object descriptions + three-phase structure + GoPro ego (unsync'd)

---

## The insight that determines everything

The evidence shows two co-existing paradigms for the next 20 years:

1. **End-to-end VLAs** (π₀, GR00T) — consume RGB features, no explicit perception outputs
2. **Modular systems** (OK-Robot, maestro, DP3) — consume depth, detections, segments, point clouds

RPX must serve BOTH. The three-phase protocol is the differentiator — it tests perception under the manipulation cycle that both paradigms encounter.

---

## Tasks organized by time horizon

### NOW (1–3 years): What deployed systems need today

These tasks evaluate the perception models that are plugged into real robot systems right now.

**T1: Monocular Metric Depth**
- Serves: DP3, AnyGrasp, Contact-GraspNet, TesserAct (pseudo-GT), world models
- GT: D435 metric depth
- Phase value: STR shows if depth degrades when hand enters scene → bad grasps
- Stays relevant: geometry-based manipulation isn't going away

**T2: Open-Vocabulary Detection & Grounding**  
- Serves: OK-Robot target finding, "pick the X" instruction following
- GT: boxes from masks
- Phase value: STR + text vs visual prompt comparison
- Also test: Florence-2, Qwen2.5-VL as unified detect+ground models

**T3: Instance Segmentation**
- Serves: SAM2 is in every pipeline. Grasp planners segment point clouds by instance.
- GT: already have masks
- Phase value: STR — does segmentation quality drop when scene is cluttered vs clean?
- Critical gap: SAM2 is universally used but never benchmarked on robot scenes under manipulation

**T4: Multi-Object Tracking**
- Serves: pick-and-place continuity, re-grasping, demonstration data quality
- GT: SAM2 tracklets + human keyframes
- Phase value: interaction phase = hand occlusion → ID switches (the real failure mode)

**T5: VLM Scene & Spatial Understanding**
- Serves: VLA backbone evaluation. π₀ uses PaliGemma (SigLIP). OpenVLA uses SigLIP+DINOv2. Testing the VLM tests the VLA's perception.
- GT: template-instantiated QA with phase-varying answers
- Phase value: spatial QA answers CHANGE when objects move — tests if VLM tracks state
- Sub-tasks: general QA + spatial QA (same models, same GT pipeline)

### NEAR-TERM (3–7 years): What's emerging and will become standard

**T6: Scene Change Detection** ⭐
- What: given frames from two phases, what changed?
- GT: cross-phase mask diff (automatic)
- Serves:
  - Post-manipulation verification: "did I pick up the right thing?" (Gemini Robotics-ER does this)
  - World model evaluation: "does the model predict the correct state change?" (TesserAct generates future frames)
  - Long-horizon planning: "what's different since my last observation?"
  - Reward model grounding: "did the task succeed?" (VIP, ReWiND, Robometer)
  - Failure detection: "something unexpected changed" (RoboFAC, I-FailSense)
- Why 3–7 years: world models and reward models are exploding now. ALL of them need perception of change. RPX is the ONLY benchmark that can test this with controlled GT.
- Why it lasts: manipulation always changes scenes. This task is defined by physics, not by model architecture.

**T7: Object State Recognition**
- What: identify functional states of objects — open/closed, full/empty, on/off, clean/dirty
- GT: derivable from RPX annotations. Three-phase protocol CHANGES object states (clutter: objects in use-state; clean: objects stored). Per-object attributes from AMT descriptions include state-relevant properties.
- Serves: home robots ("is the dishwasher open?"), task completion detection, safety ("is the stove on?")
- Evidence: STATUS Bench (Oct 2025) revealed VLMs are bad at this. No manipulation-specific benchmark exists.
- Phase value: states CHANGE across phases — clutter has objects in various states; clean phase has objects reorganized
- Why 3–7 years: π₀.5/π₀.7 need this for kitchen cleaning. Gemini Robotics-ER benchmarks success detection which requires state understanding.

**T8: Depth Completion / Transparent-Reflective Object Handling**
- What: complete depth for D435 failure regions (transparent glass, reflective metal, out-of-range)
- GT: artificial masking of valid regions + cross-frame consensus for invalid regions
- Serves: every system that uses RealSense depth. #1 real-world depth failure mode.
- Evidence: no benchmark evaluates this on real robot scenes
- Phase value: invalid regions change across phases (interaction may reveal/hide transparent objects)
- Why 3–7 years: transparent object manipulation is unsolved. As home robots scale, every kitchen has glasses.

### LONG-TERM (7–20 years): What will define the mature era

**T9: Physical Property Inference from Vision**
- What: estimate weight, material, fragility from RGB + depth
- GT: can annotate per-object physical properties (RPX objects are real household items with known properties). Weight is measurable. Material/fragility from AMT annotation extension.
- Serves: safe manipulation ("don't squeeze the egg"), grasp force selection, tool use
- Evidence: PUGS (ICRA 2025), GaussianProperty (ICCV 2025), PhysQuantAgent (2026) — all emerging
- Phase value: same objects across phases allow testing if property estimates are consistent
- Why 7–20 years: this is where vision meets physics. Amazon's Vulcan already needs this at scale. Safe home robots will need it universally.
- Caveat: needs additional annotation (object weights at minimum). But the objects exist in RPX — just weigh them.

**T10: Manipulation-Aware Failure Detection**
- What: given a sequence of perception outputs across phases, detect when perception has failed and what went wrong
- GT: derivable from accuracy drops across phases. A "failure" is a scene-phase pair where accuracy drops below a threshold vs the model's own clean-phase performance.
- Serves: robust deployment. RoboFAC shows perception failures cause cascading errors in long-horizon tasks. I-FailSense shows this is unsolved.
- Phase value: the interaction phase is where failures happen. Clean phase recovery (or lack thereof) reveals if the failure was transient or persistent.
- Why 7–20 years: as robots become more autonomous, self-monitoring becomes essential. A robot that knows it can't see well should ask for help, not fail silently.

---

## What's OUT (and why)

| Task | Why not for RPX |
|------|----------------|
| Camera pose estimation | No reliable GT; different field (SLAM) |
| Navigation perception | Out of scope (manipulation-focused) |
| Deformable object perception | Needs specialized data (RPX has rigid household objects) |
| Liquid/granular perception | Needs specialized sensors (capacitive, tactile) |
| Surface normal estimation | Can derive from D435 depth but noisy; lower priority than depth itself |
| Action/trajectory prediction | Policy-level, not perception-level |
| Tactile perception | No tactile data |
| Ego-exo correspondence | No temporal sync between GoPro and D435 |

---

## The 20-year architecture

```
Year 1-3 (NOW)                    Year 3-7                      Year 7-20
─────────────────                 ─────────────────              ─────────────────
T1: Depth                         T6: Change detection ⭐        T9: Physical properties
T2: Detection/grounding           T7: Object state              T10: Failure detection
T3: Segmentation                  T8: Depth completion
T4: Tracking
T5: VLM QA

ALL share: same scenes, same objects, three-phase protocol, STR metric
```

## For the NeurIPS 2026 paper specifically

Don't try to do all 10 in the paper. Do T1–T6 well. Mention T7–T10 as "the benchmark supports these and they are defined in the toolkit."

- **T1–T5**: solid, well-understood, models ready to evaluate
- **T6**: the differentiator that makes reviewers say "only RPX can do this"
- **T7**: important but may need more GT validation work
- **T8–T10**: future extensions with the data already collected

The paper's narrative: "we evaluate 6 tasks now, the toolkit and data support 10, and the three-phase protocol ensures new tasks can be added without new data collection."
