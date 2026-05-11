# `mask_pipeline/` — Ground-truth mask generation

> Interactive **GroundingDINO + SAM2** pipeline that turns a captured RPX
> scene into per-frame instance masks. A human operator curates bboxes
> on one keyframe per phase; **SAM2 propagates** to every other frame.

> [!IMPORTANT]
> **Most users do not need this directory.** Mask generation is how
> the RPX ground truth was *created*. To *benchmark* a model against
> the already-published RPX dataset (which already has masks),
> use [`../benchmark/`](../benchmark/README.md) instead. You only
> come here if you're producing GT masks for newly captured scenes,
> on a CUDA-capable annotation box.

**What it does.** Per-frame instance masks are required ground truth
for the benchmark, and labelling them by hand is impractical at RPX
scale. The pipeline uses GroundingDINO to suggest open-vocabulary
bounding boxes on one keyframe per phase, SAM2 to turn each curated
box into a pixel-accurate mask, and SAM2's temporal propagation to
extend that mask through every remaining frame in the phase. A human
reviews one keyframe per phase per scene; the rest is automatic.

---

## ⚡ Quick start

```bash
# 0. One-time setup (conda env with CUDA + the RoboKit library)
export CUDA_HOME=/usr/local/cuda-12.6/
conda create -n rkit-rpx python=3.9 && conda activate rkit-rpx
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

# 1. Pull one scene's raw captures from Box
#    → https://utdallas.box.com/s/saifhadoad3w136tbvfgcrd4n2zk8e7t

# 2. ITERATION 1 — first-pass mask generation (per phase)
#    Opens a GroundingDINO-pre-populated bbox UI on the last frame;
#    curate the boxes, press q, SAM2 propagates backward.
python -m maskgen_pipeline.interactive_gsam2 --scene_dir <scene_dir>/0
python -m maskgen_pipeline.interactive_gsam2 --scene_dir <scene_dir>/1
python -m maskgen_pipeline.interactive_gsam2 --scene_dir <scene_dir>/2

# 3. ITERATION 2 — automatic refinement of faulty frames
python -m maskgen_pipeline.refine_masks_flow --scene_dir <scene_dir>/1 --iter 1

# 4. Map mask IDs ↔ object names (one click per object id)
python -m visual_grounding_gt.mask_to_object \
        --scene_dir <scene_dir> --json visual_grounding_gt/scenes_test.json

# 5. Zip + upload to the iter-2 results folder on Box.
```

Output layout (consumed directly by `rpx_benchmark`):

```
<scene_dir>/<phase>/sam2/masks/<frame>.png      # int16 per-instance ID map
<scene_dir>/<phase>/sam2/labels/...             # GroundingDINO label artefacts
```

---

## 🧱 The two-stage pipeline

### Iteration 1 — interactive bbox curation + SAM2 propagation

```bash
python -m maskgen_pipeline.interactive_gsam2 --scene_dir <scene_dir>/<phase>
```

1. A window opens showing the **last frame** with GroundingDINO-suggested
   bounding boxes.
2. Keep / delete / right-click-drag-new the boxes around objects of
   interest. Press **`q`** to confirm.
3. A second window opens for bbox quality refinement
   (left-click-drag to resize). Press **`q`** to finalise.
4. SAM2 propagates the boxes backward through every frame in the phase,
   writing instance masks to `sam2/masks/`.

To visualize results + flag faulty frames after iter 1:

```bash
python -m maskgen_pipeline.review_faulty_masks --scene_dir <scene_dir>/<phase>
```

### Iteration 2 — automatic refinement of faulty frames

Faulty frames flagged in iter 1 get a second pass. **No human input** —
fully automatic.

```bash
python -m maskgen_pipeline.refine_masks_flow --scene_dir <scene_dir>/<phase> --iter 1
```

`--iter` records which round this is (start at 1; repeat with `--iter 2`,
`3`, ... if more passes are needed). Then mark the **correctly** masked
frames (faster — most are already good):

```bash
python -m maskgen_pipeline.review_faulty_masks --scene_dir <scene_dir>/<phase> --iter 2
```

### Final step — object-ID mapping

Once masks look right across all three phases, link mask IDs to object
names (one click per id, per scene):

```bash
python -m visual_grounding_gt.mask_to_object \
        --scene_dir <scene_dir> --json visual_grounding_gt/scenes_test.json
```

---

## 📂 What each script does

| Script | Purpose |
|---|---|
| `maskgen_pipeline/interactive_gsam2.py` | **Iter 1.** Interactive bbox curation + SAM2 propagation. |
| `maskgen_pipeline/interactive_gsam2_rdd_refinement.py` | Variant using RDD (Robust Deformable Detector) for tighter bboxes on small objects. |
| `maskgen_pipeline/interactive_gsam2_refine_pp_1.py` | Refinement variant: bbox-then-point-prompt on faulty frames. |
| `maskgen_pipeline/refine_masks.py` / `refine_masks_flow.py` | **Iter 2.** Automatic refinement of faulty frames (no UI). |
| `maskgen_pipeline/review_faulty_masks.py` | Visualize generated masks; tag faulty frames for the next iter. |
| `maskgen_pipeline/review_faulty_bboxes.py` | Same idea, but on raw bboxes (pre-SAM2). |
| `maskgen_pipeline/propagate_bboxes.py` | Standalone bbox-propagation step (used internally by the interactive tools). |
| `maskgen_pipeline/collect_faulty_and_gdino_bboxed_headless.py` | Headless variant: enumerate faulty frames + pre-compute GroundingDINO bboxes. |
| `maskgen_pipeline/run_sam2_from_verified_bboxes.py` | Re-run SAM2 from a verified bbox set (skip the interactive curation). |
| `maskgen_pipeline/point_prompts_via_rdd.py` | Generate point prompts using RDD correspondences. |
| `maskgen_pipeline/pds_frames_filter.py` | Poisson-disk frame filter to thin a long capture to a representative subset. |
| `maskgen_pipeline/test_sam2_point_prompts.py` | Standalone SAM2 point-prompt smoke test. |
| `maskgen_pipeline/bbox_prompts_overlay.py`, `collect_bbox_prompts.py` | Bbox utilities re-used by the interactive UIs. |
| `maskgen_pipeline/mask_refinement_tools/mask_to_bbox_refinement.py` | Optional follow-up: refine bboxes back from a verified mask. |
| `visual_grounding_gt/mask_to_object.py` | One-click mask-id ↔ object-name mapping (final step). |
| `visual_grounding_gt/correspondence_validation_viz.py` | Visual sanity check on the id mapping. |

The inner `robokit/` directory is the [IRVLUTD/robokit](https://github.com/IRVLUTD/robokit)
Python library — perception primitives, dataset adapters, evaluation
metrics. It's imported as `from robokit.perception import …` from
every pipeline script.

---

## 🐳 Docker

Ubuntu 20.04 + ROS Noetic + CUDA 11.8 + Gazebo 11. Useful when the
host environment is dirty.

```bash
cd docker
./run_container.sh
```

The image is built off
[`Dockerfile-ub20.04-ros-noetic-cuda11.8-gazebo`](docker/Dockerfile-ub20.04-ros-noetic-cuda11.8-gazebo).

---

## 🔗 Acknowledgments

The interactive pipeline is built on top of several upstream projects.
**License compatibility is mandatory — check each before redistributing.**

- [SAM2](https://github.com/facebookresearch/sam2) — Meta's Segment
  Anything v2; bbox-and-point → mask + temporal propagation.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) —
  IDEA-Research's open-vocabulary bbox suggester.
- [RoboKit](https://github.com/IRVLUTD/robokit) — IRVL UTD's perception
  toolkit; vendored as `mask_pipeline/robokit/`.
- [BundleSDF](https://github.com/NVlabs/BundleSDF) — NVIDIA's
  reference docker setup, used as a base for ours.
- [iTeach-DHYOLO](https://huggingface.co/spaces/IRVLUTD/DH-YOLO),
  [DepthAnything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything),
  [FeatUp](https://github.com/mhamilton723/FeatUp),
  [CLIP](https://github.com/openai/CLIP) — additional zero-shot
  perception primitives available in the inner `robokit/` library.

## License

MIT, matching the rest of the RPX repo. Upstream-licence compatibility
is the user's responsibility.
