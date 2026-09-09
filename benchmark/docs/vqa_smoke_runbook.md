# VQA smoke-test runbook

## Frozen scope

The benchmark now supports both one-image ("normal") and two-image
("in-context") samples. Normal rows are unchanged from the original
single-image contract. In-context rows send an ordered `[reference, target]`
image pair -- Image 1 is a deterministic crop of a *different* object that
shares an attribute with the answer (or, for the spatial task, the exact
anchor object the question refers to); Image 2 is an existing normal-task
target frame. The answer bbox is always in Image 2's own pixel coordinates,
never Image 1's. Task groups:

1. top/down and left/right binary questions (normal only);
2. spatial grounding (`spatial_lr_extreme`, `depth_closest`,
   `spatial_farthest` normal; `inctx_spatial_farthest` in-context) scored by
   bbox;
3. attribute-transfer bbox questions: `attr_single_color`,
   `attr_single_material`, `attr_single_function`, `attr_composition`,
   `attr_odd_one_out` (normal) and their `inctx_*` two-image counterparts.

GroundingDINO, Florence-2-large, and RoboPoint are bbox-only baselines. They
must not receive binary or attribute rows. The remaining roster entries run
every bbox task group, normal and in-context alike. Every vLLM engine is
constructed with `limit_mm_per_prompt={"image": 2}` -- a ceiling, not a
requirement: normal rows still send exactly one image. A model's official
processor/chat template wraps the model-neutral prompts for every image
count it receives; the benchmark must not hand-build image tokens.

### Scored inference protocol

Every model answers bbox tasks in one scored call containing the
original question and every required image. Two-image messages explicitly
interleave `Image 1 -- reference object` and `Image 2 -- target scene` labels
with the corresponding images. No scored path performs answer-then-ground
inference.

### PaliGemma single-image limitation

PaliGemma has no established multi-image interleaving convention in vLLM.
For an in-context question, which must see both images,
`vqa_models.vllm_backend` composites Image 1 and Image 2 side by side into
one visibly labelled image and feeds PaliGemma that composite -- a disclosed, model-specific
accommodation for a real architecture limit, not a hidden extra inference
stage. PaliGemma retains its official `answer en` task prefix but must select
and localize the answer in that one generation. Native `detect <label>` calls
are made only by the explicitly unscored localization diagnostic. Adapter
metadata records the composite accommodation and single-call invariant.

### Known data defect: normalized shard paths

Every row in the published `incontext_mos_attribute_bbox.parquet`'s
`target_image` locator has a float-serialized MOS phase baked into its shard
path (e.g. `scenes/scene001/0.0/rgb.tar`, which does not exist on the Hub --
the real path is `scenes/scene001/0/rgb.tar`). `rpx_benchmark.vqa.contract
.normalize_shard` fixes this at fetch time only; the stored locator in the
manifest/predictions is never mutated. Confirmed harmless for
`incontext_ego_attribute_bbox.parquet`, `incontext_mos_spatial_bbox.parquet`,
and every `reference_image` locator, all of which use a clean integer phase.

### Unresolved model-specific vLLM limitations

- **PaliGemma multi-image stage 1**: the side-by-side composite is a fallback
  for a single-image interface. If vLLM's PaliGemma implementation gains
  native multi-image support, prefer that over the composite.
- **Idefics3 / Phi-3.5-Vision engine kwargs** (`mm_processor_kwargs` sizing)
  were tuned for single-image prompts; whether they need adjustment for a
  two-image prompt's token budget is unverified.
- **reference_crop_sha256 verification** reproduces the crop+PNG-encode
  pipeline `sos_reference.py` used at generation time. Verified byte-for-byte
  identical against Pillow 9.5.0 and 12.3.0 in this repo's dev environment;
  the actual vLLM container's Pillow version has not been checked.

## Stages

| Stage | Data | Purpose | Mechanical pass condition |
|---|---|---|---|
| Unit | synthetic rows | parser, coordinates, IoU, schema | all tests pass |
| Smoke | committed 14-row fixture (normal only) | image load, inference, parse, score | complete IDs, parse rate >= 95%, no crash/OOM |
| Acceptance | deterministic 104-row set: normal general/spatial + in-context general/spatial (seed 20260908) | catch model/task and one-vs-two-image integration errors | thresholds calibrated per model and task |
| Benchmark | `build_vqa_benchmark_plan.py`'s 31,500-row available plan (seed 20260908) | full-scale scored run, sharded + batched | frozen config, no missing/duplicate shard IDs, full report |
| Paper gate | full eligible dataset | publishable result | frozen config, no missing rows, full report |

Smoke is intentionally not an accuracy claim. A weak model can pass plumbing
smoke and still fail acceptance. The benchmark stage's `benchmark_pending_4500.json`
(ego normal-spatial and ego in-context-spatial, neither of which has ground
truth yet) is a set of reserved-slot descriptors, never fabricated inference
rows -- see `build_vqa_benchmark_plan.py`'s module docstring.

## Dataset contract

- One RGB frame (the target, Image 2) per normal question; an ordered
  `[reference, target]` = `[Image 1, Image 2]` pair per in-context question.
  The answer bbox is always in Image 2's own `img_w`/`img_h` pixel space.
- MOS RGB comes from Hub revision `main`; ego RGB comes from
  `ego-preview-v2`. In-context locators (`target_image`/`reference_image`)
  are pinned to their own immutable revision
  (`93e31d378f1f98eca18a7aa01a2279c9f332440c`) and are used verbatim from the
  in-context Parquet, never recomputed the way normal locators are.
- JSON-capable models receive `[xmin,ymin,xmax,ymax]` in normalized 0--1000
  target-image coordinates. The parser rescales once to the original image.
- The model never sees scene ID, phase, evidence, target IDs, or answers.
- Object underscores are displayed as spaces and normalized back only during
  scoring.
- Raw output is always retained. Parsing failures are metrics, not silently
  repaired answers.
- Spatial near-ties are excluded from smoke. Future full GT should preserve
  full-precision evidence and an explicit winner margin.

## Determinism

Use greedy decoding (`do_sample=false`, `num_beams=1`), pin model and dataset
revisions, record package/container digests, and save the exact request and raw
response for every sample. PaliGemma bboxes are decoded from normalized
`<loc>` tokens in `yxyx` token order. Other generative models return strict
JSON in normalized 0--1000 `xyxy` coordinates. Invalid JSON, reversed boxes,
refusals, and missing boxes remain model failures rather than being repaired.
