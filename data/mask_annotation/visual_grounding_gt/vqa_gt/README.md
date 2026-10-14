# VQA ground truth

Generates question-answer ground truth from RPX's masks, depth and object questionnaires. There are three
single-image tasks:

1. Attribute VQA (`gt_attributes.py`): "which object is X" questions built from each object's questionnaire attributes.
2. Spatial yes/no (`gt_spatial.py`): left-right and up-down questions about two named objects.
3. Spatial bbox (`gt_spatial.py`): "which object is X" for spatial properties (closest, farthest, leftmost,
   rightmost), answered with a bounding box.

Each also has a two-image, in-context variant, described [below](#in-context-two-image-vqa).

## Usage

```bash
python generate_gt.py \
    --staged-root /path/to/staged/scenes \
    --sos-root /path/to/single_objects/sos_wrapped \
    --tier hard \
    --frames-per-phase 30 \
    --out gt_hard.jsonl
```

`--staged-root` must already hold the scenes in the standard layout (`<scene>/<0|1|2>/sam2/...`,
`<scene>/<0|1|2>/depth/...`, `<scene>/ego/sam2/...`); the script does not download them. `--tier` takes the scene list
for that difficulty tier from `splits/scene_splits.json` on Hugging Face. To pick scenes yourself, pass
`--scenes scene001 scene002 ...` instead. Named scenes missing from `--staged-root` are skipped with a warning.

`--sos-root` (or `RPX_SOS_ROOT`) must point to a local copy of the single-object catalog
(`single_objects/sos_wrapped/<object>/questionnaire.txt`); there is no default. Everything else is read from
`--staged-root` or Hugging Face. The in-context tasks always read the official object catalog and its reference
images from Hugging Face (`sos_catalog.py`, `sos_reference.py`).

The output has one question per line.

### Caps per frame

`--max-per-type` (default 5) caps `attr_single_color`, `attr_single_material`, `attr_single_function`,
`attr_composition`, `spatial_lr_binary`, `spatial_ud_binary` and `spatial_farthest` per frame. `0` removes the cap.
The other types have fixed counts:

| Type | Per frame |
|---|---|
| `attr_synonym` | 1 yes and 1 no, so the answers stay balanced |
| `attr_count`, `attr_odd_one_out` | uncapped; the pools are already small |
| `spatial_lr_extreme` | exactly 2 (leftmost and rightmost) |
| `depth_closest` | exactly 1 |

The cap works differently for the two families, on purpose:

- **Attribute questions are deduplicated across the phase.** The objects don't move within a phase, so "the doll is
  red" is true in every frame, and sampling each frame independently mostly repeats the same facts (up to 96.7% of
  `attr_count` questions were repeats before this change). `_gen_attributes_for_phase()` in `generate_gt.py` gathers
  every candidate fact from the phase's frames first, then `distribute()` in `dedup.py` places each distinct fact
  once, in the valid frame with the lightest load so far, before anything repeats. Here `--max-per-type` limits
  distinct facts per frame.
- **Spatial questions are sampled per frame.** "Is A left of B" depends on the viewpoint, not the arrangement: 90%
  of object pairs flip their `spatial_lr_binary` answer at least once within a phase. Asking it again in another
  frame tests viewpoint robustness, so `--max-per-type` is a plain per-frame cap (`_cap()` in `gt_spatial.py`).

## Attribute VQA

Types: `attr_single_color`, `attr_single_material`, `attr_single_function`, `attr_composition`, `attr_odd_one_out`,
`attr_count`, `attr_synonym`, `attr_absent`.

A question that asks the model to identify an object is only kept when its answer is unique in that frame. If two
visible objects share the value being asked about, the question is dropped. This "exactly one owner" check applies to
`attr_single_*`, `attr_composition` and `attr_odd_one_out`; weakening it anywhere introduces ambiguous answers.

`attr_count` and `attr_absent` don't identify an object, so their answers are unambiguous by construction: a count is
the size of a set of real objects, and an absence question names a color that no visible object has.

`attr_synonym` asks "Is there a {name} in the scene?". The yes case uses a secondary name recorded for a visible
object, provided no other visible object has that name. The no case uses a real name from the wider catalog that
matches nothing visible. An earlier version only produced "yes", which made a model's accuracy equal to how often it
answered yes. Keep both cases.

`attr_absent` ("Is there a {color} object in the scene?", always "no") uses a real color from the catalog that is
checked to be absent from the frame, never an invented one.

An object counts as present only if its mask covers at least `MIN_MASK_AREA_PX` (100 px, in
`../generate_spatial_gt.py`). Smaller fragments are segmentation noise and used to produce zero-area boxes.

`collect_candidates()` in `gt_attributes.py` returns the full candidate pool for a frame, keyed by the fact each
question depends on (a color value, a `(material, function)` pair, and so on) rather than by catalog entry. Two
entries can share a color or a name; keying by entry put both into the same frame and produced 1,111 duplicate
questions.

## Spatial questions

Spatial questions use MOS frames only. Ego frames have no depth in the dataset, and depth is what makes these
questions reliable.

Comparing raw pixel positions breaks at oblique angles: adding depth flipped 9.4% of leftmost/rightmost answers
compared with pixels alone. The dataset publishes no camera intrinsics, and none are needed. An object's horizontal
or vertical offset from the camera axis is proportional to `(pixel_position - principal_point) * depth`; the unknown
focal length is a positive factor shared by both objects and cancels when comparing them. The one assumption is that
the principal point is at the image center, which is typical for these cameras but not verified, since there is no
calibration file.

Camera roll relative to gravity is not corrected. That would require interpreting the published camera poses in a
documented world-axis convention, which we don't have with enough confidence. If one becomes available, that is the
next fix.

`spatial_farthest` ("farthest from the {reference}") is kept only when the depth-based ranking agrees with the
2D pixel-distance ranking; otherwise the question is dropped.

## Depth loading

`imread_depth()` in `generate_spatial_gt.py` reads depth PNGs as millimetres. An earlier version divided the whole
frame by 1000 whenever any pixel exceeded 5000, to catch frames stored in another unit. That pixel is usually the
sensor's no-return value (65535), so the rule fired on 63% of ordinary frames in a 200-frame sample and corrupted
their units. Ordering questions (closest, A closer than B) were unaffected because the whole frame scaled uniformly,
but absolute thresholds were not. Out-of-range values are now filtered with `DEPTH_MAX_VALID_MM` instead.

## Output schema

Every item has `scene_id`, `kind` (`"mos"` or `"ego"`), `phase` (`0`, `1`, `2` or `null`), `frame`, `mask_path`,
`img_w`, `img_h`, `type`, `question` and `answer`.

- Types that can be answered with a box (`attr_single_*`, `attr_composition`, `attr_odd_one_out`, the yes case of
  `attr_synonym`, `spatial_lr_extreme`, `depth_closest`, `spatial_farthest`) also carry `answer_bbox`,
  `[xmin, ymin, xmax, ymax]` in pixels, computed from the segmentation mask.
- The depth-based spatial types (`spatial_lr_binary`, `spatial_ud_binary`, `spatial_lr_extreme`, `depth_closest`,
  `spatial_farthest`) carry an `evidence` dict with the values the answer was derived from.
- `attr_count` carries `answer_oids`, the `fewsol_id`s that were counted.

`mask_path` is an absolute path under the `--staged-root` used to generate the file. It is useful for debugging but
not portable; drop it before publishing (`scene_id`, `kind`, `phase` and `frame` already identify the frame).

## In-context (two-image) VQA

The in-context tasks reuse five attribute families (`attr_single_color/material/function`, `attr_composition`,
`attr_odd_one_out`) and `spatial_farthest`, but replace the text question with a two-image one. Image 1 is a
reference crop of a single object, chosen deterministically; image 2 is the same MOS or Ego frame as in the
single-image task. The answer is always one box in image 2.

Types: `inctx_attr_single_color`, `inctx_attr_single_material`, `inctx_attr_single_function`,
`inctx_attr_composition`, `inctx_attr_odd_one_out`, `inctx_spatial_farthest`.

**Attribute family.** Image 1 shows a *different* object that shares the relevant value (color, material, function,
material-function pair, or majority material) with the answer. It is never the answer object itself, since the model
could then solve the task by re-identifying the instance instead of reasoning about the attribute. For odd-one-out the
reference must hold the majority material, never the minority answer. The module docstrings of `sos_catalog.py` and
`gt_incontext.py` give the full reasoning.

Because the question says "same color" rather than naming the color, a reference with several values in that field
could make a second visible object a valid answer. `gt_incontext.py` rejects such references (the multi-attribute
ambiguity gate): composition checks the full material × function product, and odd-one-out requires every applicable
material to agree on the answer. References come only from the 70 objects published on Hugging Face (`sos_catalog.py`);
150 of the 220 objects in the local catalog that `lib.py` reads have no published images and can't be used.

Every reference must satisfy `reference_global_object_id != target_global_object_id`, checked first in
`_identity_ok`. By default (`require_diff_category=True`) the reference's catalog `object_name` must also differ,
ignoring case and whitespace, from the target's name in the scene. Despite the parameter name, this compares names,
not categories: it never reads the questionnaire's `category`, and checking `class_name` as well would add nothing
because it equals `object_name` for all 70 objects. What it catches is a different physical copy of the same thing
(`boot` and `boot.2`), which the id check alone allows. On a 7-scene sample it removed no candidates beyond the id
check, so it is always on.

**Spatial family.** Image 1 shows the exact object used as the anchor in `spatial_farthest`. The answer is never the
anchor, so there is no shortcut, and locating the anchor in image 2 is part of the task.

Deduplication matches the single-image tasks: the attribute family is deduplicated across the phase
(`gen_incontext_general_for_phase`, using `dedup.py`), and the spatial family is per frame. Unlike single-image
`spatial_farthest`, it is uncapped.

## Files

| File | Purpose |
|---|---|
| `generate_gt.py` | Command-line driver: scene selection, frame sampling, output; also has `gen_incontext_for_mos_phase` and `gen_incontext_for_ego_phase` |
| `gt_attributes.py` | Attribute VQA; `collect_candidates()` returns the uncapped per-frame pool that `dedup.py` distributes |
| `gt_spatial.py` | Spatial questions, capped per frame; `compute_farthest_candidates()` is reused by the in-context spatial task |
| `dedup.py` | Phase-wide candidate distribution for attribute VQA and the in-context attribute family |
| `lib.py` | Lookup in the local 220-object catalog (`fewsol_id` to parsed questionnaire attributes, cached); not used for in-context references |
| `sos_catalog.py` | The published 70-object catalog (`manifest/object_catalog_v1.json` and `objects_meta/*/questionnaire.json` on Hugging Face) and the join from scene masks (`source_catalog_id`) to `object_id`/`global_object_id`; the only source of in-context references |
| `sos_reference.py` | Deterministic reference crops: frame choice, quality check, padding, caching |
| `gt_incontext.py` | In-context generator: both families, the ambiguity gate and reference selection |
| `../generate_spatial_gt.py` | Shared mask, instance and depth loading |
| `tests/` | Unit and integration tests, checked against already published ground truth where possible |

## Tests

Run `python -m pytest tests -q` from this directory. The unit tests use synthetic masks and catalogs and run offline.
Integration tests read `RPX_TEST_STAGED_PHASE0`, `RPX_TEST_STAGED_EGO`, `RPX_TEST_REFERENCE_MANIFEST`,
`RPX_TEST_SPATIAL_PARQUET` and `RPX_TEST_SOS_ROOT`, and are skipped when that data is not available.
