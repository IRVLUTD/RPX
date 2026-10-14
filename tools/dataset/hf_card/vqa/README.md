# VQA ground truth

Machine-generated question-answer pairs over all 100 RPX scenes, at full frame density: every frame of
every Clutter, Interaction and Clean phase and of the Ego sequence. The single-image files back task T5
(VQA) and the two-image files back task T6 (In-context VQA). Answers come from each object's real catalog
attributes and from metric depth.

To score a model, use the [benchmark subset](metadata/benchmark_subset_v1/BENCHMARK_SUBSET_README.md)
rather than these full banks.

<p align="center">
  <a href="#files">Files</a> ·
  <a href="#single-image-questions">Single-image questions</a> ·
  <a href="#in-context-questions">In-context questions</a> ·
  <a href="#schema">Schema</a> ·
  <a href="#validation-and-provenance">Validation and provenance</a>
</p>

## Files

| File | Rows | Questions |
|:--|--:|:--|
| `attribute.parquet` | 3,241,564 | which object has a given attribute |
| `spatial_binary.parquet` | 744,741 | left/right and above/below, yes or no |
| `spatial_bbox.parquet` | 543,017 | spatial questions answered with a box |
| `incontext_mos_attribute_bbox.parquet` | 317,873 | attribute transfer, MOS |
| `incontext_ego_attribute_bbox.parquet` | 113,193 | attribute transfer, Ego |
| `incontext_mos_spatial_bbox.parquet` | 318,017 | farthest from a reference, MOS |

`metadata/incontext_v1/` holds the in-context release record: `VALIDATION_REPORT.json`,
`FUNCTION_ALIAS_DECISION.md`, `SHA256SUMS` and `RELEASE_MANIFEST.json`. The in-context Parquets are
full-density question banks, with every eligible fact turned into a row; the sampled benchmark lives in
`metadata/benchmark_subset_v1/`.

## Single-image questions

Attribute questions ask which object matches a color, material, function or composition, which one is
the odd one out, how many objects share an attribute, which object a synonym refers to, or whether an
object is absent. Spatial questions compare two named objects (left/right, above/below) or ask for the
object closest to the camera, farthest from a reference, or furthest left or right.

Spatial files cover MOS only, because Ego frames have no depth map. Generation enforces unique answers,
handles depth units explicitly, and deduplicates attribute facts across each phase. The generator is in
[`data/mask_annotation/visual_grounding_gt/vqa_gt/`](https://github.com/IRVLUTD/RPX/tree/main/data/mask_annotation/visual_grounding_gt/vqa_gt)
in the code repository.

## In-context questions

Every in-context row asks about a pair of images:

1. Image 1 (`reference_image`) is a deterministic crop of one reference object from its single-object
   scan (SOS).
2. Image 2 (`target_image`) is a MOS or Ego frame from the single-image files, with the same
   `scene_id`, `kind`, `phase` and `frame` meaning.
3. `answer_bbox` is `[xmin, ymin, xmax, ymax]` in Image 2's original pixels. `img_w` and `img_h`
   describe Image 2, never the crop.

| Type | Question |
|:--|:--|
| `inctx_attr_single_color` | same color as the object in Image 1 |
| `inctx_attr_single_material` | same material |
| `inctx_attr_single_function` | same purpose |
| `inctx_attr_composition` | same material and purpose |
| `inctx_attr_odd_one_out` | not made of the same material |
| `inctx_spatial_farthest` | farthest from the object in Image 1 (MOS only) |

The five attribute types show a different object in Image 1, one that shares the asked attribute with
the answer. A candidate reference is rejected if any of its other attribute values also belongs to
another object visible in Image 2, so each question has exactly one defensible answer. References are
chosen deterministically by a SHA-256 ranking. The spatial type instead shows the anchor object
itself, which is expected to appear in Image 2.

For matching and ambiguity checks only, `gray` and `grey` count as the same color, compound materials
such as `wood/metal` are split into their parts, and `to drink`/`drink`, `to eat`/`eat` and
`to snack`/`snack` count as the same function. Stored text is never rewritten.

Because the published attribute data records some functions under both spellings, 426 rows of
`incontext_mos_attribute_bbox.parquet` (0.13%; none in Ego) describe the same reference, target and
frame twice. Both rows are kept, since each pairs exactly with its own published source row.
[`FUNCTION_ALIAS_DECISION.md`](metadata/incontext_v1/FUNCTION_ALIAS_DECISION.md) lists them.

Ego has no spatial questions of either kind in this release, single-image or in-context.

## Schema

Every single-image row has `scene_id`, `kind` (`mos` or `ego`), `phase` (`0`, `1`, `2` or null),
`frame`, `img_w`, `img_h`, `type`, `question` and `answer`. Box questions add `answer_bbox`
(`[xmin, ymin, xmax, ymax]` in pixels). Depth-based spatial types add `evidence`, a JSON string with
the measured values behind the answer, and `attr_count` adds `answer_oids` (also a JSON string). Mask
paths are omitted; `scene_id`, `kind`, `phase` and `frame` identify the frame.

In-context rows carry `source_sample_id`, the exact single-image row each one pairs with: same frame,
type, target object and box. It is the only join key the release stores.

<details>
<summary>Full in-context schema</summary>

The three in-context files share one column set. Attribute files fill the `attribute_*` columns and
leave `evidence` and `_dedup_key` null; the spatial file does the reverse. Unlike
`spatial_bbox.parquet`, the spatial file stores `evidence` as a native Parquet struct, so read it
directly instead of with `json.loads`.

```text
scene_id: string
kind: string
phase: double                  (0/1/2/null; int64 in the spatial file)
frame: string
img_w: int64
img_h: int64
type: string
question: string
answer: string
answer_bbox: list<int64>
evidence: struct<distances_2d_px: struct<name: double, ...>, reference: string>
                               (spatial file; one distance per catalog object)
schema_version: string
generator_version: string
dataset_revision: string
sample_id: string
source_type: string
source_question: string
target_image: struct<member, repo_id, revision, shard: string>
reference_image: struct<crop_bbox: list<int64>, member, repo_id, revision,
                        shard: string>
target_local_mask_id: int64
target_source_catalog_id: string
target_object_id: string
target_global_object_id: int64
reference_source_catalog_id: string
reference_object_id: string
reference_global_object_id: int64
reference_frame: string
reference_mask_bbox: list<int64>
reference_crop_bbox: list<int64>
reference_crop_sha256: string
attribute_kind: string         (attribute files)
attribute_value: string        (attribute files)
attribute_material: string     (attribute files)
attribute_function: string     (attribute files)
answer_center_x_norm: double
answer_center_y_norm: double
answer_is_centered: bool
different_category: bool       (attribute files)
source_sample_id: string       (attribute files)
_dedup_key: string             (spatial file; build bookkeeping)
```

</details>

## Validation and provenance

In-context image locators are pinned to dataset commit `93e31d378f1f98eca18a7aa01a2279c9f332440c`.
The in-context generator,
[`gt_incontext.py`](https://github.com/IRVLUTD/RPX/blob/main/data/mask_annotation/visual_grounding_gt/vqa_gt/gt_incontext.py),
was released from commit `fbf2fd03cd9972d159ac954cfed8fd194deb8d87`.

<details>
<summary>What the validation report checks</summary>

Almost every check in `VALIDATION_REPORT.json` runs on every row: box validity, identity and
reference-visibility rules, duplicate sample IDs, semantic duplicates (raw and alias-normalized),
reference crop availability and hashes, pinned revisions, row-for-row pairing with the published
single-image files, and frame coverage across all MOS and Ego cells. Each of the 318,017 spatial rows
equals one published `spatial_farthest` row.

The one sampled check is an independent recomputation of the farthest-object geometry on a subset of
scenes, run in addition to the exhaustive equality check above.

</details>

| File | SHA-256 |
|:--|:--|
| `incontext_mos_attribute_bbox.parquet` | `86a334247198848390c158537a235937dda2d386c022d56ede3bf135d798a94b` |
| `incontext_ego_attribute_bbox.parquet` | `398565fc10b58f5beec3f237c5efb8cf0fd18123fed2782f6ab1c15561a9c186` |
| `incontext_mos_spatial_bbox.parquet` | `6652cba3b471166e4f2dd366d9434495e63a7b80b5271cca61eeceb2ac0ec156` |

---

<sub>Part of <a href="https://github.com/IRVLUTD/RPX">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset card</a></sub>
