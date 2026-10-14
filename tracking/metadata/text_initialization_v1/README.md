# Tracking text prompts (v1)

Fixed text prompts, such as `black hammer` or `silver alarm clock`, for every tracked object in RPX. In the paper
they initialize the text-promptable trackers (Grounded SAM 2 and SAM 3.1) in place of a first-frame mask or box.

There are 70 objects, each with one prompt of the form `<primary color> <object name>`. Prompts are given per clip,
and there are 400 clips: 100 scenes × 3 phases (MOS) plus 100 egocentric recordings (Ego). That makes 2,739 rows,
one per object per clip. A prompt stays the same for the whole clip, including frames where the object is occluded.
Rows join to the released masks on `scene_id`, `kind`, `phase` and `mask_index`.

## Usage

```python
import pandas as pd

vocab = pd.read_parquet("scene_condition_vocab.parquet")
clip = vocab.query("scene_id == 'scene001' and kind == 'mos' and phase == 1")
prompts = dict(zip(clip.mask_index, clip.prompt_text))   # {1: 'tan lion', 2: 'white boot', ...}
```

`phase` is `0`, `1` or `2` (Clutter, Interaction, Clean) for MOS and empty for Ego. The `.jsonl` files hold the same
rows as the Parquet files.

## Files

| File | Rows | Contents |
|:--|--:|:--|
| `scene_condition_vocab.parquet` / `.jsonl` | 2,739 | One row per object per clip: `mask_index`, `prompt_text`, the object's identity and the annotation it was joined from |
| `object_catalog_vocab.parquet` / `.jsonl` | 70 | One row per object: name, all authored colors, materials, functions and categories, `primary_color`, `prompt_text` |
| `manifest_summary.json` | | Counts, source revisions, validation results, known limitations and file hashes |
| `SHA256SUMS` | | SHA-256 of every file in this directory |

## How prompts are built

`prompt_text` is `primary_color + " " + display_name`.

`display_name` is the object's catalog name with underscores replaced by spaces (`air_duster_can` becomes
`air duster can`). `primary_color` is the first color in the object's authored questionnaire, lowercased, with
`gray` spelled `grey`. It is fixed per object, not the color that happens to be most visible in a given clip.

Object identities come from the dataset's object catalog and each clip's released `mask_to_object.json`. The VQA
attribute table is only used as a cross-check.

## Known limitations

- 60 Ego mask entries are labeled `unknown`/`unlabeled` in the released annotations. They have no identity, so they
  get no prompt. They are listed in `manifest_summary.json`.
- 252 of 658,254 visible Ego object occurrences (0.038%) use a mask index that has no identity at all in the
  released annotations. They fall in 14 of the 100 Ego clips, and 179 of them are index 8 in `scene058`. MOS is not
  affected. A tracker may therefore see a mask index in those clips with no matching row.
- Colors are free text, so a `primary_color` can be `black and white`. An object has the same prompt in every scene
  it appears in.

## Reproducing

```bash
cd benchmark/scripts/tracking_text_vocab
python3 build_tracking_text_vocab.py --out-dir /path/to/out
python3 validate_tracking_text_vocab.py --bundle-dir /path/to/out --report /path/to/out/validation_report.json
```

The validator exits nonzero on this bundle because of the 252 occurrences above; every other check passes. To
confirm your run matches the release, compare its `missing_mappings_count` with the one in `manifest_summary.json`.
The full coverage check streams every mask archive one clip at a time and deletes each when done, so disk use stays
bounded. `--skip-exhaustive-masks` skips it for quick iteration; don't use it for a release check.

<details>
<summary><b>Design notes</b></summary>

### Why not use the VQA color questions

`vqa/attribute.parquet` only asks an `attr_single_color` question when color alone identifies one visible object in
that frame. It is a discrimination question, not a list of attributes. Building prompts from it would drop every
object whose color is shared with something else on screen, or that was never the subject of a color question. Its
answers also use questionnaire synonyms ("cleaner", "male toy", "analog alarm clock", "joystick") rather than the
catalog name.

Prompts are built instead from each object's catalog identity (`object_name` in `manifest/object_catalog_v1.json`)
and its authored questionnaire (the color, material, function and category lists in
`objects_meta/<id>/questionnaire.json`). Both exist for all 70 published objects. `vqa/attribute.parquet` is read
only to cross-check the result and never fills in a missing attribute.

### Source revisions

Each input is read at a fixed commit of the dataset:

| Purpose | Revision | Origin |
|---|---|---|
| Catalog and questionnaires | `93e31d378f1f98eca18a7aa01a2279c9f332440c` | The dataset's `main` branch when the bundle was built. The two tracking revisions below predate the color, material and function questions; their catalog copies only have category and name for all 70 objects, so they can't be the attribute source. |
| MOS `sam2_meta` and `masks` tars | `2e2a387f7f93e98c177b2e039c141eacda94e5fc` | `PINNED_DATASET_REVISION` in `benchmark/scripts/run_tracking.py` on this repo's `docker/tracking-all-yoloe` branch, the revision the tracking runner pins for MOS |
| Ego `sam2_meta` and `masks` tars | `f082723002bad5800dd85e583115b4ea05734d31` | `PINNED_EGO_DATASET_REVISION` in the same file |
| VQA cross-check | `vqa` branch | `vqa/attribute.parquet`, read only |

### Join

1. `manifest/object_catalog_v1.json` lists the 70 published objects. Each entry's `source_catalog_id` is the join
   key. It is a string (for example `"18.2"`) and is never converted to a number.
2. Each entry's `questionnaire_path` points to `objects_meta/<id>/questionnaire.json`. Its `questions` are matched
   against fixed patterns: "made of" gives material, "used for" function, "color" color, "category" category and
   "name" name. This is the same parsing as
   `data/mask_annotation/visual_grounding_gt/vqa_gt/sos_catalog.py`, copied rather than imported so that this package
   has no dependency on the annotation code.
3. For each clip, `sam2/mask_to_object.json` inside `scenes/<scene>/<phase>/labels/sam2_meta/v1.tar` (MOS) or
   `scenes/<scene>/ego/labels/sam2_meta/v1.tar` (Ego) maps each `mask_index` to
   `{"id": <source_catalog_id>, "name": <name in that scene>}`. Joining `id` to the catalog gives every tracked
   object in the clip its catalog name and attributes.
4. As a check, the object ids in `vqa/attribute.parquet` (non-null `target_oid` and the JSON-encoded `answer_oids`)
   are matched against the same catalog.

### Normalization

- `colors_raw`, `materials_raw`, `functions_raw` and `categories_raw` keep the authored lists verbatim and in order.
- `colors_normalized` is `colors_raw` lowercased and stripped, with internal whitespace collapsed and `gray`
  written as `grey`, the same rule as `_canon_color` in `gt_incontext.py`.
- `primary_color` is always the first entry of `colors_normalized`. The full lists are kept next to it.
- `display_name` is `canonical_name` lowercased with underscores replaced by spaces. None of the 70 `object_name`
  values carries an instance suffix (suffixes live only in `object_id` and `source_catalog_id`, e.g.
  `object_id="baking_tray.1"` with `object_name="baking_tray"`), and all 70 are distinct, so no suffix stripping is
  needed.
- `prompt_text` is `f"{primary_color} {display_name}"`.

### `unknown`/`unlabeled` entries

Some Ego `mask_to_object.json` files contain `{"id": "unknown", "name": "unlabeled"}` for a mask that was never
given an identity. Across all 400 clips this happens for 60 of 2,799 entries, all of them Ego and all with exactly
this pair; MOS has none, and there are no other unresolved ids. With no name or color there is nothing to build a
prompt from, so these entries are left out of `scene_condition_vocab` and listed in `manifest_summary.json` and the
validator's report.

### Ego mask indices with no identity

The validator checks every mask in every one of the 98,124 MOS and Ego frames. It found 252 visible object
occurrences (0.038% of 658,254) whose mask index has pixels in `labels/masks/v1.tar` but no entry in that clip's
`mask_to_object.json`, not even the placeholder above. MOS is clean (0 of 75,000 frames).

This is a gap in the published Ego annotations rather than in this pipeline. For example, `scene058`'s
`mask_to_object.json` lists indices 1, 3–7 and 9 but not 8. Ego clips also have no `masks_aux` tar, which MOS uses as
a second identity source, so there is nothing to build a prompt from without inventing an identity.

14 of the 100 Ego clips are affected. 179 of the 252 occurrences are index 8 in `scene058`, which is present in 179
of that clip's 182 frames. The other 73 are one- to three-frame blips across 13 scenes. Several of those use indices
in the range `10001`–`10009`, which otherwise never occur in those clips, so they look like brief segmentation
artifacts rather than missed objects. The full list of scenes, frames and indices is in `manifest_summary.json`
under `coverage_and_validation.exhaustive_mask_coverage_check.missing_mappings` and in the validator's report.

The bundle covers every mask index that has an identity: all of MOS and 99.96% of Ego occurrences.

### One prompt per clip, not per frame

`scene_condition_vocab` has one row per object per scene, kind and phase. A promptable tracker is usually
initialized once at the start of a clip and needs a prompt for every object it may have to follow, including one that
is occluded for a while and comes back. The per-frame mask check in `validate_tracking_text_vocab.py` only proves that
every mask index appearing in any frame is covered by these clip-level rows. The bundle does not provide per-frame
prompts.

### Authored colors

- Colors are free-text questionnaire answers, not a fixed palette. Values such as `black and white`,
  `grey and yellow` or `white black red and blue` are real answers and appear verbatim in `primary_color` and
  `prompt_text` when they come first.
- `primary_color` is the first authored color. It is not necessarily the dominant or most visible color under a
  particular clip's lighting.
- Attributes belong to the object, not to a scene, so the same physical object has the same prompt everywhere it
  appears.

</details>
