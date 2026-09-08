# Tracking text-initialization vocabulary v1

## Intended use

A constant, per-video (scene-condition) text prompt for every tracked
object, for text-promptable trackers that need a fixed initialization label
rather than a mask or box. The prompt is:

    <primary color> <canonical object name>

e.g. `black hammer`, `white boot`, `silver alarm clock`, `grey air duster can`.

This vocabulary is **constant for the whole clip** -- one prompt per object
per (scene, kind, phase), not one prompt per frame. A tracked object keeps
its one fixed prompt even in frames where it is temporarily occluded; this
is intentional (see "Scene-condition vocabulary vs. per-frame visibility"
below), not a bug to fix.

## Why question strings alone are insufficient

`vqa/attribute.parquet`'s `attr_single_color` question is only emitted when
color uniquely identifies one visible object in that specific frame -- it is
a discrimination question, not an attribute listing. Reusing it as a
vocabulary source would silently omit every object whose color happens to
be shared with something else on screen, or that simply never became the
subject of a color question. It also answers with a synonym drawn from the
object's questionnaire ("cleaner", "male toy", "analog alarm clock",
"joystick", ...), not the catalog's canonical identity.

The authoritative source is the object's own **catalog identity**
(`manifest/object_catalog_v1.json`'s `object_name`) and its own **authored
questionnaire** (`objects_meta/<id>/questionnaire.json`'s color/material/
function/category lists) -- present for every one of the 70 officially
published objects regardless of whether any VQA question ever used it.
`vqa/attribute.parquet` is used here only as an independent cross-check
(see below), never to fill in an attribute the questionnaire doesn't have.

## Source revisions (all immutable 40-character commit SHAs)

Three distinct revisions are used, each for a different purpose -- do not
conflate them:

| Purpose | Revision | Origin |
|---|---|---|
| Catalog + questionnaires (color/material/function/category/name) | `93e31d378f1f98eca18a7aa01a2279c9f332440c` | Resolved from the dataset's `main` branch. The two tracking-pinned revisions below predate color/material/function questions being authored -- their catalog copies have category/name only, confirmed empirically across all 70 objects, and must not be used as the attribute source. |
| MOS `sam2_meta`/`masks` tars | `2e2a387f7f93e98c177b2e039c141eacda94e5fc` | `PINNED_DATASET_REVISION` in `benchmark/scripts/run_tracking.py` on the `docker/tracking-all-yoloe` branch of this repo -- the revision the D3 tracking benchmark runner already pins for MOS. |
| Ego `sam2_meta`/`masks` tars | `f082723002bad5800dd85e583115b4ea05734d31` | `PINNED_EGO_DATASET_REVISION`, same file. |
| VQA cross-check only | `vqa` (the dataset's `vqa` branch) | `vqa/attribute.parquet`, read-only, see below. |

## Join logic

1. **Catalog identity**: `manifest/object_catalog_v1.json` lists exactly 70
   officially published objects. Each entry's `source_catalog_id` (a
   string -- e.g. `"18.2"` -- never coerced to a numeric type anywhere in
   this pipeline) is the join key.
2. **Constant attributes**: each catalog entry's `questionnaire_path` points
   to `objects_meta/<id>/questionnaire.json`, whose `questions` dict is
   parsed by matching question text against fixed patterns ("made of" ->
   material, "used for" -> function, "color" -> color, "category" ->
   category, "name" -> name) -- the same pattern
   `data/mask_annotation/visual_grounding_gt/vqa_gt/sos_catalog.py` uses for
   VQA ground truth, ported (not imported) here to keep this package free of
   a cross-branch dependency.
3. **Scene-condition identity**: for every MOS `scenes/<scene>/<phase>/
   labels/sam2_meta/v1.tar` and every Ego `scenes/<scene>/ego/labels/
   sam2_meta/v1.tar`, `sam2/mask_to_object.json` maps each `mask_index` to
   `{"id": <source_catalog_id>, "name": <scene-local name>}`. Joining `id`
   against the catalog's `source_catalog_id` gives every tracked object in
   that clip its canonical name and constant attributes.
4. **VQA cross-check** (report-only, never authoritative): `vqa/
   attribute.parquet`'s non-null `target_oid` and JSON-encoded `answer_oids`
   are parsed and their union is checked against the same catalog. This
   proves consistency; it never fills in a missing attribute.

## Normalization

- `colors_raw`/`materials_raw`/`functions_raw`/`categories_raw` preserve the
  complete authored list, verbatim, in authored order.
- `colors_normalized` = `colors_raw` lowercased, stripped, with internal
  whitespace collapsed, and the one known spelling variant `gray`
  canonicalized to `grey` (matching `gt_incontext.py`'s `_canon_color` rule
  from the VQA in-context work).
- `primary_color` is **always the first entry of `colors_normalized`** --
  deterministic by construction, never chosen by frequency, never random.
  The complete `colors_raw`/`colors_normalized` lists are preserved
  alongside it so nothing is lost.
- `display_name` = `canonical_name` lowercased with underscores replaced by
  spaces. Every one of the 70 `object_name` values is already free of
  instance suffixes (the suffix, where one exists, lives only in `object_id`
  / `source_catalog_id`, e.g. `object_id="baking_tray.1"` but
  `object_name="baking_tray"`) and is unique across the catalog -- audited
  empirically (0 collisions among 70 `object_name` values) before deciding
  no suffix-stripping logic was needed at all.
- `prompt_text` = `f"{primary_color} {display_name}"`.

## Sentinel entries: "unknown"/"unlabeled"

Some Ego `mask_to_object.json` entries carry the literal placeholder
`{"id": "unknown", "name": "unlabeled"}` for a tracked mask instance that
was never assigned a real object identity. Confirmed empirically across all
400 scene-conditions: 60 of 2,799 raw entries, 100% in `kind=ego`, 100%
exactly this `id`/`name` pair -- never seen in MOS, never any other
unresolved id. These carry no color or name to build a prompt from, so they
are excluded from `scene_condition_vocab` **by design, not silently**:
every excluded entry is reported (see `manifest_summary.json` and the
validator's report), separately from a genuine unresolved catalog gap
(there were zero of those).

## Known limitation: undeclared Ego mask indices (documented, not fixed)

The validator's exhaustive per-frame mask-coverage check (requirement 9,
streaming every one of 98,124 real MOS+Ego frames) found 252 visible
frame-object occurrences (0.038% of all 658,254 visible frame-object
occurrences; MOS: 0/75,000 frames affected) where a mask index appears with
real pixels in `labels/masks/v1.tar` but has **no entry at all** --
not even the `"unknown"`/`"unlabeled"` sentinel above -- in that
scene-condition's own `sam2/mask_to_object.json`. Confirmed to be a real gap
in the published Ego annotations, not a defect in this pipeline: verified by
downloading and inspecting the raw `sam2_meta` JSON directly (e.g. scene058's
mask index `8` is genuinely absent from its `mask_to_object.json`, which
lists only indices 1, 3-7, 9), and Ego scene-conditions have no `masks_aux`
tar (unlike MOS) that could supply an alternate identity source. There is
therefore no name, color, or any other attribute anywhere to build a prompt
from for these specific mask indices -- they cannot be added to
`scene_condition_vocab` without inventing an identity, which this pipeline
will not do.

Affected: 14 of 100 Ego scene-conditions (86 are completely clean); 179 of
the 252 occurrences (71%) are concentrated in a single scene (`scene058`,
mask index `8`, present in 179 of that clip's 182 frames); the remaining 73
occurrences are 1-3-frame blips spread across 13 other scenes, several with
a mask index in a distinct 5-digit range (`10001`-`10009`) that never
otherwise appears in that clip's own small integer indexing, suggesting a
brief segmentation artifact rather than a real missed tracked object. The
complete scene/frame/index list is in `manifest_summary.json`'s
`coverage_and_validation.exhaustive_mask_coverage_check.missing_mappings`
and the validator's own JSON report.

This is published as a known, quantified, documented limitation rather than
withheld: the vocabulary is complete and correct for every mask index that
*does* have a declared identity (100% of MOS, 99.96% of Ego occurrences),
and blocking the whole release over an upstream annotation gap this pipeline
has no way to fill would not serve anyone. A tracker consuming this
vocabulary should be prepared to encounter a mask index in these 14 scenes
that has no corresponding `scene_condition_vocab` row.

## Scene-condition vocabulary vs. per-frame visibility

`scene_condition_vocab` is **one row per object per (scene, kind, phase)**,
not one row per object per frame. It answers "what may this tracker need to
initialize on, across the whole clip" -- a promptable tracker is normally
initialized once, at the start of a clip, from a released frame-0
annotation, and needs a stable prompt for every object it might be asked to
track through the clip, including one that becomes occluded partway through
and reappears later. The exhaustive per-frame mask audit performed by
`validate_tracking_text_vocab.py` (streaming every mask PNG in every clip)
exists only to *prove* that every mask index that ever actually appears in
any frame is covered by this constant, clip-level vocabulary -- it is a
coverage proof, not a step toward a changing, per-frame prompt vocabulary,
which this release deliberately does not produce.

## Prompt examples

    black hammer
    white boot
    silver alarm clock
    grey air duster can
    tan lion
    red doll
    yellow and white mustard bottle

## Limitations of authored color descriptions

- Authored colors come from free-text questionnaire answers, not a
  controlled color palette -- values like `"black and white"`, `"grey and
  yellow"`, or `"white black red and blue"` are real, valid entries and
  will appear verbatim as `primary_color` (and therefore inside
  `prompt_text`) whenever they happen to be the first authored color for an
  object. This is not a parsing bug; it reflects how the object was
  actually described.
- `primary_color` is positional (first-authored), not perceptually
  dominant, most-frequent, or most-salient under the actual camera
  lighting/exposure of any particular clip -- an object's real-world
  appearance in a given scene may differ from its catalog `primary_color`.
- Authored attributes are per-object-identity, not per-scene-instance --
  the same catalog object (and therefore the same `prompt_text`) is reused
  everywhere that exact physical object appears, across every scene and
  condition it was placed in.

## Files in this bundle

| File | Rows | Contents |
|---|---|---|
| `object_catalog_vocab.parquet` / `.jsonl` | 70 | One row per officially published catalog object: identity, full authored attribute lists, deterministic `primary_color`/`prompt_text`. |
| `scene_condition_vocab.parquet` / `.jsonl` | see `manifest_summary.json` | One row per object per scene-condition: identity, `mask_index`, `prompt_text`, source revision, and the exact `sam2_meta` shard/member locator it was joined from. |
| `manifest_summary.json` | -- | Generation timestamp, generator commit, all immutable source revisions, counts, coverage/validation results, output hashes. |
| `SHA256SUMS` | -- | SHA-256 of every file in this bundle. |

## Reproduce

```bash
cd benchmark/scripts/tracking_text_vocab
python3 build_tracking_text_vocab.py --out-dir /path/to/out
python3 validate_tracking_text_vocab.py --bundle-dir /path/to/out --report /path/to/out/validation_report.json
```

The validator's exhaustive per-frame mask-coverage check streams every MOS
and Ego mask archive one condition at a time, staging and deleting local
files as it goes, so disk usage stays bounded regardless of corpus size.
Omit `--skip-exhaustive-masks` for a real release check; that flag exists
for fast iteration only.

**Expected exit status**: the validator exits nonzero even against this
exact released bundle, because of the "Known limitation: undeclared Ego
mask indices" above (requirement 9's exhaustive mask-coverage check reports
252 real, documented, upstream-data gaps -- see that section). Every other
check passes. This is expected, not a sign the reproduction failed; compare
your `missing_mappings_count` against the value recorded in
`manifest_summary.json` to confirm you reproduced the same result.
