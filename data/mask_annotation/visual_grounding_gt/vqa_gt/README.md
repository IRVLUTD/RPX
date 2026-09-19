# VQA ground truth generation

Generates ground truth for three VQA tasks on RPX scene data:

1. **Attribute VQA** (`gt_attributes.py`) -- "which object is X" questions built from each object's real questionnaire attributes.
2. **Left-right / up-down yes-no** (`gt_spatial.py`) -- binary spatial relationship questions between two named objects.
3. **Spatial bbox** (`gt_spatial.py`) -- "which object is X" questions answered with a bounding box instead of a name, for spatial properties (closest, farthest, leftmost/rightmost).

## Quick start

```bash
python generate_gt.py \
    --staged-root /path/to/staged/scenes \
    --sos-root /path/to/single_objects/sos_wrapped \
    --tier hard \
    --frames-per-phase 30 \
    --out gt_hard.jsonl
```

`--staged-root` must already contain the standard per-scene layout (`<scene>/<0|1|2>/sam2/...`, `<scene>/<0|1|2>/depth/...`, `<scene>/ego/sam2/...`) -- this script does not download scenes itself. `--tier` pulls the real scene list for that difficulty tier from the dataset's `splits/scene_splits.json` on Hugging Face; use `--scenes scene001 scene002 ...` instead to name scenes explicitly. Any named scene not present under `--staged-root` is skipped with a warning, not an error.

Output is one JSON object per line, one question per line. `--max-per-type` (default **5**) caps `attr_single_color`, `attr_single_material`, `attr_single_function`, `attr_composition`, `spatial_lr_binary`, `spatial_ud_binary`, and `spatial_farthest` at 5 questions per frame. `attr_synonym` is always capped at 1 yes + 1 no per frame regardless of `--max-per-type` (a balanced split matters more here than volume). `attr_count` and `attr_odd_one_out` are never capped -- their real per-phase pools are already small. `spatial_lr_extreme` always produces exactly 2 (leftmost and rightmost, unconditionally). `depth_closest` always produces exactly 1 (a single scene-global question). Pass `--max-per-type 0` for no cap anywhere the cap normally applies.

**How the cap is applied differs by task, and this matters:**

- **Attribute VQA is deduped phase-wide, not per-frame.** A phase is a camera sweep around one static object arrangement, so an attribute fact (e.g. "the doll is red") stays true in every frame of that phase -- sampling it independently per frame just repeats the same fact (measured up to 96.7% repeat rate on `attr_count` before this fix). `generate_gt.py`'s `_gen_attributes_for_phase()` collects every candidate fact across all selected frames of a phase first, then `dedup.py`'s `distribute()` places each distinct fact once, in whichever of its valid frames currently has the lightest load, before anything repeats -- so `--max-per-type` here caps *distinct facts per frame*, spread fairly across the whole phase, not a per-frame independent sample.
- **Spatial tasks (`gt_spatial.py`) are deliberately per-frame, un-deduped.** "Is A left of B" is a property of the current camera viewpoint, not the object arrangement -- confirmed on real data: 90% of object pairs flipped `spatial_lr_binary`'s answer at least once across a single phase. Repeating a spatial question across frames of the same phase is a legitimate viewpoint-robustness test, not redundancy, so `--max-per-type` here is a plain per-frame cap (`gt_spatial.py`'s `_cap()`), independently resampled every frame.

## Task 1: Attribute VQA

Types: `attr_single_color`, `attr_single_material`, `attr_single_function`, `attr_composition`, `attr_odd_one_out`, `attr_count`, `attr_synonym`, `attr_absent`.

**The core guarantee: every identify-an-object question only fires when its answer is provably unique in that frame.** If two visible objects share the value being asked about, the question is dropped rather than kept with an ambiguous answer -- this is enforced with an explicit "exactly one owner" check everywhere it applies (`attr_single_*`, `attr_composition`, `attr_odd_one_out`). Weakening that guard anywhere is a correctness bug.

`attr_count` and `attr_absent` aren't identity questions, so uniqueness doesn't apply the same way -- their answers are unambiguous by construction instead (a count is just `len()` of a real set; an absence probe uses a color value checked against everything actually present).

`attr_synonym` asks "Is there a {name} in the scene?" and now has a **real yes/no split**: the yes case uses a genuine secondary name recorded for a visible object (guarded so that name doesn't also belong to some other visible object), and the no case uses a real name from the wider catalog that matches no visible object at all. An earlier version of this generator only ever produced "yes" -- every model's accuracy on it was mathematically identical to how often it said yes, regardless of whether it recognized anything. Don't regress this.

`attr_absent` ("Is there a {color} object in the scene?", always "no") uses the same standard: a real color value pulled from the wider catalog, confirmed genuinely absent from this frame -- never a made-up trap value.

Objects are only considered "present" if their mask covers at least `MIN_MASK_AREA_PX` (100px, in `../generate_spatial_gt.py`) -- a 1-2 pixel mask fragment is segmentation noise, not a meaningfully visible object, and was previously producing degenerate zero-area bounding boxes.

`collect_candidates()` (`gt_attributes.py`) returns the full, uncapped candidate pool for one frame, keyed by whatever fact each candidate actually depends on (e.g. a color value, or a `(material, function)` pair) -- not by which catalog object/folder it came from. Two different catalog entries can share the same color or name and would otherwise be treated as different "facts" and both get placed into the same frame, producing literal duplicate questions (measured: 1,111 such rows before this fix).

## Task 2 & 3: spatial (left-right/up-down yes-no, and bbox)

**mos frames only.** Ego frames have no depth map in this dataset (confirmed against the published HF repo listing -- not a staging gap), and depth is what makes the spatial tasks reliable.

Comparing raw pixel position to decide "which object is more left" or "closer" breaks down at oblique camera angles -- confirmed on real data: recomputing the leftmost/rightmost question with depth factored in flipped **9.4%** of answers versus the pixel-only version. The fix used here doesn't need camera intrinsics (none are published for this dataset): real-world horizontal/vertical offset from the camera axis is proportional to `(pixel_position - principal_point) * depth` -- the unknown focal length is a shared positive constant, so it cancels out when comparing which of two objects is further left, right, up, or down. The one assumption made explicit: principal point ~= image center (standard for these cameras, not verified against a calibration file, because none exists in this dataset).

**Known remaining limitation:** this does not correct for camera roll relative to true gravity. That would need the published camera-pose file interpreted against a documented world-axis convention that isn't available with enough confidence to trust silently -- so it's left uncorrected rather than guessed at. If a future calibration/pose convention becomes available, that's the next fix here.

`spatial_farthest` ("farthest from the {reference}") is kept only when the depth-based 3D ranking agrees with the 2D pixel-distance ranking -- if they disagree, the question is dropped rather than published as a perspective illusion.

## Depth loading

`generate_spatial_gt.py`'s `imread_depth()` loads depth PNGs as millimeters. An earlier version divided the whole frame by 1000 whenever any pixel exceeded 5000, meant to catch a different-unit frame -- but that pixel is very often a sensor "no return" sentinel (65535, the max 16-bit value) rather than a real distance, so the heuristic fired on **63% of ordinary frames** (checked against a 200-frame sample) and silently corrupted their units. Ordering-only comparisons (closest-to-camera, is-A-closer-than-B) were never affected, since the bug scaled an entire frame uniformly -- but any absolute-distance threshold was not safe under it. Sentinel/out-of-range values are now filtered directly (`DEPTH_MAX_VALID_MM`) instead of triggering a frame-wide rescale.

## Output schema

Common fields on every item: `scene_id`, `kind` (`"mos"`/`"ego"`), `phase` (`0`/`1`/`2`/`null`), `frame`, `mask_path`, `img_w`, `img_h`, `type`, `question`, `answer`.

Bbox-eligible types (`attr_single_*`, `attr_composition`, `attr_odd_one_out`, `attr_synonym` yes-case, `spatial_lr_extreme`, `depth_closest`, `spatial_farthest`) also carry `answer_bbox` (`[xmin, ymin, xmax, ymax]` in pixel coordinates, computed directly from the real segmentation mask -- not estimated). The 4 depth-based spatial types (`spatial_lr_binary`, `spatial_ud_binary`, `spatial_lr_extreme`, `depth_closest`, `spatial_farthest`) carry an `evidence` dict recording the real values (proxies/distances) the answer was derived from, for auditability. `attr_count` carries `answer_oids` (the real `fewsol_id`s counted, not just the count).

`mask_path` is an absolute local path into whatever `--staged-root` this was generated against -- useful for local debugging, not portable, and should be dropped when publishing (`scene_id`+`kind`+`phase`+`frame` already identify the frame).

## Files

| File | Purpose |
|---|---|
| `generate_gt.py` | CLI driver -- scene selection, frame sampling, writes output; also exposes `gen_incontext_for_mos_phase`/`gen_incontext_for_ego_phase` (see below) |
| `gt_attributes.py` | Task 1 generator -- also exposes `collect_candidates()`, the uncapped per-frame candidate pool `dedup.py` distributes |
| `gt_spatial.py` | Tasks 2 & 3 generator -- per-frame, independently capped (see above); also exposes `compute_farthest_candidates()`, reused verbatim by the in-context spatial task |
| `dedup.py` | Phase-wide, order-fair candidate distribution used by Task 1 and the in-context general family |
| `lib.py` | SOS catalog lookup (`fewsol_id` -> parsed questionnaire attributes, cached) -- the LOCAL, wider (220-object) catalog; not used for in-context reference selection, see `sos_catalog.py` |
| `sos_catalog.py` | The OFFICIAL, published 70-object SOS catalog (`manifest/object_catalog_v1.json` + `objects_meta/*/questionnaire.json` on Hugging Face) and the scene-mask identity join (`source_catalog_id` -> `object_id`/`global_object_id`) -- the only valid reference-image pool for in-context tasks |
| `sos_reference.py` | Deterministic SOS reference-crop builder (frame selection, quality gate, padding, caching) |
| `gt_incontext.py` | In-context (two-image) generator -- attribute-transfer general family + spatial-anchor family, the cross-modal ambiguity gate, deterministic reference selection |
| `../generate_spatial_gt.py` | Shared geometry/depth utilities (mask loading, instance extraction, depth loading) |
| `tests/` | Unit + integration tests (regression-checked against already-published ground truth where possible) |

## In-context (two-image) VQA

Three additional tasks reuse the five retained attribute families (`attr_single_color/material/function`, `attr_composition`, `attr_odd_one_out`) and `spatial_farthest`, but replace the text-only question with a two-image one: Image 1 is a deterministic SOS reference crop, Image 2 is the same MOS/Ego target frame as the single-image task, and the answer is always exactly one bbox in Image 2.

**General family (attribute transfer, Option B):** Image 1 shows a DIFFERENT SOS object that shares the intended color/material/function/composition-pair/majority-material with the real answer object -- never the answer object itself (that would let a model solve it by instance re-identification instead of attribute reasoning; see `sos_catalog.py`'s and `gt_incontext.py`'s module docstrings for the full analysis, including why odd-one-out's reference must hold the *majority* material and can never be the minority answer). Because the in-context question never names the specific value ("same color", not "same red"), a reference with more than one value in the relevant field can create a second textually-valid answer if some OTHER visible object owns that other value -- `gt_incontext.py` rejects any such candidate (the "multi-attribute ambiguity gate"; composition checks the full material x function cross product, odd-one-out requires every applicable material to agree on the same answer). The reference pool is restricted to the 70 *officially published* SOS objects (`sos_catalog.py`) -- the wider 220-object local catalog `lib.py` scans has no HF-published rgb/mask assets for 150 of those objects, so they cannot serve as an Image-1 source.

**The mandatory rule, always enforced:** `reference_global_object_id != target_global_object_id` (`gt_incontext.py::_identity_ok`, checked first, unconditionally). On top of that, an additional "stricter" check is applied by default (`require_diff_category=True`): the reference's catalog `object_name` must also differ (case/whitespace-insensitively) from the target's local scene name. This is a **lexical same-name check, not a taxonomic/semantic category check** -- despite the name, it does not consult the questionnaire's `category` field at all, and `class_name` is redundant with `object_name` (verified: 0/70 catalog objects have `class_name != object_name`, so checking both is checking the same string twice). What it actually catches is a reference that is a *different physical instance of the same named thing* (e.g. `boot` vs `boot.2`), which the mandatory `global_object_id` rule alone would not exclude. Measured (7-scene sample): this additional check costs **zero** observed coverage loss beyond the mandatory rule -- every candidate that clears the identity check also clears it in practice -- so it's applied unconditionally, but it should not be read as a real category/taxonomy filter.

**Spatial family (Option A, intentionally different):** Image 1 shows the EXACT SOS identity of the existing `spatial_farthest` anchor object -- there's no attribute-transfer shortcut risk here since the answer is never the anchor, and finding the anchor in Image 2's real geometry is a legitimate part of the task.

Types: `inctx_attr_single_color`, `inctx_attr_single_material`, `inctx_attr_single_function`, `inctx_attr_composition`, `inctx_attr_odd_one_out`, `inctx_spatial_farthest`. Dedup semantics mirror the single-image tasks exactly: the general family is phase-wide deduped (`gen_incontext_general_for_phase`, same `dedup.py`), the spatial family is per-frame and uncapped (full density -- unlike the single-image `spatial_farthest`, which applies a `max_per_type` cap).

## Environment

`--sos-root` (or the `RPX_SOS_ROOT` environment variable) must point at a local checkout of the SOS catalog (`single_objects/sos_wrapped/<object>/questionnaire.txt` layout) -- there's no safe machine-independent default. Everything else resolves relative to `--staged-root` or Hugging Face. The in-context tasks additionally reach Hugging Face directly (`sos_catalog.py`, `sos_reference.py`) for the official 70-object catalog and its published rgb/mask tars -- there is no local-only path for these two modules.

## Tests

Run `python -m pytest tests -q` from this directory. Offline unit tests use
synthetic masks and catalogs. Optional integration fixtures are selected with
`RPX_TEST_STAGED_PHASE0`, `RPX_TEST_STAGED_EGO`, `RPX_TEST_REFERENCE_MANIFEST`,
`RPX_TEST_SPATIAL_PARQUET` and `RPX_TEST_SOS_ROOT`; those tests skip when the
corresponding released data or reference crops are absent.
