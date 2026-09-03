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

Output is one JSON object per line, one question per line. By default each of `attr_single_color`, `attr_single_material`, `attr_single_function`, `attr_composition`, `spatial_lr_binary`, `spatial_ud_binary`, and `spatial_farthest` is capped at **5 questions per frame** (random sample of the valid, unique-answer candidates when more than 5 exist -- see `--max-per-type`). Left uncapped, a single busy frame can produce dozens of valid `attr_composition` questions alone (material x function combinations grow fast), which would dominate any downstream sample. Pass `--max-per-type 0` for no cap and the full candidate pool. `spatial_lr_extreme` always produces exactly 2 (leftmost and rightmost, unconditionally -- there are only ever two, so there's nothing to cap or sample). `depth_closest` always produces exactly 1 (a single scene-global question -- "closest to the *camera*" has no reference object to vary it by).

## Task 1: Attribute VQA

Types: `attr_single_color`, `attr_single_material`, `attr_single_function`, `attr_composition`, `attr_odd_one_out`, `attr_count`, `attr_synonym`, `attr_absent`.

**The core guarantee: every identify-an-object question only fires when its answer is provably unique in that frame.** If two visible objects share the value being asked about, the question is dropped rather than kept with an ambiguous answer -- this is enforced with an explicit "exactly one owner" check everywhere it applies (`attr_single_*`, `attr_composition`, `attr_odd_one_out`). Weakening that guard anywhere is a correctness bug.

`attr_count` and `attr_absent` aren't identity questions, so uniqueness doesn't apply the same way -- their answers are unambiguous by construction instead (a count is just `len()` of a real set; an absence probe uses a color value checked against everything actually present).

`attr_synonym` asks "Is there a {name} in the scene?" and now has a **real yes/no split**: the yes case uses a genuine secondary name recorded for a visible object (guarded so that name doesn't also belong to some other visible object), and the no case uses a real name from the wider catalog that matches no visible object at all. An earlier version of this generator only ever produced "yes" -- every model's accuracy on it was mathematically identical to how often it said yes, regardless of whether it recognized anything. Don't regress this.

## Task 2 & 3: spatial (left-right/up-down yes-no, and bbox)

**mos frames only.** Ego frames have no depth map in this dataset (confirmed against the published HF repo listing -- not a staging gap), and depth is what makes the spatial tasks reliable.

Comparing raw pixel position to decide "which object is more left" or "closer" breaks down at oblique camera angles -- confirmed on real data: recomputing the leftmost/rightmost question with depth factored in flipped **9.4%** of answers versus the pixel-only version. The fix used here doesn't need camera intrinsics (none are published for this dataset): real-world horizontal/vertical offset from the camera axis is proportional to `(pixel_position - principal_point) * depth` -- the unknown focal length is a shared positive constant, so it cancels out when comparing which of two objects is further left, right, up, or down. The one assumption made explicit: principal point ~= image center (standard for these cameras, not verified against a calibration file, because none exists in this dataset).

**Known remaining limitation:** this does not correct for camera roll relative to true gravity. That would need the published camera-pose file interpreted against a documented world-axis convention that isn't available with enough confidence to trust silently -- so it's left uncorrected rather than guessed at. If a future calibration/pose convention becomes available, that's the next fix here.

`spatial_farthest` ("farthest from the {reference}") is kept only when the depth-based 3D ranking agrees with the 2D pixel-distance ranking -- if they disagree, the question is dropped rather than published as a perspective illusion.

## Depth loading

`generate_spatial_gt.py`'s `imread_depth()` loads depth PNGs as millimeters. An earlier version divided the whole frame by 1000 whenever any pixel exceeded 5000, meant to catch a different-unit frame -- but that pixel is very often a sensor "no return" sentinel (65535, the max 16-bit value) rather than a real distance, so the heuristic fired on **63% of ordinary frames** (checked against a 200-frame sample) and silently corrupted their units. Ordering-only comparisons (closest-to-camera, is-A-closer-than-B) were never affected, since the bug scaled an entire frame uniformly -- but any absolute-distance threshold was not safe under it. Sentinel/out-of-range values are now filtered directly (`DEPTH_MAX_VALID_MM`) instead of triggering a frame-wide rescale.

## Output schema

Common fields on every item: `scene_id`, `kind` (`"mos"`/`"ego"`), `phase` (`0`/`1`/`2`/`null`), `frame`, `mask_path`, `type`, `question`, `answer`.

Bbox-eligible types (`attr_single_*`, `attr_composition`, `attr_odd_one_out`, `attr_synonym` yes-case, `spatial_lr_extreme`, `depth_closest`, `spatial_farthest`) also carry `answer_bbox` (`[xmin, ymin, xmax, ymax]` in pixel coordinates, computed directly from the real segmentation mask -- not estimated). Yes/no types carry an `evidence` dict recording the real values the answer was derived from, for auditability.

## Files

| File | Purpose |
|---|---|
| `generate_gt.py` | CLI driver -- scene selection, frame sampling, writes output |
| `gt_attributes.py` | Task 1 generator |
| `gt_spatial.py` | Tasks 2 & 3 generator |
| `lib.py` | SOS catalog lookup (`fewsol_id` -> parsed questionnaire attributes) |
| `../generate_spatial_gt.py` | Shared geometry/depth utilities (mask loading, instance extraction, depth loading) |

## Environment

`--sos-root` (or the `RPX_SOS_ROOT` environment variable) must point at a local checkout of the SOS catalog (`single_objects/sos_wrapped/<object>/questionnaire.txt` layout) -- there's no safe machine-independent default. Everything else resolves relative to `--staged-root` or Hugging Face.
