# RPX capture wireframe

This directory is the **on-disk source layout** the team should arrange
the real ~890 GB capture set in. Once the data is laid out this way on
the target system, the dataset hub pipeline (in
`rpx_benchmark.dataset_hub`) walks it, packs per-modality tar shards,
writes the per-frame manifest, dedupes per-object questionnaires, and
pushes the result to `IRVLUTD/RPX` on HuggingFace.

Two scene families live under their own top-level subdirectories
(`mos/` and `sos/`). Anything else under the root is ignored by the
scanner and reported in its `skipped` list.

```
test_dataset_aggregated/
├── mos/                    # multi-object scenes (100 of them)
│   ├── scene1/
│   ├── scene2/
│   └── ... scene100/
└── sos/                    # single-object scenes (220 of them)
    ├── tape_and_holder/    # ← bare object name = canonical object_id
    ├── coffee_mug/
    └── ... 220 unique objects ...
```

**Naming invariants (the team must follow exactly):**

* MOS dirs are named `scene<N>` (no building/area suffix). N runs 1..100.
* SOS dirs are named with the **bare object name only** (`tape_and_holder`,
  `coffee_mug`, …) — no `object001.` prefix. The directory name itself
  is the canonical `object_id` used everywhere downstream
  (`mask_to_object.json`, `objects_meta/<object_id>/questionnaire.json`).
* Object names are unique across the 220 SOS scenes (no two objects
  share a directory name).

## Multi-object scenes (MOS) — under `mos/`

```
mos/scene<N>/                      # e.g. mos/scene1, mos/scene42
└── <phase>/                       # 0 = clutter, 1 = interaction, 2 = clean
    ├── rgb/             00000.png … 00249.png   8-bit RGB, D435
    ├── depth/           00000.png … 00249.png   16-bit grayscale, D435 (mm)
    ├── fisheye/
    │   ├── left/        00000.png … 00249.png   8-bit grayscale, T265 left
    │   └── right/       00000.png … 00249.png   8-bit grayscale, T265 right
    ├── cam_pose/        00000.json … 00249.json T265 SLAM pose, see cam_pose/NAMING.md
    └── sam2/
        ├── masks/                       00000.png … 00249.png   8-bit instance masks
        ├── bbox_overlay/                00000.png …             SAM2 bbox visualisations
        ├── contour_gt_masks/            00000.png …             contour overlays
        ├── dino_output/                 00000.png …             DINO features
        ├── masks_contour_with_hidden/   00000.png …             extra mask viz
        ├── palette/                     00000.png …             colour-coded mask viz
        ├── rgb_and_mask/                00000.png …             RGB + mask composite
        ├── mask_to_object.json          {mask_id → canonical object_id}
        ├── verified_masks.txt           list of hand-verified frames
        └── iter1_faulty.txt … iter4_faulty.txt   per-iteration flagged frames
```

* **Three phases per MOS scene.**
* **250 frames per phase**, 5-digit zero-padded filenames, indices align
  across all modalities.
* **No `questionnaire.txt`** at the scene root for MOS — questionnaires
  are looked up *per object* through `sam2/mask_to_object.json`. The
  packer uploads one `objects_meta/<object_id>/questionnaire.json` per
  unique object, shared between MOS and SOS, no duplication.

## Single-object scenes (SOS) — under `sos/`

```
sos/<object_name>/                 # e.g. sos/tape_and_holder, sos/coffee_mug
├── questionnaire.txt              # FewSOL-style Q&A, see file for format
└── 0/                             # only one "phase" — a 360° turntable view
    ├── rgb/             …  (same modality structure as MOS, sans phase)
    ├── depth/           …
    ├── fisheye/{left,right}/
    ├── cam_pose/
    └── sam2/
        ├── masks/                       8-bit single-channel; values in {0, 1}
        ├── bbox_overlay/ … rgb_and_mask/   (same auxiliary viz dirs as MOS)
        ├── mask_to_object.json          trivially {"1": "<scene_dir_name>"}
        ├── verified_masks.txt
        └── iter1_faulty.txt … iter4_faulty.txt
```

* **One phase only** (`0/`).
* **Scene directory name *is* the canonical object ID.** Use exactly
  this string in MOS `mask_to_object.json` so the packer can dedupe
  questionnaires across scene families.
* **One `questionnaire.txt`** at the scene root (not inside `0/`). See
  `object001.tape_and_holder/questionnaire.txt` for the expected
  FewSOL-style format.

## Conventions to honour

| Convention | Why it matters |
|---|---|
| Frame names `00000.png`, 5-digit zero-padded | Lexicographic sort = temporal sort; loaders depend on it. |
| Frame indices align across modalities | A loader reads `rgb/00042.png` + `depth/00042.png` + `cam_pose/00042.json` as one sample. |
| Object ID = SOS scene directory name (bare object name) | Single canonical key; dedupes questionnaires across MOS/SOS. |
| Mask value 0 = background (always) | Implicit; not listed in `mask_to_object.json`. |
| 250 frames per phase | Manifest size and shard size estimates assume this. |

## What lives WHERE on the HF repo

When the packer ships this layout to HuggingFace, it produces:

```
IRVLUTD/RPX/
├── manifest/
│   ├── frames_v1.parquet        # one row per (scene, phase, frame); the source of truth
│   └── current.json             # {label_versions: {masks: v1, ...}}
├── splits/
│   ├── scene_splits.json
│   ├── easy.txt                 # MOS scenes only
│   ├── medium.txt
│   └── hard.txt
├── scenes/<scene_id>/<phase>/   # MOS — 100 × 3 phases
│   ├── rgb.tar, depth.tar, fisheye.tar, cam_pose.tar
│   └── labels/{masks,masks_aux,sam2_meta,vqa}/v1.tar
├── objects/<object_id>/0/       # SOS — 220 scenes
│   └── (same modality tars as scenes/)
└── objects_meta/<object_id>/    # the questionnaire dedup layer
    └── questionnaire.json       # ONE per unique object, shared by SOS and MOS
```

A user running `rpx download --task vqa --split easy` pulls:

* `manifest/frames_v1.parquet`, `manifest/current.json`  (always)
* `scenes/<easy-scene>/{0,1,2}/rgb.tar`                  (recipe input)
* `scenes/<easy-scene>/{0,1,2}/labels/vqa/v1.tar`        (recipe label)
* `objects_meta/<object_id>/questionnaire.json`          (one per object referenced by easy-scene `mask_to_object.json` files)

A subsequent `rpx download --task segmentation --split easy` is a
near-no-op for the cache: `rgb.tar` already on disk; only
`labels/masks/v1.tar` is fetched as the delta.

## Reserved for future versions

* `labels/vqa/v1.tar` is a placeholder slot — VQA generation has not
  landed yet. The packer writes an empty tar (or skips the file
  entirely) until the labels exist.
* `objects_meta/<object_id>/{template_3d.glb, canonical_pose.json}`
  are slots for future per-object artefacts (3D meshes, canonical
  template poses) that we may add post-NeurIPS.

## Validating your layout against this wireframe

Once your real captures are arranged, run:

```bash
python -m rpx_benchmark.dataset_hub.cli scan /path/to/test_dataset_aggregated
```

Anything in the source root that isn't a `scene*` or `object*` directory
ends up in the scanner's `skipped` list — useful for catching typos
(`scence1.library...`) or stray files.
