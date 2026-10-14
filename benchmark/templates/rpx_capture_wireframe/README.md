# RPX capture layout

This is the on-disk layout the raw RPX captures (about 890 GB) must follow before the
[dataset hub](../../rpx_benchmark/dataset_hub/README.md) can scan, pack and upload them. The Hub layout those
commands produce is documented there.

```
DATA/
├── mos/            multi-object scenes: scene1 … scene100
├── sos/            single-object scans, one directory per object
└── ego/            egocentric captures, written by `prepare-ego`
```

The scanner ignores anything else under the root and lists it as `skipped`.

## Multi-object scenes

```
mos/scene<N>/
└── <phase>/                       0 = Clutter, 1 = Interaction, 2 = Clean
    ├── rgb/          00000.png …  8-bit RGB, D435
    ├── depth/        00000.png …  16-bit depth in mm, D435
    ├── fisheye/
    │   ├── left/     00000.png …  8-bit grayscale, T265 left
    │   └── right/    00000.png …  8-bit grayscale, T265 right
    ├── cam_pose/     00000.npz …  T265 pose
    └── sam2/
        ├── masks/                 8-bit instance masks
        ├── bbox_overlay/, contour_gt_masks/, dino_output/,
        │   masks_contour_with_hidden/, palette/, rgb_and_mask/
        │                          visualizations, packed together as masks_aux
        ├── mask_to_object.json    mask id → object id
        ├── verified_masks.txt     frames checked by hand
        └── iter1_faulty.txt … iter4_faulty.txt
```

Each phase has 250 frames. MOS scenes have no questionnaire of their own: each mask is mapped to an object through
`mask_to_object.json`, and the object's questionnaire comes from its single-object scan.

## Single-object scans

```
sos/<object_name>/
├── questionnaire.txt              FewSOL-style questions and answers about the object
└── 0/                             one 360° turntable sequence
    └── same modality directories as MOS; masks hold 0 and 1,
        and mask_to_object.json is {"1": "<object_name>"}
```

The directory name is the object's canonical ID (`tape_and_holder`, `coffee_mug`, no numeric prefix). Use exactly
the same string as the value in MOS `mask_to_object.json` files so the packer can write one
`objects_meta/<object_id>/questionnaire.json` per object. Names must be unique. The local catalog holds 220 objects,
of which 70 are in the public release.

## Rules

- Frames are named `00000.png` with five digits, so sorting by name sorts by time.
- Frame indices line up across modalities: `rgb/00042.png`, `depth/00042.png` and `cam_pose/00042.npz` are one
  sample.
- Mask value 0 is always background and never appears in `mask_to_object.json`.

## Checking a tree

```bash
python -m rpx_benchmark.dataset_hub.cli scan /path/to/DATA
```

`scan` lists every scene it found by ID, so a misnamed directory such as `scence1` shows up there. Top-level
entries other than `mos/`, `sos/` and `ego/`, and loose files where a scene directory is expected, are listed as
`skipped`.
