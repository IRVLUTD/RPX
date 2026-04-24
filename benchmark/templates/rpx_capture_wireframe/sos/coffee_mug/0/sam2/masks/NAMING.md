# Mask convention (single-object scene)

One PNG per frame, named `00000.png` … `<N-1>.png` (5-digit zero-padded).
Frame indices align with the matching `rgb/00042.png`.

Mask values:

* `0` — background
* `1` — the object

Single-channel 8-bit PNG. The object ID for this entire scene is the
scene-directory name itself (e.g. `object001.tape_and_holder`); see the
`mask_to_object.json` in the sibling `sam2/` directory.
