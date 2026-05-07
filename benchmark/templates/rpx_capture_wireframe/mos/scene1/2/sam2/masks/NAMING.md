# Mask convention (multi-object scene)

One PNG per frame, named `00000.png` … `<N-1>.png` (5-digit zero-padded).
Frame indices align with the matching `rgb/00042.png`.

Mask values:

* `0`         — background
* `1, 2, 3 …` — instance IDs for individual objects in the scene

Single-channel 8-bit PNG. The mapping from instance ID to canonical
object ID (which then keys into `objects_meta/<obj>/questionnaire.json`
on the HF repo) lives at the sibling `sam2/mask_to_object.json`.
