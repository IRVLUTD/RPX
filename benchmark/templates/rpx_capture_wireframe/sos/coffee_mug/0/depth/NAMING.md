# Frame naming convention

Files in this directory are named `00000.png` through `<N-1>.png`,
zero-padded to 5 digits, where `N` is the per-phase frame count
(typically 250 for v1).

* `rgb/`           — 3-channel 8-bit RGB PNG, captured by D435 colour stream.
* `depth/`         — single-channel 16-bit PNG, depth in millimetres (D435 depth stream, post-rectification).
* `fisheye/left/`  — single-channel 8-bit PNG, T265 left fisheye.
* `fisheye/right/` — single-channel 8-bit PNG, T265 right fisheye.

Frame indices align across modalities: `rgb/00042.png`, `depth/00042.png`,
`fisheye/left/00042.png`, etc. all describe the same instant.
