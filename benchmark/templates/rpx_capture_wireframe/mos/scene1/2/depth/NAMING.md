# Frame naming convention

Files: `00000.png` … `<N-1>.png` (5-digit zero-padded). Frame indices
align across all modalities for the same `(scene, phase)`.

Modalities:
* `rgb/`           — D435 colour, 8-bit RGB PNG
* `depth/`         — D435 depth, 16-bit grayscale PNG, millimetres
* `fisheye/left/`  — T265 left fisheye, 8-bit grayscale PNG
* `fisheye/right/` — T265 right fisheye, 8-bit grayscale PNG
