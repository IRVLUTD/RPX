# Official ZipDepth RPX overlay

This thin image pins the official `fabiotosi92/ZipDepth` source and its
released GPU/server checkpoint:

- source revision: `94da7527f7030a0e79d54f33b113bdce4065d735`
- checkpoint: `checkpoints/zipdepth_base.pth`

RPX uses one native PyTorch forward per requested batch rather than repeatedly
calling ZipDepth's single-image convenience method. Frames retain the official
aspect-ratio-preserving resize, BGR-to-RGB conversion, normalization and
bilinear restoration. Compressed prediction writes are parallelized within
each batch. Start production at batch size 8 and increase only after checking
available GPU memory on the target system.
- SHA-256: `a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65`

ZipDepth emits affine-invariant inverse depth. The RPX adapter preserves that
native output and declares `ls_disparity`. RPX then performs one pooled
scale-and-shift fit to GT disparity per scene-phase cell and converts the
aligned result to metric depth, matching the official evaluation domain
without per-frame alignment. Legitimate zeros from the released ReLU head are
excluded from the fit, matching the authors' evaluator.

Build by passing the current local paper/FE2E image as `BASE_IMAGE`, run the
`micro` then `acceptance` smoke gates, and only then start the three resumable
production splits. See the server commands in the corresponding handoff.
