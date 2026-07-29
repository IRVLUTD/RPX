# Official ZipDepth RPX overlay

This thin image pins the official `fabiotosi92/ZipDepth` source and its
released GPU/server checkpoint:

- source revision: `94da7527f7030a0e79d54f33b113bdce4065d735`
- checkpoint: `checkpoints/zipdepth_base.pth`
- SHA-256: `a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65`

ZipDepth emits affine-invariant inverse depth. The RPX adapter converts that
output to its reciprocal depth-domain representation and declares
`ls_disparity`. RPX then performs one pooled disparity scale-and-shift fit per
scene-phase cell, matching the official evaluation domain without per-frame
alignment.

Build by passing the current local paper/FE2E image as `BASE_IMAGE`, run the
`micro` then `acceptance` smoke gates, and only then start the three resumable
production splits. See the server commands in the corresponding handoff.
