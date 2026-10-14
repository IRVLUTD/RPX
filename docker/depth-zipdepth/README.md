# ZipDepth overlay

A thin image on top of the cumulative depth image that adds the official `fabiotosi92/ZipDepth` source and its
released GPU checkpoint:

- source revision: `94da7527f7030a0e79d54f33b113bdce4065d735`
- checkpoint: `checkpoints/zipdepth_base.pth`, SHA-256
  `a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65`

The fused checkpoint is not numerically batch-invariant, so RPX keeps ZipDepth's single-image inference path. Batching
only groups frames for parallel prediction writes and does not change outputs. Several split workers can share one
large GPU because they write to separate directories.

ZipDepth predicts affine-invariant inverse depth. The adapter returns it unchanged and declares `ls_disparity`, so RPX
fits one scale and shift to GT disparity per scene-phase cell and converts the result to metric depth. This matches
the official evaluation domain without per-frame alignment. Zeros from the ReLU head are excluded from the fit, as in
the authors' evaluator.

To build, pass the FE2E image as `BASE_IMAGE`. Run the `micro` and then `acceptance` gates from the
[depth guide](../depth-smoke/README.md) before starting the three resumable splits.
