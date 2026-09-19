# RPX DepthSplat NVS image

This is the first cumulative NVS image. It pins DepthSplat source revision
`2dad25a7b9ba7c5537aba08463ef88d8511a06be` and uses the released small,
two-view, 256x256 Gaussian-splatting checkpoint. The RPX adapter supplies the
published D435 intrinsics, converts raw T265 axes to OpenCV axes, and removes
only the arbitrary trajectory origin.

The gate runner evaluates real RPX multi-object scenes and saves both NPZ
predictions and directly viewable PNG renderings. The SOS runner consumes the
single-object tar archives, evaluates held-out target views, and optionally
compares AR-board detection and pose between each generated view and its real
target image.

Build through `build_depthsplat_rpx.sh`, then run
`benchmark/scripts/run_nvs_gate.py` for `smoke`, `micro`, and `acceptance`.
Use `benchmark/scripts/run_nvs_sos.py` for single-object inference and
`package_nvs_results.sh` to create checksummed SCP archives.

## Single-object protocols

The SOS runner defaults to a deterministic, local forward-extrapolation
benchmark.  Every approximately 500-frame object trial contributes four
windows.  Each window has two context views at offsets `0,49` and three
targets at offsets `55,60,70`, giving 12 samples per object.  This keeps the
target outside the context span while avoiding the extreme `0,199 -> 300+`
baseline used by the far stress test.

Available protocols are:

* `identity`: target equals context 2; adapter/calibration sanity diagnostic.
* `interpolation`: target is the temporal midpoint between the contexts.
* `sliding-near`: four local forward-extrapolation windows (default).
* `far`: original first-40% to final-40% stress test.
* `all`: runs every tier and reports both combined and per-protocol metrics.

Example:

```bash
python benchmark/scripts/run_nvs_sos.py \
  --model depthsplat \
  --dataset-root /cache/huggingface/datasets--IRVLUTD--RPX/snapshots/$RPX_REVISION \
  --output-root /outputs \
  --protocol sliding-near \
  --max-objects 10 \
  --device cuda
```

The local layout can be overridden with `--window-starts`,
`--context-offsets`, and `--target-offsets`.  Every result row records the
exact frames plus the nearest-context translation, rotation, and frame-gap
baseline. Protocols are written under `DepthSplat/sos/<protocol>` so results
cannot be mixed accidentally.
