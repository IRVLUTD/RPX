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
