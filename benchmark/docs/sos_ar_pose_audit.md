# SOS AR-board camera-pose audit

This audit checks the synchronized RPX single-object camera poses against the
18-marker board visible in the D435 RGB frames. It follows FewSOL's published
board geometry and marker-center approach, but estimates one board pose from all
detected marker corners in each frame.

The protocol assumes, as documented by the RPX dataset owners, that the D435
RGB camera frame and saved T265 camera pose frame are already aligned. It does
not estimate or apply a D435-to-T265 transform. The unrelated world origins are
removed by comparing motion relative to the first detected frame.

The capture script stores raw librealsense T265 axes (X right, Y up, Z back).
The default `--pose-axis-convention t265` applies the fixed OpenCV basis change
to X right, Y down, Z forward. This is a coordinate convention conversion, not
an estimated extrinsic. Use `--pose-axis-convention opencv` only if the pose
files were converted after capture.

## Install

```bash
cd benchmark
python -m pip install -e '.[ar-pose-audit]'
```

## Run every SOS frame

```bash
python scripts/audit_sos_ar_pose.py \
  --dataset-root /path/to/RPX \
  --output-dir /path/to/ar-pose-audit \
  --workers 8
```

The defaults use FewSOL's published D435 matrix. Override `--fx`, `--fy`,
`--cx`, `--cy`, and `--distortion` if the exact RPX calibration is recovered.

Outputs are:

- `report.md`: concise protocol and aggregate findings;
- `summary.json`: machine-readable aggregate and per-sequence metrics;
- `per_sequence.csv`: object-wise coverage, correlation, drift, and error;
- `per_frame.csv`: detections, marker IDs, reprojection quality, and pose error.

Every frame is inspected. Primary metrics default to the robust subset with at
least two RANSAC-consistent tags and at most 2 px reprojection RMSE. Raw
single-tag detections remain visible through `raw_detected` in `per_frame.csv`;
use `--min-markers 1` only for a deliberate FewSOL-style permissive analysis.

Interpret correlation together with metric SE(3) errors. High correlation can
coexist with scale or constant-bias errors. Low marker coverage or high
reprojection RMSE indicates the AR estimate, rather than the saved camera pose,
may be the limiting measurement.
