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
- `overview.pdf` / `overview.png`: aggregate coverage and jitter diagnostics;
- `per_sequence_diagnostics.pdf`: one time-series page per SOS object.

Generate the plots after the audit with:

```bash
python scripts/plot_sos_ar_pose_audit.py \
  --audit-dir /path/to/ar-pose-audit
```

Every frame is inspected. Primary metrics default to the robust subset with at
least two RANSAC-consistent tags and at most 2 px reprojection RMSE. Raw
single-tag detections remain visible through `raw_detected` in `per_frame.csv`;
use `--min-markers 1` only for a deliberate FewSOL-style permissive analysis.

Interpret correlation together with metric SE(3) errors. High correlation can
coexist with scale or constant-bias errors. Low marker coverage or high
reprojection RMSE indicates the AR estimate, rather than the saved camera pose,
may be the limiting measurement.

## Reporting rationale

The primary automated outputs follow the trajectory-evaluation terminology used
by the TUM RGB-D benchmark:

- anchor-relative translation/rotation error is an ATE-style global consistency
  diagnostic, without fitting a sensor transform;
- gap-normalized step translation/rotation error is the local, RPE-style drift
  diagnostic;
- static-board translation and rotation dispersion report median, RMSE,
  standard deviation, P95 and maximum residuals around a central board pose;
- coverage, marker count and reprojection RMSE expose failure of the fiducial
  reference itself rather than silently attributing it to the T265 poses;
- a median/MAD outlier rate gives an automated, distribution-robust failure
  count.

References:

- Sturm, Burgard and Cremers, *Evaluating Egomotion and Structure-from-Motion
  Approaches Using the TUM RGB-D Benchmark* (ATE/RPE definitions):
  https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros_ws.pdf
- FewSOL toolkit, *Regarding Object Poses in the Real Object Data* (RANSAC over
  AR tags and marker-board centre):
  https://github.com/IRVLUTD/fewsol-toolkit#regarding-object-poses-in-the-real-object-data
- Kalaitzakis et al., *Fiducial Markers for Pose Estimation* (accuracy,
  detection rate and computational-cost reporting across marker systems):
  https://doi.org/10.1007/s10846-020-01307-9
- EuRoC MAV dataset (explicit camera intrinsics/extrinsics, synchronization and
  accurate external pose ground truth as benchmark requirements):
  https://projects.asl.ethz.ch/datasets/euroc-mav/

The central-pose jitter is the most direct answer to "how stationary does the
fixed board remain?". It is not pure T265 error: it also contains tag-corner/PnP
noise, calibration error, synchronization error, and any unmodelled D435/T265
extrinsic. Therefore the report must keep reprojection and coverage beside the
pose residuals and must not call the AR result exact ground truth.

The RPX Hugging Face release supplies the RGB and camera-pose shards used here,
but it does not expose the capture unit's factory distortion coefficients or a
numeric D435-to-T265 extrinsic. The default matrix is consequently the
paper-declared 640×480 D435 calibration. If the retained device calibration is
recovered, rerun with its exact intrinsics/distortion; if the rigid inter-camera
transform is recovered, add it explicitly rather than fitting it on the same
frames being evaluated.
