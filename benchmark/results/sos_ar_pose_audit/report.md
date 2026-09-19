# RPX SOS AR-board / camera-pose audit

## Protocol

- Sequences: 70 (70 with detections)
- Frames inspected: 35000
- Frames with a valid AR-board pose: 11317 (32.3%)
- Frames with any raw tag pose: 16731 (47.8%)
- Camera matrix: `[[611.10888672, 0.0, 315.51083374], [0.0, 610.02844238, 237.73669434], [0.0, 0.0, 1.0]]`
- Sensor-frame assumption: D435 RGB and saved T265 camera poses are already aligned.
- Sensor transform fitting: **none**.
- World-origin handling: relative poses from the first detected frame in each sequence.
- Primary quality gate: at least 2 RANSAC-consistent markers and reprojection RMSE <= 2.0 px.

## Aggregate results

| Measure | Value |
|---|---:|
| Translation correlation, all components | 0.9792 |
| Translation-magnitude correlation | 0.9549 |
| Rotation-vector correlation, all components | 0.9724 |
| Rotation-magnitude correlation | 0.9952 |
| Anchor-relative translation RMSE | 0.1705 m |
| Anchor-relative rotation RMSE | 10.1193 deg |
| Step translation RMSE | 0.1397 m |
| Step rotation RMSE | 9.3656 deg |
| Static-board translation drift RMSE | 0.0933 m |
| Static-board rotation drift RMSE | 10.1193 deg |
| Mean marker count | 3.14 |
| Mean marker reprojection RMSE | 0.8567 px |

Across the 70 per-object results, the median translation correlation was
0.9949 and the median rotation-vector correlation was 0.9933. Median
static-board drift was 0.0803 m and 6.60 degrees. The frame-weighted aggregate
is lower because longer, noisier sequences contribute more measurements.

## Finding

The AR-derived and saved trajectories have strongly correlated shape and
rotation magnitude. They are not metrically interchangeable at the precision
of this audit: the robust subset still shows 0.0933 m / 10.12 degree aggregate
static-board drift. This does not by itself identify T265 pose error. Residuals
also contain ALVAR/PnP noise, the use of FewSOL's published D435 intrinsics
rather than a retained RPX device calibration, synchronization error, and any
unmodelled difference hidden by the statement that the sensor frames were
"aligned". Object-wise results should therefore be used to isolate repeatable
T265 drift from isolated AR-estimation failures.

## Interpretation

Relative-motion metrics remove only the arbitrary world origin; they do not learn or apply a
D435-to-T265 transform. The static-board residual is an independent consistency check: with
correct synchronization, conventions and the stated camera-frame alignment, the composed board
pose should be constant. Review per-sequence and per-frame files before drawing conclusions from
the aggregate correlations, especially where marker coverage or reprojection quality is low.
