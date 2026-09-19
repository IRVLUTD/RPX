# RPX SOS AR-board / camera-pose audit

## Protocol

- Dataset: `IRVLUTD/RPX` at revision `2e2a387f7f93e98c177b2e039c141eacda94e5fc`
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
| Gap-normalized step translation RMSE | 0.1144 m/frame |
| Gap-normalized step rotation RMSE | 7.8625 deg/frame |
| Static-board translation drift RMSE | 0.0933 m |
| Static-board rotation drift RMSE | 10.1193 deg |
| Static-board translation jitter, median / P95 | 0.0468 / 0.1181 m |
| Static-board rotation jitter, median / P95 | 1.5260 / 5.8861 deg |
| Robust-inlier translation jitter, median / P95 | 0.0443 / 0.0880 m |
| Robust-inlier rotation jitter, median / P95 | 1.4422 / 3.4654 deg |
| Translation drift slope | 0.000230 m/frame |
| Rotation drift slope | 0.016576 deg/frame |
| Robust residual outlier rate | 0.0832 |
| Mean marker count | 3.14 |
| Mean marker reprojection RMSE | 0.8567 px |

## Automated finding

The AR-derived and stored motion tracks are strongly correlated (translation
`0.9792`, rotation
`0.9724`). Across accepted frames,
the fixed-board residual has median/P95 translation
`0.0468 / 0.1181 m`
and median/P95 rotation
`1.5260 / 5.8861 deg`.
After removing the automatically identified median/MAD residual outliers, those P95 values are
`0.0880 m` and
`3.4654 deg`.

These values show measurable disagreement, but they are an upper bound on T265 pose error rather
than a pure T265 accuracy measurement. The released files do not contain the capture device's
factory distortion coefficients or a numerical D435-to-T265 extrinsic. The residual therefore also
contains tag-corner/PnP uncertainty, approximate-intrinsics error, synchronization error and any
unmodelled sensor-frame offset. Large orientation branches in the per-sequence PDF should be
treated as planar-PnP/reference failures until independently verified.

## Interpretation

Relative-motion metrics remove only the arbitrary world origin; they do not learn or apply a
D435-to-T265 transform. The static-board residual is an independent consistency check: with
correct synchronization, conventions and the stated camera-frame alignment, the composed board
pose should be constant. Review per-sequence and per-frame files before drawing conclusions from
the aggregate correlations, especially where marker coverage or reprojection quality is low.

The first-frame drift metrics are directly comparable to anchor-based ATE. The central-pose jitter
metrics use the sequence median position and rotation medoid, making them less sensitive to one
bad anchor. Gap-normalized step errors are the RPE-style local-motion measurements. A robust
outlier is a translation or rotation residual more than 3.5 scaled MAD above its sequence median.
