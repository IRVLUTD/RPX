# Camera pose convention

One JSON file per frame, named `00000.json` … `<N-1>.json` (5-digit
zero-padded). Frame indices align with `rgb/`, `depth/`, etc.

Each file holds the **camera-to-world** pose for that frame, in the T265
SLAM frame (right-handed, +X right, +Y up, +Z back, units in metres).

```json
{
  "frame":      42,
  "timestamp":  1681423927.123456,
  "pose":       [[r00, r01, r02, tx],
                 [r10, r11, r12, ty],
                 [r20, r21, r22, tz],
                 [   0,    0,    0,  1]],
  "covariance": [...optional 6×6 row-major...],
  "tracker_state": "high_confidence" | "low_confidence" | "lost"
}
```

`pose` is a 4×4 row-major homogeneous transform. `tracker_state` mirrors
the T265's own confidence flag — frames with `lost` should typically be
filtered out for pose-estimation tasks.
