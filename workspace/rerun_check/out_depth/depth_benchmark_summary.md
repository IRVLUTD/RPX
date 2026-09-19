# RPX Depth Benchmark QA Summary

- Status: `ok`
- Units checked: `371`
- Frames sampled: `1113`
- Errors: `0`
- Warnings: `0`
- Ground-truth depth source: `uint16 millimeters`
- Benchmark depth unit: `meters`
- Invalid/hole values: `0` and `65535`

## Depth Quality

- `zero_fraction` min/mean/max: `0` / `0.0260455` / `0.24626`
- `saturated_fraction` min/mean/max: `0` / `0.00286753` / `0.138415`
- `valid_fraction` min/mean/max: `0.75374` / `0.971087` / `1`
- `valid_min_m` min/mean/max: `0.155` / `0.864058` / `1.736`
- `valid_max_m` min/mean/max: `0.557` / `25.6663` / `61.977`
- `valid_mean_m` min/mean/max: `0.420815` / `2.67552` / `13.5238`
- `valid_std_m` min/mean/max: `0.0455113` / `2.40191` / `14.8789`
- `valid_p01_m` min/mean/max: `0.292` / `0.944449` / `1.891`
- `valid_p50_m` min/mean/max: `0.412` / `1.80439` / `12.912`
- `valid_p99_m` min/mean/max: `0.547` / `11.8391` / `61.977`

## Identity Metric Smoke

- `abs_rel` min/mean/max: `0` / `0` / `0`
- `sq_rel` min/mean/max: `0` / `0` / `0`
- `rmse` min/mean/max: `0` / `0` / `0`
- `rmse_log` min/mean/max: `0` / `0` / `0`
- `mae` min/mean/max: `0` / `0` / `0`
- `log10` min/mean/max: `0` / `0` / `0`
- `silog` min/mean/max: `0` / `0` / `0`
- `delta1` min/mean/max: `1` / `1` / `1`
- `delta2` min/mean/max: `1` / `1` / `1`
- `delta3` min/mean/max: `1` / `1` / `1`
