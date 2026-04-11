# Quickstart

This page gets you from a blank environment to a written benchmark
report in about 60 seconds.

## 1. Install

```bash
pip install 'rpx-benchmark[depth]'
```

See [Installation](installation.md) for the full extras matrix.

## 2. List what's available

```bash
rpx ls        # tasks + ESD splits + required modalities
rpx models    # runnable adapter names (also shows deferred stubs)
```

Example output of `rpx models`:

```text
Runnable models:
  depth_anything_v2_metric_indoor_base
  depth_anything_v2_metric_indoor_large
  depth_anything_v2_metric_indoor_small
  depth_pro
  metric3d_v2_vit_giant2
  metric3d_v2_vit_large
  metric3d_v2_vit_small
  unidepth_v2_vitb
  unidepth_v2_vitl
  zoedepth_nyu

Deferred (registered for visibility; raise on resolve):
  depth_anything_3
  prompt_depth_anything_vits
  video_depth_anything_large
```

## 3. Run a registered model on the Hard split

```bash
rpx bench monocular_depth --model depth_pro --split hard
```

## 4. Or run ANY HuggingFace depth checkpoint — zero code

```bash
rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard
```

Works with any model loadable via
`transformers.AutoModelForDepthEstimation`.

## 5. Run segmentation

```bash
rpx bench object_segmentation \
    --hf-checkpoint facebook/mask2former-swin-tiny-coco-instance \
    --split hard
```

## 6. Read the output

Every run writes two files under
`./rpx_results/<model>/<split>/`:

- **`result.json`** — machine-readable. Contains:
    - `aggregated`: metrics averaged over the split.
    - `per_sample`: one row per sample with metric values **and**
      `id` / `phase` / `difficulty` metadata, so downstream analysis
      can group back to scenes and phases without re-reading the
      manifest.
    - `deployment_readiness`: Weighted Phase Score, State-Transition
      Robustness, Temporal Stability, FLOPs, median latency,
      parameter count.
- **`summary.md`** — human-readable. Rendered with the same tables
  the CLI prints.

## 7. Live terminal output

If `rich` is installed (pulled in by `[depth]`), the CLI renders
Claude-Code-style panels:

- A header panel with model / split / device
- Live progress bar during inference
- Aggregated metric table
- ESD-weighted phase score table with coloured Δ_int / Δ_rec
- Efficiency table (params / FLOPs / latency)
- Footer pointing at the output files

Pass `--plain` to disable the rich UI for CI logs / plain ssh / tmux
weirdness.

## 8. Global flags

```bash
rpx --verbose <command>   # DEBUG-level logging
rpx --quiet <command>     # WARNING-level logging only
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | `RPXError` (config / dataset / model / metric / download failure) |
| `2` | CLI argument error (handled by argparse) |
| `130` | KeyboardInterrupt |

All errors subclass `rpx.RPXError` and carry a `hint` line telling
the user exactly what to fix.
