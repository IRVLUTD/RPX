# RPX Team Benchmarking Guide

How to run the RPX benchmark on the shared lab GPU server, add a new
model adapter, and ship results to the team Box folder. Covers both
**D1-F** (frame-level depth) and **D1-V** (video-level depth) end to
end.

## TL;DR

```bash
# 1. SSH into the lab GPU box and clone / pull
cd ~/code/RPX/benchmark
git pull origin main

# 2. Activate the project env (assumes you ran `pip install -e .` once)
conda activate rpx  # or source venv/bin/activate

# 3. Run a D1-F model on the easy split
PYTHONPATH=. python scripts/run_depth.py \
    --model da-v2-large --split easy

# 4. Run a D1-V model on the easy split
#    `da-v2-video` is the first real adapter — DA-V2 Large wrapped per-clip.
#    Use it as your smoke test before adding new D1-V adapters.
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da-v2-video --split easy

# 5. Same, with Box upload (requires BOX_DEVELOPER_TOKEN in env)
PYTHONPATH=. python scripts/run_depth.py \
    --model da-v2-large --split easy --upload-to-box
```

Outputs land in `./rpx_results/<model>/<split>/`:
- `result.json` — aggregated metrics + per-axis report
- `cells.parquet` — one row per (scene, phase) — the canonical artefact every downstream plot reads from
- `summary.md` — quick human-readable snapshot

---

## What's in the box today

| Task | Models implemented | Runner | Status |
| --- | --- | --- | --- |
| **D1-F** (frame depth) | 11 adapters under `scripts/depth_models/` (DA-V2, Depth Pro, UniDepth V2, Metric3D V2, MoGe, Marigold, Lotus, ZoeDepth, HyDen, Geowizard, PatchFusion) | `scripts/run_depth.py` | Works today |
| **D1-V** (video depth) | First real adapter `da-v2-video` (DA-V2 Large per-clip baseline) under `scripts/video_depth_models/`; 9 others (DepthCrafter, MonST3R, RollingDepth, ChronoDepth, VGGT-Ω, D4RT, ViGeo, GemDepth, Video DA) still need real forward calls | `scripts/run_video_depth.py` | First model runs end-to-end; team adds the remaining 9 per the recipe below |

The full canonical roster (10 D1-F + 10 D1-V models, 19 unique) is in
`rpx_benchmark/adapters/depth_scaffold.py:DEPTH_MODEL_CARDS`. Tests in
`tests/test_depth_adapter_scaffolds.py` enforce that every roster entry
has a registered skeleton class.

## The reference D1-V adapter pattern: `da-v2-video`

Before you write a new adapter, look at `scripts/video_depth_models/da_v2_video.py` — it's ~20 lines and shows the two pieces every adapter needs:

1. A **per-clip model class** (here, `FrameDepthAsVideo` wrapping the per-frame DA-V2 adapter). Either subclass `BenchmarkModel` directly (for true video models with temporal state) or reuse `FrameDepthAsVideo` (for any per-frame adapter you want as a baseline).
2. A **`build(device)` factory** that the runner discovers by name.

The reusable `FrameDepthAsVideo` wrapper turns *any* `scripts/depth_models/*.py` adapter into a D1-V model in one line. That gives you a baseline row per existing D1-F model with zero new code — useful for paper Table 4's "what does temporal context buy?" diagnostic.

## Adding a new TRUE video model adapter — DepthCrafter example

Three small files, ~30 minutes of work once you have the model package
installed.

### Step 1 — implement the forward call

Create `scripts/video_depth_models/da3.py` (mirror of
`scripts/depth_models/`):

```python
"""DA3 (Depth Anything 3) video adapter.

Install: pip install depth-anything-3
"""
from __future__ import annotations

import numpy as np
import torch

from rpx_benchmark.api import (
    BenchmarkModel,
    TaskType,
    VideoDepthPrediction,
)


class DA3Video(BenchmarkModel):
    task = TaskType.VIDEO_DEPTH
    name = "DA3-video"
    depth_output_kind = "metric"  # DA3 outputs metres directly

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None  # lazy-load in setup()

    def setup(self) -> None:
        from depth_anything_3 import DepthAnything3
        self._model = DepthAnything3.from_pretrained(
            "depth-anything/Depth-Anything-3-Large",
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, batch):
        out = []
        for sample in batch:  # one VideoSample per iteration
            rgb_seq = torch.from_numpy(sample.rgb_seq).to(self.device)
            # rgb_seq: (T, H, W, 3) uint8 → (T, 3, H, W) float in [0, 1]
            x = rgb_seq.permute(0, 3, 1, 2).float() / 255.0
            depth = self._model(x)  # (T, H, W) float, metres
            out.append(VideoDepthPrediction(
                depth_map_seq=depth.cpu().numpy().astype(np.float32),
            ))
        return out


def build(device: str = "cuda") -> BenchmarkModel:
    """Factory the runner looks for."""
    return DA3Video(device=device)
```

### Step 2 — verify it loads

```bash
PYTHONPATH=. python -c "
from video_depth_models.da3 import build
m = build('cuda')
m.setup()
print('loaded', m.name)
"
```

### Step 3 — run on one scene

```bash
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da3 --split easy
```

### Step 4 — relative-depth models need one extra line

If your model is **affine-invariant** (DepthCrafter, RollingDepth,
ChronoDepth, etc.), set the class attribute:

```python
class DepthCrafterVideo(BenchmarkModel):
    task = TaskType.VIDEO_DEPTH
    name = "DepthCrafter"
    depth_output_kind = "relative"  # <-- pipeline applies per-clip (s, t) alignment
```

The runner will solve a single `(s, t)` over each (scene, phase) clip
and apply it before metric computation. You don't have to do anything
else — the alignment math is in
`rpx_benchmark/metrics/depth_alignment.py:align_pred_to_gt_pooled` and
the per-clip alignment is the community convention (DepthCrafter,
Video Depth Anything).

---

## Running on a locally-staged lossless (v2-webp) dataset

The team will upload the v2-webp version to HF when ready. In the
meantime, you can benchmark on a locally-staged tree:

```bash
# 1. Generate the lossless v2-webp tree from the raw capture root
#    (you only do this once on the data machine)
python -m rpx_benchmark.dataset_hub.cli lossless-convert \
    --input /path/to/raw-capture/ \
    --output /path/to/v2-webp/

# 2. Build per-task manifests for the local tree
python -m rpx_benchmark.dataset_hub.cli manifest \
    --src /path/to/raw-capture/ \
    --staging /path/to/v2-webp/ \
    --splits benchmark/data/splits/scene_splits.json

# 3. Run a model against the local manifest (no HF download needed)
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da3 --split easy \
    --manifest-path /path/to/v2-webp/manifests/video_depth/easy.json
```

Both `run_depth.py` and `run_video_depth.py` accept `--manifest-path`.
When set, the pipeline skips `download_split` and loads from disk
directly.

The decode contracts in `rpx_benchmark/decode_contracts.py` are
format-agnostic — WebP RGB, PNG-9 depth, `.npy` cam-pose, and the
legacy PNG / `.npz` forms all decode bit-identically. You don't need
to change anything in adapters or metrics.

---

## Temporal-resolution ablation (D1-V only)

The paper §5.2 ablation sweeps the number of frames per phase clip.
The `frame_budget` + `sampling` knobs are wired through the runner:

```bash
# Stride-uniform subsample of 75 frames per phase
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da3 --split easy \
    --frame-budget 75 --sampling stride

# Farthest-point sampling in T265 trajectory space (recommended)
PYTHONPATH=. python scripts/run_video_depth.py \
    --model da3 --split easy \
    --frame-budget 75 --sampling fps_se3
```

The cell log records the budget per cell, so downstream J / Φ
computation can stratify by budget without extra bookkeeping.

---

## Shared GPU server etiquette

The lab box has one consumer GPU (RTX 5070, 8 GB VRAM). Two rules:

1. **Use `nvidia-smi` before launching** to check no one else is
   running. If someone is, wait or pick a CPU-friendly model
   (impossible for most depth models; just wait).
2. **Cache HF weights once in a shared dir** so others don't re-download:
   ```bash
   export HF_HOME=/data/shared/hf-cache
   ```
   Add that to your `~/.bashrc` on the lab box. The first run for any
   model downloads weights; subsequent runs across the team share them.

If a model OOMs at the default batch size:

```bash
# D1-F: drop batch size
PYTHONPATH=. python scripts/run_depth.py --model X --split easy --batch-size 1

# D1-V: clips are always batch=1, but try smaller --frame-budget
PYTHONPATH=. python scripts/run_video_depth.py --model X --split easy \
    --frame-budget 75 --sampling stride
```

---

## Where the outputs go and what to do with them

Each run writes:

| File | What it is | Who reads it |
| --- | --- | --- |
| `cells.parquet` | One row per (model, task, scene, phase). Carries every metric we computed. **This is the canonical artefact** — downstream J / Φ / paper-table fills all read from here. | `scripts/fill_paper_table.py`, `scripts/analyze_experiment.py`, the paper |
| `result.json` | Aggregated metrics (means / stds across phases) + deployment-readiness report. Convenient single-file summary. | Reviewers, quick eyeballing |
| `summary.md` | Two-paragraph markdown of the result. | Slack / Box previews |
| `predictions/` (D1-F with `--save-predictions`) | Per-frame `.npz` depth maps. Off by default — opt in only when you need post-hoc analytics. | Analytics, qualitative figures |

### Upload to Box

```bash
export BOX_DEVELOPER_TOKEN=<your-token-from-utd-box>
PYTHONPATH=. python scripts/run_depth.py --model X --split easy --upload-to-box
```

Outputs land under `<box_folder_id>/<task>/<model>/<split>/` (the
team's `RPX-Outputs` folder by default, id `380510613151`). Re-uploads
are size-matched and skip identical files, so re-running is cheap.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ConfigError: manifest_path does not exist` | Path typo or missing `dataset_hub.cli manifest` step | Re-run the manifest command from the "lossless" section above |
| `ImportError: No module named depth_anything_3` (or any model package) | Adapter dependency not installed in this env | `pip install depth-anything-3` (check the adapter's docstring for the exact package) |
| `RuntimeError: CUDA out of memory` | Model + batch size > VRAM | Drop `--batch-size` (D1-F) or `--frame-budget` (D1-V) |
| `Box upload failed (401)` | Token expired | Mint a new token in UTD Box, re-export `BOX_DEVELOPER_TOKEN`, re-run `scripts/sync_results_to_box.py` to backfill |
| `D1-V cells.parquet missing some scenes` | Scene's manifest didn't list `pose_filenames` and `sampling="fps_se3"` was used | Switch to `--sampling stride` or fix the manifest |

---

## Reference: the runner data flow

```
HF cache (or local v2-webp tree)
        ↓ download_split / --manifest-path
manifest.json (per-task, per-split)
        ↓ RPXDataset / D1VDataset
batched samples
        ↓ model.predict
predictions
        ↓ per-clip (s, t) alignment if depth_output_kind == "relative"
aligned predictions
        ↓ MetricSuite.for_task
per-sample metric dict
        ↓ cells_from_per_sample
cells.parquet  ←  this is the artefact the paper reads from
        ↓ phi.py / jedi.py / fill_paper_table.py
Tables 3 & 4 in neurips.root.pdf
```

See `rpx_benchmark/tasks/_pipeline.py` (D1-F) and
`rpx_benchmark/tasks/_video_pipeline.py` (D1-V) for the runner
implementations.
