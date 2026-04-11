# RPX Benchmark Toolkit

> **Bring your model. We bring the dataset, the splits, the metrics, and the tables.**
>
> **RPX enables you to choose and rank perception models for robot learning** — on real-world RGB-D data, under embodied deployment conditions, with ESD-stratified difficulty splits and deployment-readiness scoring.

`rpx-benchmark` is the reference toolkit for [RPX — Robot Perception
X](https://github.com/IRVLUTD/RPX), a unified real-world RGB-D
benchmark for evaluating perception models actually deployed inside
robot learning stacks (not generic perception leaderboards). It's
built so a researcher can run an off-the-shelf HuggingFace model on
an RPX difficulty split in **one command**, compare results across
the slate of robot-learning backbones, and a team can add a whole
new task or metric in **one file**.

```bash
pip install 'rpx-benchmark[depth]'

rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard
```

That line will: download only the RGB + depth files the Hard split
references (via HuggingFace), load the model, run inference with a
live progress bar, print an ESD-weighted phase-score table, measure
FLOPs and median latency, and write `result.json` + `summary.md`.

No Python code required.

---

## Table of contents

- [What RPX is](#what-rpx-is)
- [60-second quickstart](#60-second-quickstart)
- [Installation](#installation)
- [Bring your own model — three paths](#bring-your-own-model--three-paths)
- [Available tasks](#available-tasks)
- [Available models](#available-models)
- [Dataset access](#dataset-access)
- [What a run produces](#what-a-run-produces)
- [Extending the toolkit](#extending-the-toolkit)
- [CLI reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

---

## What RPX is

- **75,000 frames** across 100 indoor scenes, captured with an Intel
  RealSense D435 (RGB-D) + T265 (6-DoF VIO) rig.
- **Three-phase capture protocol**: each scene is recorded under
  *Clutter* → *Interaction* (human grasps/moves objects) → *Clean*.
  This isolates scene reconfiguration from scene identity so
  performance deltas mean something.
- **Effort-Stratified Difficulty (ESD) splits** per `(scene, phase)` —
  Easy / Medium / Hard derived from real annotation effort.
- **Ten benchmark tasks** evaluated on identical scenes: monocular
  absolute depth, object segmentation, object tracking, object
  detection, open-vocab detection, visual grounding, sparse depth,
  relative camera pose, novel view synthesis, keypoint matching.
- **Scoped first-class around models used as backbones in robot
  learning**, not generic perception SOTA. See the paper for the full
  model slate rationale.

---

## 60-second quickstart

```bash
# 1. Install (core + dataset download + depth slate + pretty terminal)
pip install 'rpx-benchmark[depth]'

# 2. See what is available
rpx ls                           # tasks + ESD splits
rpx models                       # runnable + deferred adapters

# 3. Run any registered model on the Hard split
rpx bench monocular_depth --model depth_pro --split hard

# 4. Or point at ANY HuggingFace depth checkpoint — no code needed
rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard

# 5. Run segmentation the same way
rpx bench object_segmentation \
    --hf-checkpoint facebook/mask2former-swin-tiny-coco-instance \
    --split hard
```

Output lives in `./rpx_results/<model>/<split>/{result.json, summary.md}`.

---

## Installation

### Stable

```bash
pip install 'rpx-benchmark[depth]'
```

### From source

```bash
git clone https://github.com/IRVLUTD/RPX.git
cd RPX/benchmark
pip install -e '.[depth,dev,docs]'
```

### Optional extras matrix

| Extra | What it installs | When to use |
|---|---|---|
| *(core)* | `numpy`, `Pillow` | Always — manifest loading, metric computation |
| `hub` | `huggingface_hub[hf_xet]` | Pulling datasets from HuggingFace |
| `ui` | `rich` | Pretty terminal output (progress bar, tables) |
| `depth-hf` | `torch`, `torchvision`, `transformers`, `accelerate` | Any HuggingFace depth model (DA-v2, Depth Pro, ZoeDepth, PromptDA) |
| `depth-unidepth` | `depth-hf` + `einops`, `timm`, `huggingface_hub` | UniDepth V2 |
| `depth-metric3d` | `depth-hf` + `timm`, `mmcv-lite`, `mmengine` | Metric3D V2 (CUDA only) |
| `depth` | `depth-hf` + `ui` | **Recommended default** for monocular depth benchmarking |
| `depth-all` | `depth-hf` + `ui` + unidepth + metric3d | Everything for monocular depth |
| `dev` | `pytest`, `ruff` | Contributing, running the test suite |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building documentation locally |

UniDepth V2 also needs a manual pip from GitHub (not on PyPI):

```bash
pip install 'unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git'
```

---

## Bring your own model — three paths

The toolkit is built around **two adapters and a model**:

```
  Sample ───► InputAdapter.prepare ───► PreparedInput(payload, context)
                                                │
                                                ▼
                                          model(payload)
                                                │
                                                ▼
  Sample, context, model_output ───► OutputAdapter.finalize ───► Prediction
```

Users touch **only the model**. Input and output adapters are already
shipped for the two common families (HuggingFace transformers, raw
numpy callables). Three fast paths:

### Path 1 — Any HuggingFace depth checkpoint (zero code)

```bash
rpx bench monocular_depth --hf-checkpoint my-org/my-depth-model --split hard
```

Works with any checkpoint loadable via
`transformers.AutoModelForDepthEstimation` — Depth Anything V2
(metric), Depth Pro, ZoeDepth, PromptDA, Video Depth Anything
(per-frame).

### Path 2 — Plain numpy callable

```python
import numpy as np
import rpx_benchmark as rpx

def my_depth(rgb: np.ndarray) -> np.ndarray:
    """rgb: H x W x 3 uint8 → returns H x W float32 in metres."""
    ...
    return depth_map

bm = rpx.make_numpy_depth_model(my_depth, name="my_numpy_depth")

cfg = rpx.MonocularDepthRunConfig(model=bm, split="hard", device="cpu")
result, report, paths = rpx.run_monocular_depth(cfg)

print(result.aggregated)                        # absrel, rmse, delta1..3
print(report.weighted_phase_score.to_dict())    # phase-stratified scores
```

Segmentation has the symmetric helper `rpx.make_numpy_mask_model(fn)`
where `fn(rgb) → int32 mask`.

### Path 3 — Custom torch / transformers stack

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import rpx_benchmark as rpx
from rpx_benchmark.adapters.depth_hf import (
    HFDepthInputAdapter, HFDepthOutputAdapter,
)

processor = AutoImageProcessor.from_pretrained("my-org/my-model")
model = (
    AutoModelForDepthEstimation
    .from_pretrained("my-org/my-model")
    .to("cuda").eval()
)

bm = rpx.BenchmarkableModel(
    task=rpx.TaskType.MONOCULAR_DEPTH,
    input_adapter=HFDepthInputAdapter(processor=processor, device="cuda"),
    model=model,
    output_adapter=HFDepthOutputAdapter(processor=processor),
    name="my_model",
)

rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(model=bm, split="hard"))
```

For non-HuggingFace model families, clone `adapters/depth_unidepth.py`
(UniDepth pattern) or `adapters/depth_metric3d.py` (torch.hub +
letterbox pattern).

---

## Available tasks

Every task ships the full stack: **dataset + split slicing, metrics,
loaders, numpy adapter, CLI subcommand**. The only thing you bring
is the model.

| Task | Primary metric | Modalities | BYO-model fast path | Status |
|---|---|---|---|---|
| `monocular_depth` | AbsRel (↓) | rgb, depth | `make_numpy_depth_model` + `make_hf_depth_model` | ✅ Runnable |
| `object_segmentation` | mIoU (↑) | rgb, mask | `make_numpy_mask_model` + `make_hf_instance_seg_model` | ✅ Runnable |
| `object_detection` | F1 (↑) | rgb, boxes | `make_numpy_detection_model` | ✅ Runnable |
| `open_vocab_detection` | F1 (↑) | rgb, boxes, questionnaires | `make_numpy_detection_model(..., task=OPEN_VOCAB_DETECTION)` | ✅ Runnable |
| `visual_grounding` | grounding_acc (↑) | rgb, spatial_qa | `make_numpy_grounding_model` | ✅ Runnable |
| `relative_camera_pose` | rotation_err (↓) | rgb (pair), pose | `make_numpy_pose_model` | ✅ Runnable |
| `keypoint_matching` | keypoint_acc (↑) | rgb (pair), keypoints | `make_numpy_keypoint_model` | ✅ Runnable |
| `sparse_depth` | sparse_absrel (↓) | rgb, depth, sparse samples | `make_numpy_sparse_depth_model` | ✅ Runnable |
| `novel_view_synthesis` | psnr (↑) | rgb (src+tgt), depth, pose | `make_numpy_nvs_model` | ✅ Runnable |
| `object_tracking` | MOTA (↑) | rgb, tracklets | — | ⚠ Deferred (sequence protocol) |

**9 of 10 tasks are runnable end-to-end.** Tracking is deferred
because the sample contract (sequence-per-sample vs per-frame) needs
a protocol decision before a task runner can be cloned. Every other
task has a full CLI subcommand (`rpx bench <task> …`), a numpy
fast-path factory for zero-ceremony BYO-model evaluation, and end-
to-end tests against a synthetic dataset fixture.

Run `rpx ls` for the live list; `rpx bench --help` for all nine
subcommands the CLI auto-generates from the task registry.

### BYO-model callable signatures

| Task | Callable signature | Returns |
|---|---|---|
| `monocular_depth` | `fn(rgb_uint8)` | `(H, W) float` metric depth (metres) |
| `object_segmentation` | `fn(rgb_uint8)` | `(H, W) int` instance mask |
| `object_detection` | `fn(rgb_uint8)` | `{"boxes", "scores", "labels"}` or `(boxes, scores, labels)` |
| `visual_grounding` | `fn(rgb_uint8, text)` | `{"boxes", "scores"}` or `(boxes, scores)` |
| `relative_camera_pose` | `fn(rgb_a, rgb_b)` | `{"rotation", "translation"}` |
| `keypoint_matching` | `fn(rgb_a, rgb_b)` | `(points0, points1[, scores])` |
| `sparse_depth` | `fn(rgb_uint8, coords)` | `(N,) float` depths at the provided coordinates |
| `novel_view_synthesis` | `fn(rgb_src, target_pose_4x4)` | `(H, W, 3) uint8` synthesised RGB |

---

## Available models

First-round slate scoped to robot-learning backbones. Run `rpx models`
to see the live state.

### Monocular absolute depth

| Model | Registered name | CPU | CUDA | Verified live |
|---|---|---|---|---|
| Depth Anything V2 metric indoor (S/B/L) | `depth_anything_v2_metric_indoor_{small,base,large}` | ✅ | ✅ | ✅ (S) |
| Depth Pro (Apple) | `depth_pro` | ✅ | ✅ | via adapter contract |
| ZoeDepth (legacy anchor) | `zoedepth_nyu` | ✅ | ✅ | ✅ |
| UniDepth V2 (ViT-B/L) | `unidepth_v2_{vitb,vitl}` | ✅ | ✅ | ✅ (B) |
| Metric3D V2 (ViT-S/L/G2) | `metric3d_v2_vit_{small,large,giant2}` | ❌ | ✅ | ⏳ (CUDA required) |

**Metric3D V2 is CUDA-only** — its upstream decoder hardcodes
`torch.linspace(..., device="cuda")`, so the adapter raises a clean
error on CPU hosts rather than failing mid-inference.

**Deferred** (registered for visibility, raise on `resolve`): Video
Depth Anything (sequence model, needs temporal benchmark mode),
Prompt Depth Anything (needs sparse-depth prompt, belongs in a
prompted-depth task), Depth Anything 3 (not yet in `transformers`).

### Object segmentation

`make_hf_instance_seg_model("<checkpoint>")` works with any
Mask2Former / OneFormer / MaskFormer / DETR-panoptic / SegFormer
checkpoint whose processor exposes
`post_process_{instance,panoptic,semantic}_segmentation`. The output
adapter introspects the processor signature and forwards only the
kwargs it accepts.

---

## Dataset access

The toolkit pulls dataset slices from HuggingFace on demand. The
downloader is **task-aware**: it only fetches the modalities your
requested task actually needs, and reuses the HF content-addressed
cache so switching tasks on the same scenes costs zero redundant
bytes.

```python
import rpx_benchmark as rpx

# First call: fetches RGB + depth for Hard scenes only.
depth_ds = rpx.load("monocular_depth", "hard")
for batch in depth_ds:
    sample = batch[0]
    print(sample.rgb.shape, sample.ground_truth.depth_map.shape)

# Next call reuses the cached RGB; only mask PNGs are fetched.
seg_ds = rpx.load("object_segmentation", "hard")
```

CLI equivalents:

```bash
rpx info     --task monocular_depth --split hard   # counts + split stats
rpx download --task monocular_depth --split hard   # materialise locally
```

On-disk layout on the hub (single dataset repo):

```
rpx-benchmark/
├── README.md                          # dataset card
├── metadata/
│   ├── scenes.parquet                 # scene_id, env, categories
│   └── esd_scores.parquet             # (scene, phase) → ESD features + difficulty
├── manifests/                         # logical Easy/Medium/Hard views (not duplicated)
│   └── <task>/{easy,medium,hard}.json
└── scenes/scene_000/{0,1,2}/          # 100 scenes, 0=clutter 1=interaction 2=clean
    ├── rgb/*.png                      # 640×480 uint8
    ├── depth/*.png                    # 16-bit mm (D435 raw)
    ├── mask/*.png                     # int32 instance IDs
    ├── pose/*.npz                     # T265 position + quaternion
    ├── tracklets.json                 # per-phase tracks
    ├── questionnaires.json            # FewSOL attributes
    ├── spatial_qa.json                # spatial Q&A
    └── general_qa.json                # general Q&A
```

**Difficulty splits are logical** — a frame used in Hard is the same
byte-for-byte file as in Easy. Nothing is duplicated server-side.

---

## What a run produces

Every `rpx bench <task>` invocation writes two files under
`./rpx_results/<model>/<split>/`:

**`result.json`** — machine-readable. Contains:
- `aggregated`: task metrics averaged over the split
- `per_sample`: one row per sample with metric values **and**
  `id` / `phase` / `difficulty` metadata, so downstream analysis can
  group back to scenes and phases without re-reading the manifest
- `deployment_readiness`: Weighted Phase Score, State-Transition
  Robustness, Temporal Stability, FLOPs, median latency, parameter
  count

**`summary.md`** — human-readable. Rendered with the same tables the
CLI prints.

**Terminal output** (via the `rich` UI when installed) includes:
- A header panel with model / split / device
- Live progress bar during inference
- Aggregated metric table
- ESD-weighted phase score table with coloured Δ_int / Δ_rec
- Efficiency table (params / FLOPs / latency)
- Footer pointing at the output files

Use `--plain` on the `bench` subcommand to disable the rich UI (for
CI logs, plain ssh, etc.).

---

## Extending the toolkit

The toolkit is built around **three plugin registries**: models,
metrics, tasks. Adding a new entry to any of them is a **one-file
change**.

### Add a new metric

```python
from rpx_benchmark.api import TaskType
from rpx_benchmark.metrics import MetricCalculator, register_metric

@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthMedian(MetricCalculator):
    """Median AbsRel — complementary to the mean AbsRel baseline."""
    name = "depth_median"

    def compute(self, prediction, ground_truth):
        import numpy as np
        valid = ground_truth.depth_map > 0
        rel = np.abs(prediction.depth_map - ground_truth.depth_map) / \
              np.maximum(ground_truth.depth_map, 1e-6)
        return {"median_absrel": float(np.median(rel[valid]))}
```

Save to any importable location, import it once at program start, and
every subsequent monocular-depth run includes `median_absrel` in its
output without touching the runner, CLI, or reports.

### Add a new task

Clone `rpx_benchmark/tasks/monocular_depth.py`, swap the task-specific
knobs (`TaskType`, primary metric, required modalities,
`higher_is_better`, any config fields), and register via:

```python
from rpx_benchmark.tasks.registry import TaskSpec, register_task

TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_TRACKING,
    display_name="Object Tracking",
    description="...",
    primary_metric="mota",
    required_modalities=["rgb", "mask", "tracklets"],
    higher_is_better=True,
    build_config=_build_config,       # argparse.Namespace -> Config
    run=run_tracking,                 # Config -> (result, dr, paths)
    add_cli_arguments=_add_cli_args,  # argparse.ArgumentParser -> None
)

register_task(TASK_SPEC)
```

The CLI auto-discovers this at parser-build time and exposes
`rpx bench object_tracking` with your flags. **Zero changes to
`cli.py`.**

### Add a new model adapter

Write a factory that returns a `BenchmarkableModel`:

```python
from rpx_benchmark.adapters import BenchmarkableModel
from rpx_benchmark.adapters.depth_hf import make_hf_depth_model
from rpx_benchmark.models.registry import register

def my_new_depth_factory(*, device="cuda", **kwargs):
    return make_hf_depth_model(
        "my-org/my-checkpoint",
        device=device,
        name="my_new_depth",
        **kwargs,
    )

# One-line registration so `rpx bench --model my_new_depth` works.
register("my_new_depth", "path.to.module", "my_new_depth_factory")
```

For non-HuggingFace families (like UniDepth), write your own
`InputAdapter` + `OutputAdapter` pair and compose them in the
factory. See `rpx_benchmark/adapters/depth_unidepth.py` for a full
example.

---

## CLI reference

```text
rpx [--verbose/--quiet] <command> [flags]

Commands
--------
  ls                                 List tasks + splits + modalities
  models                             List runnable + deferred model adapters
  info       --task <t> --split <d>  Split stats without downloading frames
  download   --task <t> --split <d>  Pre-fetch files via HuggingFace
  bench      <task> [flags]          End-to-end benchmark

Global flags
------------
  --verbose, -v    DEBUG-level logging
  --quiet, -q      WARNING-level logging only

Exit codes
----------
  0    success
  1    RPXError (config / dataset / model / metric / download failure)
  2    CLI argument error (handled by argparse)
  130  KeyboardInterrupt
```

All errors are subclasses of `rpx.RPXError` and carry a `hint` line
telling the user exactly what to fix.

---

## Terminal banner

Every long-running RPX operation (CLI subcommands and the data-prep
scripts under [`scripts/`](scripts/)) prints a Claude-Code-style
banner at startup with the RPX logo, tagline, version, and links:

```text
╭─ ◆ RPX Benchmark ────────────────────────────────────────────────────────────╮
│                                                                              │
│   ██████╗ ██████╗ ██╗  ██╗                                                   │
│   ██╔══██╗██╔══██╗╚██╗██╔╝                                                   │
│   ██████╔╝██████╔╝ ╚███╔╝                                                    │
│   ██╔══██╗██╔═══╝  ██╔██╗                                                    │
│   ██║  ██║██║     ██╔╝ ██╗                                                   │
│   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝                                                   │
│                                                                              │
│   Robot Perception X  ·  v0.1.0  ·  MIT                                      │
│   Choose and rank perception models for robot learning                       │
│   Bring your model — we bring the dataset, splits, metrics, and tables.      │
│                                                                              │
│   docs    https://irvlutd.github.io/RPX/                                     │
│   source  https://github.com/IRVLUTD/RPX                                     │
│                                                                              │
╰────────────────────────────────────────────── robot perception, benchmarked ─╯
```

Three suppression paths:

1. **`rpx --quiet <command>`** — also sets logging to WARNING level.
2. **`RPX_NO_BANNER=1`** environment variable.
3. **`2>/dev/null`** — the banner goes to `stderr`, so redirecting
   `stderr` is enough for scripting pipelines. `stdout` output
   (listings, metric rows) is never touched.

Call it from your own scripts:

```python
from rpx_benchmark import show_banner

show_banner(
    subtitle="my_script.py",
    context="scene=scene_042 phase=clutter",
)
```

Falls back to a plain-text box when `rich` is not installed, so it
works over bare ssh, tmux, and CI logs without leaking escape
codes.

---

## Troubleshooting

**`CUDA requested but torch.cuda.is_available() is False`**
The CLI auto-falls-back to CPU with a warning. If you want strict
failure, pass `--device cpu` up front.

**`Metric3D V2 requires device='cuda'`**
Upstream hardcodes `device="cuda"` in its decoder. Use a CUDA host or
swap to `depth_anything_v2_metric_indoor_*`.

**`DownloadError: snapshot_download failed`**
Check your network, `HF_TOKEN` (for private repos), and that the
cache directory is writable. Offline mirrors work via
`HF_HUB_OFFLINE=1`.

**`ManifestError: Manifest file not found`**
The HuggingFace pull failed mid-way, or the repo id is wrong. Re-run
with `--verbose` to see the underlying call.

**`ModelError: Task mismatch between model and dataset`**
You built the dataset for one `TaskType` and the model for another.
Use the same `TaskType` on both.

**rich UI looks mangled in `tmux` / SSH**
Pass `--plain` to `rpx bench` or set `TERM=dumb`.

---

## Documentation site

The full API reference is generated automatically from the docstrings
in the source tree — no separate rewrite needed.

```bash
pip install 'rpx-benchmark[docs]'
mkdocs serve                       # live-reload local server on :8000
mkdocs build                       # static site under site/
```

Adding a new public class or function is picked up by the next build
automatically; mkdocstrings reads its numpydoc docstring and renders
it with parameters, returns, raises, and examples.

### Hosted docs on GitHub Pages

A ready-to-use GitHub Actions workflow lives at
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml). It
builds the site on every push to `main` and deploys to GitHub Pages.
To enable it once:

1. Push the repository to GitHub.
2. Go to **Settings → Pages** and set **Source = "GitHub Actions"**.
3. Push to `main` (or run `workflow_dispatch` from the Actions tab).
4. The site goes live at `https://<org>.github.io/RPX/` — the URL is
   pinned in `mkdocs.yml` (`site_url`).

The tests workflow at
[`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs
the 120-test offline suite on Python 3.10 / 3.11 / 3.12 and a ruff
check on every push and pull request.

---

## Contributing

```bash
git clone https://github.com/IRVLUTD/RPX.git
cd RPX/benchmark
pip install -e '.[depth,dev,docs]'
pytest tests/                      # 128 tests in ~1 s
```

### ⚠ Always use an editable install while developing

**The `-e` flag is mandatory** when hacking on the codebase. Without
it the installed package is a frozen snapshot of the source at
install time, and your edits to `rpx_benchmark/` will silently have
no effect on `rpx`, `python -m rpx_benchmark.cli`, or
`pytest`. Confusing symptoms include: new features missing, the
banner not appearing, ImportErrors on symbols you just added, tests
passing locally but CI showing a different set.

If you ever see stale behaviour, re-run the editable install:

```bash
pip install -e . --force-reinstall --no-deps
```

Never `pip install .` (without `-e`) or `pip install <wheel>` from
inside the repo for dev work — use those only for release
verification in a disposable virtualenv.

### Style + discipline

- Follow the docstring convention (`numpydoc` style) so the
  autogenerated site stays consistent.
- Raise `rpx.exceptions.RPXError` subclasses, **never** bare
  `ValueError` / `RuntimeError` / `KeyError`. The
  `test_no_bare_stdlib_raises_in_library` test enforces this.
- New metrics / tasks / models go through their respective registry
  (see `docs/guides/`).
- Every public API change should come with a test in the matching
  `tests/test_<module>.py` file.
- Run `ruff check rpx_benchmark tests scripts` before pushing; CI
  will reject unused imports, undefined names, or syntax errors.

---

## Citation

If you use this toolkit or the RPX dataset in your work, please cite
the accompanying NeurIPS 2026 Datasets and Benchmarks paper. The full
BibTeX entry will be added here once the camera-ready version is
released.

---

## License

- Benchmark toolkit (this repository): **MIT**
- RPX dataset: **CC BY 4.0**
