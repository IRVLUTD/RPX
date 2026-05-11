<div align="center">

# RPX — Robot Perception X

**Choose and rank perception models for robot learning** — on real-world RGB-D data, under embodied deployment conditions, with ESD-stratified difficulty splits and deployment-readiness scoring.

[![tests](https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests&logo=github&style=flat-square)](https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/docs.yml?branch=main&label=docs&logo=materialformkdocs&style=flat-square)](https://irvlutd.github.io/RPX/)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/rpx-benchmark/)
[![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![ruff](https://img.shields.io/badge/lint-ruff-000000?logo=ruff&style=flat-square)](https://github.com/astral-sh/ruff)
[![tests passing](https://img.shields.io/badge/tests-138%20passing-brightgreen?style=flat-square)](benchmark/tests/)
[![tasks](https://img.shields.io/badge/runnable%20tasks-9%20%2F%2010-brightgreen?style=flat-square)](benchmark/README.md#-available-tasks)
[![dataset](https://img.shields.io/badge/dataset-RGB--D-6366f1?style=flat-square)](#-what-rpx-is)
[![scope](https://img.shields.io/badge/scope-robot%20learning-ff69b4?style=flat-square)](https://github.com/IRVLUTD/RPX)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](#-contributing)

[**Quickstart**](#-quickstart--benchmark-a-model-without-touching-the-dataset) · [**Benchmark Toolkit**](benchmark/README.md) · [**Docs**](https://irvlutd.github.io/RPX/) · [**Data Capture**](data_capture/README.md) · [**Mask Pipeline**](robokit/README.md) · [**Paper**](#-paper)

</div>

---

RPX is a unified real-world RGB-D benchmark for evaluating the
perception models actually deployed inside robot learning stacks
(not generic perception leaderboards). This repository contains
everything behind the benchmark:

- 📷 the **data-collection rig** scripts (Intel D435 RGB-D + T265 VIO),
- 🎨 the **ground-truth mask generator** (SAM2 + GroundingDINO),
- 🧰 the **benchmark toolkit** (`pip install rpx-benchmark`), and
- 📄 the **NeurIPS 2026 Datasets & Benchmarks paper** draft.

## ✨ At a glance

| | |
|---|---|
| 📷 **Sensor rig** | Intel RealSense D435 (RGB-D) + T265 (6-DoF VIO), pose logged at 200 Hz |
| 🎬 **3-phase capture protocol** | Clutter → Interaction (human-in-scene) → Clean on identical scenes |
| 🧪 **~75 K frames** | 100 indoor scenes, tabletop + room-scale, ~70 object categories |
| 🎯 **10 benchmark tasks** | depth, segmentation, detection (×2), grounding, pose, keypoints, sparse depth, NVS, tracking |
| 🪜 **ESD difficulty splits** | Easy / Medium / Hard derived from real annotation effort, per `(scene, phase)` |
| 🔌 **Bring-your-own-model** | HF checkpoint · numpy callable · custom adapter — pick one, run in one command |
| 📊 **Deployment-readiness scoring** | ESD-weighted phase score, state-transition robustness, temporal stability, FLOPs, latency |
| 🧰 **Full CI** | pytest matrix 3.10 / 3.11 / 3.12 + ruff + auto docs deploy to GitHub Pages |
| 📚 **Auto docs** | MkDocs + mkdocstrings reads numpydoc; adding a class = zero doc work |
| ⚖️ **License** | Code MIT · Dataset CC BY 4.0 |

## 📚 Table of contents

<table>
<tr>
<td valign="top">

**Overview**
- [✨ At a glance](#-at-a-glance)
- [🗺️ Repository map](#️-repository-map)
- [⚡ Quickstart — benchmark a model](#-quickstart--benchmark-a-model-without-touching-the-dataset)
- [🧱 The three parts of the system](#-the-three-parts-of-the-system)

**Components**
- [🧰 Benchmark toolkit](benchmark/README.md)
- [📷 Data capture rig](data_capture/README.md)
- [🎨 Mask pipeline](robokit/README.md)
- [🐳 Docker workflow](#-docker-workflow)

</td>
<td valign="top">

**Reference**
- [🛠️ Data capture prerequisites](#-data-capture-prerequisites)
- [📄 Paper](#-paper)
- [⚙️ CI / automated docs](#-ci--automated-docs)

**Community**
- [🧪 Contributing](#-contributing)
- [📑 Citation](#-citation)
- [⚖️ License](#-license)

</td>
</tr>
</table>

---

## 🗺️ Repository map

```text
RPX/
├── benchmark/              Python package + CLI: load, run, metric, report
│   ├── rpx_benchmark/      Library source
│   ├── scripts/            Per-task runners (run_depth, run_relative_pose, ...)
│   ├── tests/              Offline test suite
│   ├── docs/               MkDocs site (auto-generated from docstrings)
│   └── README.md           ←★ start here for users
│
├── data_capture/               Data-capture rig scripts (Intel D435 + T265)
│   └── README.md           Run `save_device_data.py` to capture a scene
│
├── robokit/                Ground-truth mask pipeline (SAM2 + GroundingDINO)
│   ├── maskgen_pipeline/   Maskgen scripts + bbox / point-prompt utilities
│   ├── visual_grounding_gt/  Visual-grounding GT helpers
│   ├── robokit/            Inner Python package (perception, datasets, ...)
│   ├── docker/             Maskgen-specific Docker setup
│   └── README.md           Interactive GSAM2 refinement UI
│
├── experiments/            ESD difficulty-split analysis + figures
│
├── docker/                 Top-level Dockerised reproducible env
│
├── .github/workflows/      CI: pytest matrix + ruff + MkDocs Pages deploy
│
├── README.md               This file
├── CONTRIBUTING.md
└── LICENSE
```

Each subdirectory has its own README with the full details for that
piece of the system.

---

## ⚡ Quickstart — benchmark a model without touching the dataset

The fastest way to try RPX is **evaluation mode**: install the
`rpx-benchmark` package and run any HuggingFace depth, segmentation,
detection, grounding, pose, keypoint, sparse-depth, or NVS model
against an ESD difficulty split. No data capture or annotation
needed.

```bash
pip install 'rpx-benchmark[depth]'

# Run any HuggingFace depth model
rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard

# Run any HuggingFace instance-segmentation model
rpx bench object_segmentation \
    --hf-checkpoint facebook/mask2former-swin-tiny-coco-instance \
    --split hard
```

Full toolkit docs: [`benchmark/README.md`](benchmark/README.md) and
the hosted site at <https://irvlutd.github.io/RPX/>.

---

## 🧱 The three parts of the system

### 📷 1. Data collection rig — [`data_capture/`](data_capture/README.md)

Two-sensor capture with **Intel RealSense D435 (RGB-D) + T265
(6-DoF VIO)**. Captures each scene under the three-phase protocol
(Clutter → Interaction → Clean) at synchronised FPS with pose logged
from T265.

```bash
cd data_capture
python save_device_data.py <task-name> <fps> <sync-threshold>
```

> [!NOTE]
> **T265 compatibility** — T265 support was **removed** in
> `librealsense2 > 2.47.0`. This project pins to
> `librealsense2==2.47.0` and `pyrealsense2==2.47.0.3313`. See
> [Data capture prerequisites](#-data-capture-prerequisites)
> below for install instructions.

### 🎨 2. Mask generation — [`robokit/`](robokit/README.md)

Ground-truth instance masks are generated via an interactive pipeline
combining **GroundingDINO** (open-vocabulary detection) and **SAM2**
(segment anything v2). A human operator curates the bounding box set
for one keyframe per phase; SAM2 propagates masks across the rest of
the phase.

```bash
cd robokit
python -m maskgen_pipeline.interactive_gsam2 --scene_dir /path/to/scene/1
```

### 🧰 3. Benchmark toolkit — [`benchmark/`](benchmark/README.md)

Python library + CLI. The user supplies a model; the toolkit handles:

- **Dataset download** — task-aware, fetches only the modalities your
  task needs; reuses the HuggingFace content-addressed cache.
- **Splits** — Easy / Medium / Hard per `(scene, phase)` via
  Effort-Stratified Difficulty.
- **Metrics** — pluggable per-task calculators (AbsRel, RMSE, δ-acc,
  mIoU, F1, MOTA, PSNR, SSIM, geodesic pose error, keypoint accuracy,
  ...).
- **Deployment-readiness scoring** — ESD-weighted phase score,
  State-Transition Robustness, Temporal Stability, FLOPs, median
  latency.
- **Reports** — JSON + markdown + rich terminal UI with true-colour
  gradient banner.
- **Extensibility** — add a new task, metric, or model adapter in
  **one file** via the plugin registries.

```bash
pip install 'rpx-benchmark[depth]'
rpx models                # list registered adapters
rpx bench --help          # list task subcommands (9 runnable)
```

**138 tests**, 0 network deps, runs in under a second on CI.
**9 of 10 tasks** are runnable end-to-end; only `object_tracking` is
deferred pending a sequence-per-sample protocol decision.

---

## 🛠️ Data capture prerequisites

Capturing new scenes requires the RealSense SDK built from source
against v2.47.0 (T265 support was removed afterwards).

### Install dependencies

```bash
sudo apt update
sudo apt install \
  libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev \
  git wget cmake build-essential libglfw3-dev libgl1-mesa-dev \
  libglu1-mesa-dev at
```

### Clone + build librealsense v2.47.0

```bash
git clone -b v2.47.0 https://github.com/IntelRealSense/librealsense.git
cd librealsense && ./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake ../ -DBUILD_EXAMPLES=true -DBUILDTYPE=Release
sudo make uninstall && make clean && make -j12 && sudo make install
```

> [!TIP]
> `-j12` uses 12 cores. Leave at least 2 cores free so the system
> stays responsive.

Verify the install:

```bash
realsense-viewer                # launches the GUI; connect the cameras
```

### Python wrapper (matching version)

```bash
pip install pyrealsense2==2.47.0.3313
```

> [!WARNING]
> **Disconnect devices first** — unplug all RealSense devices
> *before* running `make install`; live devices can lock the udev
> rules mid-install.

---

## 🐳 Docker workflow

For a fully reproducible environment (useful for CI, GPU setup,
multi-machine replays) — see [`docker/`](docker/README.md) for the
full details.

```bash
cd docker
./build_docker_image.sh    # one-time, a few minutes
./start_docker.sh          # detached
./start_docker.sh -i       # interactive
./enter_docker.sh          # shell into the running container
./stop_docker.sh           # stop
```

The container ships with the RealSense SDK, the benchmark toolkit,
and the robokit mask pipeline already installed.

---

## 📄 Paper

The NeurIPS 2026 Datasets & Benchmarks submission lives under
[`paper-submission/neurips-2026/`](paper-submission/neurips-2026/).
The full model slate rationale, ESD formulation, three-phase
protocol details, and experiment tables are in the paper.

---

## ⚙️ CI / automated docs

| Workflow | What it does | When it runs |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | 138-test pytest suite on Python 3.10 / 3.11 / 3.12 + ruff lint | push / PR touching `benchmark/**` |
| [`docs.yml`](.github/workflows/docs.yml) | `mkdocs build` + deploy to GitHub Pages | push to `main` touching `benchmark/docs/**` or `benchmark/rpx_benchmark/**` |

### One-time GitHub Pages setup

1. Push the repo to GitHub.
2. **Settings → Pages → Source = "GitHub Actions"**.
3. Next push to `main` triggers the `docs` workflow and the site goes
   live at <https://irvlutd.github.io/RPX/>.

Until step 2 is done, the `docs` workflow will fail with
`HttpError: Not Found` at the deploy step — that's the Pages API
telling you Pages isn't enabled yet. Harmless before the one-time
setup; fatal to the docs site after.

---

## 🧪 Contributing

Each subproject has its own contribution workflow:

- **`benchmark/`** — `pip install -e '.[dev,docs]'`, `pytest tests/`,
  `ruff check`. New tasks / metrics / model adapters land through
  the plugin registries (see
  [`benchmark/docs/guides/`](benchmark/docs/guides/)). **Always use
  the editable (`-e`) install when developing** — frozen wheel
  installs will silently show stale behaviour.
- **`data_capture/`** — changes to the capture rig need a real RealSense
  device for smoke testing.
- **`robokit/`** — mask generation changes need access to the
  interactive annotation UI and a CUDA-capable box.
- **`paper-submission/`** — LaTeX edits through whatever your usual
  Overleaf / local workflow is.

---

## 📑 Citation

If you use RPX (dataset, toolkit, or any part of this repository) in
your work, please cite the accompanying NeurIPS 2026 Datasets &
Benchmarks paper. The BibTeX entry will be added here once the
camera-ready version is released.

---

## ⚖️ License

- **Code** in this repository (benchmark toolkit, data-collection
  scripts, mask generator, docker setup): **MIT**.
- **RPX dataset** (once released): **CC BY 4.0**.

See [`LICENSE`](LICENSE).
