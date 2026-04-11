# RPX — Robot Perception X

> **Choose and rank perception models for robot learning** — on
> real-world RGB-D data, under embodied deployment conditions, with
> ESD-stratified difficulty splits and deployment-readiness scoring.

RPX is a unified real-world RGB-D benchmark for evaluating the
perception models actually deployed inside robot learning stacks
(not generic perception leaderboards). This repository contains
everything behind the benchmark:

- the **data-collection rig** scripts (Intel D435 RGB-D + T265 VIO),
- the **ground-truth mask generator** (SAM2 + GroundingDINO),
- the **benchmark toolkit** (`pip install rpx-benchmark`), and
- the **NeurIPS 2026 Datasets & Benchmarks paper** draft.

---

## Repository map

```text
RPX/
├── benchmark/              Python package + CLI: load, run, metric, report
│   ├── rpx_benchmark/      Library source
│   ├── docs/               MkDocs site (auto-generated from docstrings)
│   ├── tests/              120-test offline suite
│   └── README.md           Toolkit-specific README ←★ start here for users
│
├── dc/                     Data-collection rig scripts (D435 + T265)
│   └── README.md           Run `save_device_data.py` to capture a scene
│
├── robokit/                Ground-truth mask pipeline (SAM2 + GroundingDINO)
│   └── README.md           Interactive GSAM2 refinement UI
│
├── docker/                 Dockerised reproducible environment
│   └── README.md           Build, start, stop, exec helpers
│
├── paper-submission/       LaTeX source for the NeurIPS 2026 paper
│
├── external/               Third-party submodules (rerun visualiser, ...)
│
├── scripts/                Auxiliary helpers
│
├── .github/workflows/      CI: pytest matrix + ruff + MkDocs Pages deploy
│
└── README.md               This file
```

Each subdirectory has its own README with the full details for that
piece of the system.

---

## Quickstart — benchmark a model without touching the dataset

The fastest way to try RPX is **evaluation mode**: install the
`rpx-benchmark` package and run any HuggingFace depth or segmentation
checkpoint against an ESD difficulty split. No data capture or
annotation needed.

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

## The three parts of the system

### 1. Data collection rig — [`dc/`](dc/README.md)

Two-sensor capture with **Intel RealSense D435 (RGB-D) + T265
(6-DoF VIO)**. Captures each scene under the three-phase protocol
(Clutter → Interaction → Clean) at synchronised FPS with pose logged
from T265.

```bash
cd dc
python save_device_data.py <task-name> <fps> <sync-threshold>
```

!!! note "T265 compatibility"
    T265 support was **removed** in `librealsense2 > 2.47.0`. This
    project pins to `librealsense2==2.47.0` and
    `pyrealsense2==2.47.0.3313`. See the *Data capture prerequisites*
    section below for install instructions.

### 2. Mask generation — [`robokit/`](robokit/README.md)

Ground-truth instance masks are generated via an interactive pipeline
combining **GroundingDINO** (open-vocabulary detection) and **SAM2**
(segment anything v2). A human operator curates the bounding box set
for one keyframe per phase; SAM2 propagates masks across the rest of
the phase.

```bash
cd robokit
python -m maskgen_pipeline.interactive_gsam2 --scene_dir /path/to/scene/1
```

### 3. Benchmark toolkit — [`benchmark/`](benchmark/README.md)

Python library + CLI. The user supplies a model; the toolkit handles:

- **Dataset download** — task-aware, fetches only the modalities your
  task needs; reuses the HuggingFace content-addressed cache.
- **Splits** — Easy / Medium / Hard per `(scene, phase)` via
  Effort-Stratified Difficulty.
- **Metrics** — pluggable per-task calculators (AbsRel, RMSE, δ-acc,
  mIoU, MOTA, PSNR, SSIM, ...).
- **Deployment-readiness scoring** — ESD-weighted phase score,
  State-Transition Robustness, Temporal Stability, FLOPs, median
  latency.
- **Reports** — JSON + markdown + rich terminal UI.
- **Extensibility** — add a new task, metric, or model adapter in
  **one file** via the plugin registries.

```bash
pip install 'rpx-benchmark[depth]'
rpx models                # list registered adapters
rpx bench --help          # list task subcommands
```

120 tests, 0 network deps, runs in under a second.

---

## Data capture prerequisites

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

!!! tip
    `-j12` uses 12 cores. Leave at least 2 cores free so the system
    stays responsive.

Verify the install:

```bash
realsense-viewer                # launches the GUI; connect the cameras
```

### Python wrapper (matching version)

```bash
pip install pyrealsense2==2.47.0.3313
```

!!! warning "Disconnect devices first"
    Unplug all RealSense devices **before** running `make install` —
    live devices can lock the udev rules mid-install.

---

## Docker workflow — [`docker/`](docker/README.md)

For a fully reproducible environment (useful for CI, GPU setup,
multi-machine replays):

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

## Paper

The NeurIPS 2026 Datasets & Benchmarks submission lives under
[`paper-submission/neurips-2026/`](paper-submission/neurips-2026/).
The full model slate rationale, ESD formulation, three-phase
protocol details, and experiment tables are in the paper.

---

## CI / automated docs

| Workflow | What it does | When it runs |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | 120-test pytest suite on Python 3.10 / 3.11 / 3.12 + ruff lint | push / PR touching `benchmark/**` |
| [`docs.yml`](.github/workflows/docs.yml) | `mkdocs build` + deploy to GitHub Pages | push to `main` touching `benchmark/docs/**` or `benchmark/rpx_benchmark/**` |

To enable GitHub Pages once (one-time setup):

1. Push the repo to GitHub.
2. **Settings → Pages → Source = "GitHub Actions"**.
3. Next push to `main` triggers the workflow and the site goes live at
   <https://irvlutd.github.io/RPX/>.

---

## Repository-level contributing

Each subproject has its own contribution workflow:

- **`benchmark/`** — `pip install -e '.[dev,docs]'`, `pytest tests/`,
  `ruff check`. New tasks / metrics / model adapters land through
  the plugin registries (see
  [benchmark/docs/guides/](benchmark/docs/guides/)).
- **`dc/`** — changes to the capture rig need a real RealSense
  device for smoke testing.
- **`robokit/`** — mask generation changes need access to the
  interactive annotation UI and a CUDA-capable box.
- **`paper-submission/`** — LaTeX edits through whatever your usual
  Overleaf / local workflow is.

---

## Citation

If you use RPX (dataset, toolkit, or any part of this repository) in
your work, please cite the accompanying NeurIPS 2026 Datasets &
Benchmarks paper. The BibTeX entry will be added here once the
camera-ready version is released.

---

## License

- **Code** in this repository (benchmark toolkit, data-collection
  scripts, mask generator, docker setup): **MIT**.
- **RPX dataset** (once released): **CC BY 4.0**.

See [`LICENSE`](LICENSE).
