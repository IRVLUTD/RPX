<div align="center">

# RPX

**A real-world RGB-D benchmark for the perception models that actually
ship in robot learning stacks.**

[![tests](https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests&logo=github&style=flat-square)](https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/docs.yml?branch=main&label=docs&logo=materialformkdocs&style=flat-square)](https://irvlutd.github.io/RPX/)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue?logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/rpx-benchmark/)
[![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

<pre style="line-height:1.1">
<span style="color:#4F46E5"> ██████╗ </span><span style="color:#DB2777"> ██████╗ </span><span style="color:#C2410C"> ██╗  ██╗</span>
<span style="color:#4F46E5"> ██╔══██╗</span><span style="color:#DB2777"> ██╔══██╗</span><span style="color:#C2410C"> ╚██╗██╔╝</span>
<span style="color:#4F46E5"> ██████╔╝</span><span style="color:#DB2777"> ██████╔╝</span><span style="color:#C2410C">  ╚███╔╝ </span>
<span style="color:#4F46E5"> ██╔══██╗</span><span style="color:#DB2777"> ██╔═══╝ </span><span style="color:#C2410C">  ██╔██╗ </span>
<span style="color:#4F46E5"> ██║  ██║</span><span style="color:#DB2777"> ██║     </span><span style="color:#C2410C"> ██╔╝ ██╗</span>
<span style="color:#4F46E5"> ╚═╝  ╚═╝</span><span style="color:#DB2777"> ╚═╝     </span><span style="color:#C2410C"> ╚═╝  ╚═╝</span>
</pre>

</div>

```bash
pip install 'rpx-benchmark[depth]'
rpx bench monocular_depth \
    --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
    --split hard
```

Three minutes on one GPU. Writes a `result.json` with the Deployment
Readiness Score, all timing CIs, and the full metric basket.

---

### Pick your path

|                                  |                                  |                                  |
|----------------------------------|----------------------------------|----------------------------------|
| 🧪 **Bring your own model**       | 🤖 **Run the canonical sweep**    | 🛠️  **Contribute**               |
| Wrap a HuggingFace checkpoint or | Reproduce the full 19-depth +    | Add a task, a metric, an adapter.|
| any numpy callable in one        | 10-pose sweep behind the paper.  | 552 tests · ruff + mypy clean.   |
| function — three lines.          |                                  |                                  |
| → [`benchmark/README`](benchmark/README.md#bring-your-own-model) | → [`benchmark/TEAM_LAUNCH.md`](benchmark/TEAM_LAUNCH.md) | → [`benchmark/docs/guides/`](benchmark/docs/guides/) |

### How it works

📷 **Capture** real RGB-D + 6-DoF pose with a D435 + T265 rig, in three
phases (clutter → interaction → clean) of the same scene, with pose
logged at 200 Hz. → [`data_capture/`](data_capture/README.md)

🎨 **Annotate** with SAM2 + GroundingDINO — a human operator approves
the bounding boxes for one keyframe per phase; SAM2 propagates masks
across the rest. → [`mask_pipeline/`](mask_pipeline/README.md)

🧰 **Benchmark** any model against ESD-stratified easy/medium/hard
splits and score it on a single Deployment Readiness number that
combines accuracy, robustness, and FLOPs.
→ [`benchmark/`](benchmark/README.md)

---

<details>
<summary><b>📊 Tasks · metrics · deployment scoring</b></summary>

| | |
|---|---|
| 📷 **Sensor rig** | Intel RealSense D435 (RGB-D) + T265 (6-DoF VIO), pose logged at 200 Hz |
| 🎬 **3-phase capture protocol** | Clutter → Interaction (human-in-scene) → Clean on identical scenes |
| 🧪 **~75 K frames** | 99 scenes (64 indoor + 35 outdoor), tabletop + room-scale, ~70 object categories |
| 🎯 **10 benchmark tasks** | depth, segmentation, detection (×2), grounding, pose, keypoints, sparse depth, NVS, tracking |
| 🪜 **ESD difficulty splits** | Easy / Medium / Hard derived from real annotation effort, per `(scene, phase)` |
| 🔌 **Bring-your-own-model** | HF checkpoint · numpy callable · custom adapter — pick one, run in one command |
| 📊 **Deployment Readiness Score** | TP × R × E — ESD-weighted phase score, state-transition robustness, FLOPs-anchored efficiency |
| 🧰 **Full CI** | pytest matrix 3.10 / 3.11 / 3.12 + ruff + auto docs deploy to GitHub Pages |
| 📚 **Auto docs** | MkDocs + mkdocstrings reads numpydoc; adding a class = zero doc work |
| ⚖️ **License** | Code MIT · Dataset CC BY 4.0 |

**The 10 tasks (9 runnable end-to-end):** monocular depth · object
segmentation · object detection · open-vocab detection · object
tracking · visual grounding · relative camera pose · sparse depth ·
novel-view synthesis · keypoint matching. Per-task primary metrics +
numpy factory entry points live in
[`benchmark/README`](benchmark/README.md#tasks-10-contracts-loader-side-support-varies).

**Deployment readiness** combines an ESD-weighted phase score, a
State-Transition Robustness score across phase changes, and a
FLOPs-anchored efficiency factor into one number. Full axiomatization
in the methodology paper (kept local under `paper-submission/`).

</details>

<details>
<summary><b>🗺️ Repository map</b></summary>

```text
RPX/
├── benchmark/              Python package + CLI: load, run, metric, report
│   ├── rpx_benchmark/      Library source
│   ├── scripts/            Per-task runners (run_depth, run_relative_pose, …)
│   ├── tests/              Offline test suite (552 passing)
│   ├── docs/               MkDocs site (auto-generated from docstrings)
│   └── README.md           ←★ start here for users
│
├── data_capture/           Data-capture rig scripts (Intel D435 + T265)
│   └── README.md           Run `save_device_data.py` to capture a scene
│
├── mask_pipeline/          Ground-truth mask pipeline (SAM2 + GroundingDINO)
│   ├── maskgen_pipeline/   Maskgen scripts + bbox / point-prompt utilities
│   ├── visual_grounding_gt/  Visual-grounding GT helpers
│   ├── robokit/            Inner Python package — `import robokit.perception`
│   ├── docker/             Maskgen-specific Docker setup
│   └── README.md           Interactive GSAM2 refinement UI
│
├── experiments/            ESD difficulty-split analysis + figures
├── docker/                 Top-level Dockerised reproducible env
├── .github/workflows/      CI: pytest matrix + ruff + MkDocs Pages deploy
├── README.md               This file
├── CONTRIBUTING.md
└── LICENSE
```

Each subdirectory has its own README with the full details for that
piece of the system.

</details>

<details>
<summary><b>🛠️ Data capture prerequisites (RealSense build)</b></summary>

Capturing new scenes requires the RealSense SDK built from source
against v2.47.0 (T265 support was removed afterwards).

**Install dependencies**

```bash
sudo apt update
sudo apt install \
  libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev \
  git wget cmake build-essential libglfw3-dev libgl1-mesa-dev \
  libglu1-mesa-dev at
```

**Clone + build librealsense v2.47.0**

```bash
git clone -b v2.47.0 https://github.com/IntelRealSense/librealsense.git
cd librealsense && ./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake ../ -DBUILD_EXAMPLES=true -DBUILDTYPE=Release
sudo make uninstall && make clean && make -j12 && sudo make install
```

> [!TIP]
> `-j12` uses 12 cores. Leave at least 2 free so the system stays
> responsive.

```bash
realsense-viewer                # verify — launches GUI; connect the cameras
pip install pyrealsense2==2.47.0.3313
```

> [!WARNING]
> **Disconnect devices first** — unplug all RealSense devices *before*
> running `make install`; live devices can lock the udev rules
> mid-install.

</details>

<details>
<summary><b>🐳 Docker workflow</b></summary>

For a fully reproducible environment (useful for CI, GPU setup,
multi-machine replays) — see [`docker/`](docker/README.md) for full
details.

```bash
cd docker
./build_docker_image.sh    # one-time, a few minutes
./start_docker.sh          # detached
./start_docker.sh -i       # interactive
./enter_docker.sh          # shell into the running container
./stop_docker.sh           # stop
```

The container ships with the RealSense SDK, the benchmark toolkit,
and the mask-generation pipeline already installed.

</details>

<details>
<summary><b>⚙️ CI · automated docs · paper</b></summary>

| Workflow | What it does | When it runs |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | 552-test pytest suite on Python 3.10 / 3.11 / 3.12 + ruff lint | push / PR touching `benchmark/**` |
| [`docs.yml`](.github/workflows/docs.yml) | `mkdocs build` + deploy to GitHub Pages | push to `main` touching `benchmark/docs/**` or `benchmark/rpx_benchmark/**` |

**One-time GitHub Pages setup**

1. Push the repo to GitHub.
2. **Settings → Pages → Source = "GitHub Actions"**.
3. Next push to `main` triggers the `docs` workflow and the site goes
   live at <https://irvlutd.github.io/RPX/>.

Until step 2 is done, the `docs` workflow will fail with
`HttpError: Not Found` at the deploy step — that's the Pages API
telling you Pages isn't enabled yet. Harmless before the one-time
setup; fatal to the docs site after.

**Paper.** The methodology paper is under preparation. Drafts, briefs,
and the LaTeX project are kept **local-only** under `paper-submission/`
on each contributor's box (not in this public repo). The full model
slate rationale, ESD formulation, three-phase protocol details, and
experiment tables live in the paper.

</details>

<details>
<summary><b>🧪 Contributing · citation · license</b></summary>

Each subproject has its own contribution workflow:

- **`benchmark/`** — `pip install -e '.[dev,docs]'`, `pytest tests/`,
  `ruff check`. New tasks / metrics / model adapters land through
  the plugin registries (see
  [`benchmark/docs/guides/`](benchmark/docs/guides/)). **Always use
  the editable (`-e`) install when developing** — frozen wheel
  installs will silently show stale behaviour.
- **`data_capture/`** — changes to the capture rig need a real RealSense
  device for smoke testing.
- **`mask_pipeline/`** — mask generation changes need access to the
  interactive annotation UI and a CUDA-capable box.
- **`paper-submission/`** — LaTeX edits through whatever your usual
  Overleaf / local workflow is.

**Citation.** If you use RPX (dataset, toolkit, or any part of this
repository) in your work, please cite the accompanying paper. The
BibTeX entry will be added here once the paper is publicly released.

**License.** Code in this repository (benchmark toolkit, data-collection
scripts, mask generator, docker setup) is **MIT**. The RPX dataset
(once released) is **CC BY 4.0**. See [`LICENSE`](LICENSE).

</details>

---

<sub>📄 Paper: `paper-submission/` (local-only) · ⚖️ Code MIT · Dataset CC BY 4.0 · 🧪 PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md)</sub>
