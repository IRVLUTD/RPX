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

Three minutes on one GPU. Writes a `result.json` reporting **three
independent axes** — task performance, scene-change robustness,
compute cost — plus timing CIs and the full metric basket. No
combined score.

---

## Why RPX

Perception models routinely lose 10–30% of their accuracy when they
leave clean lab demos and meet cluttered, human-shared environments.
RPX measures that drop on **100 indoor + outdoor scenes** — each
captured in three phases (clutter → human-interaction → clean; pair
tasks like RCPE / NVS evaluate on clutter + clean only) — and
reports every model on **three independent axes**, never collapsed
into a single composite:

1. **Task Performance** — per task (AUC, AbsRel, rotation error, …).
2. **Scene-change robustness** — STR, cross-phase Δ, temporal drift.
   *Unique to RPX.*
3. **Compute cost** — params, FLOPs, latency†.

The headline result of the paper is that **rankings disagree across
these axes**: the model that wins on accuracy is rarely the model
that wins on robustness or cost. RPX is the benchmark that surfaces
that disagreement instead of hiding it behind a single number.

The dataset lives on HuggingFace. The toolkit is one `pip install`.
Most users only ever touch `benchmark/`; capture and annotation exist
to *produce* the dataset, not to consume it.

```
   CAPTURE                 ANNOTATE                 BENCHMARK
   ───────                 ────────                 ─────────
   D435 + T265        →    SAM2 + GroundingDINO  →  load → score on
   RGB-D + 6-DoF pose      per-frame instance       three independent axes
                           masks (1 keyframe of     · task performance
                           human review per phase)  · scene-change Δ
                                                    · compute cost

   data/capture/           data/mask_annotation/           benchmark/  ←  start here
```

---

## Get started

|                                       |                                       |                                       |
|---------------------------------------|---------------------------------------|---------------------------------------|
| **Bring your own model**              | **Reproduce the paper sweep**         | **Contribute to the toolkit**         |
| Wrap a HuggingFace checkpoint or any  | Run the full 19-depth + 10-pose       | Add a task, a metric, an adapter.     |
| numpy callable in three lines.        | sweep behind the paper.               | 552 tests · ruff + mypy clean.        |
| → [`benchmark/README`](benchmark/README.md#bring-your-own-model) | → [`benchmark/TEAM_LAUNCH.md`](benchmark/TEAM_LAUNCH.md) | → [`benchmark/docs/guides/`](benchmark/docs/guides/) |

---

## How it works

**Capture.** Intel RealSense D435 (RGB-D) and T265 (6-DoF VIO at
200 Hz) record each scene through three phases — a cluttered initial
state, a human-interaction pass, and a clean organised state — at
synchronised frame rates. Same scene, three states.
→ [`data/capture/`](data/capture/README.md)

**Annotate.** SAM2 propagates curated bounding boxes through every
frame in a phase; a human operator approves one keyframe per phase.
The output is per-frame instance masks aligned to a globally
consistent object-ID mapping.
→ [`data/mask_annotation/`](data/mask_annotation/README.md)

**Benchmark.** Models run against ESD-stratified easy / medium / hard
splits. Each run emits a `result.json` reporting all three axes
separately — task performance, scene-change robustness, compute cost —
plus per-stage timing with 95% bootstrap CIs. No combined score.
→ [`benchmark/`](benchmark/README.md)

---

<details>
<summary><b>Concepts and acronyms</b></summary>

| Term | Meaning |
|---|---|
| **RGB-D** | Color image + per-pixel depth (distance from the camera, in metres). The D435 captures both simultaneously. |
| **Depth** | A per-pixel distance map. The depth model's job: predict it from RGB alone. |
| **6-DoF pose** | The camera's full 3D position + 3D orientation — six numbers per frame. The T265 measures this in real time. |
| **VIO** | Visual-Inertial Odometry. The T265 fuses fisheye stereo + IMU to estimate 6-DoF pose at 200 Hz. |
| **Phase** | One of three capture passes per scene: **clutter** (objects scattered), **interaction** (a human reaches in), **clean** (organised state). Same scene, three states — used to measure phase-transition robustness. |
| **ESD** | Effort-Stratified Difficulty. Splits the 100 scenes into easy / medium / hard by mixing a `perception_score` (how visually hard the scene is) with an `effort_score` (how much human labor the masks took). |
| **Task Performance** | Axis 1 of the report. Per-task primary metric (AUC, AbsRel, rotation error, …) plus 95% bootstrap CIs. |
| **Scene-change robustness** | Axis 2 of the report. Captures how a model's output changes when the same scene transitions through clutter → interaction → clean. Components: STR, cross-phase Δ, temporal drift. Unique to RPX. |
| **Compute cost** | Axis 3 of the report. Params (M), FLOPs (G), Latency† (hardware-dependent, supplementary). |
| **STR** | State-Transition Robustness. The headline scene-change-robustness number: how stable a model's output is across phase transitions of the same scene. |
| **RCPE** | Relative Camera Pose Estimation. Given two RGB frames, estimate the relative 3D rotation + translation between them. One of the 10 tasks. |
| **Adapter** | A small Python class wrapping your model behind the toolkit's uniform interface. Three flavours: HF checkpoint, numpy callable, custom subclass. |
| **Manifest** | A JSON listing the scenes / frames / modalities for a given task + split. The loader reads it; you almost never write one. |
| **Split** | A named subset of the dataset (`easy`, `medium`, `hard`). All three phases of one scene always land in the same split so STR is measurable. |
| **Operating point** | One (precision × accuracy × FLOPs) triple per model. A model can publish several (fp32, fp16, int8); each is reported separately on the three axes. |
| **HF** | HuggingFace — where the RPX dataset and most reference checkpoints live. |
| **`rpx_results/<model>/<split>/`** | The canonical output directory. `result.json` is the load-bearing artefact. |

</details>

<details>
<summary><b>Tasks · metrics · deployment scoring</b></summary>

| | |
|---|---|
| 📷 **Sensor rig** | Intel RealSense D435 (RGB-D) + T265 (6-DoF VIO), pose logged at 200 Hz |
| 🎬 **3-phase capture protocol** | Clutter → Interaction (human-in-scene) → Clean on identical scenes |
| 🧪 **~75 K frames** | 100 scenes (indoor + outdoor), tabletop + room-scale, ~70 object categories |
| 🎯 **10 benchmark tasks** | depth, segmentation, detection (×2), grounding, pose, keypoints, sparse depth, NVS, tracking |
| 🪜 **ESD difficulty splits** | Easy / Medium / Hard derived from real annotation effort, per `(scene, phase)` |
| 🔌 **Bring-your-own-model** | HF checkpoint · numpy callable · custom adapter — pick one, run in one command |
| 📊 **Three-axis reporting** | Task Performance · scene-change robustness (STR, cross-phase Δ, temporal drift) · compute cost — reported separately, never combined into a single score |
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
<summary><b>Repository map</b></summary>

```text
RPX/
├── benchmark/              Python package + CLI: load, run, metric, report
│   ├── rpx_benchmark/      Library source
│   ├── scripts/            Per-task runners (run_depth, run_relative_pose, run_nvs, …)
│   ├── tests/              Offline test suite (552 passing)
│   ├── docs/               MkDocs site (auto-generated from docstrings)
│   └── README.md           ←★ start here for users
│
├── data/capture/           Data-capture rig scripts (Intel D435 + T265)
│   └── README.md           Run `save_device_data.py` to capture a scene
│
├── data/mask_annotation/          Ground-truth mask pipeline (SAM2 + GroundingDINO)
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
<summary><b>Data capture prerequisites — RealSense build</b></summary>

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
<summary><b>Docker workflow</b></summary>

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
<summary><b>CI · automated docs · paper</b></summary>

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
<summary><b>Contributing · citation · license</b></summary>

Each subproject has its own contribution workflow:

- **`benchmark/`** — `pip install -e '.[dev,docs]'`, `pytest tests/`,
  `ruff check`. New tasks / metrics / model adapters land through
  the plugin registries (see
  [`benchmark/docs/guides/`](benchmark/docs/guides/)). **Always use
  the editable (`-e`) install when developing** — frozen wheel
  installs will silently show stale behaviour.
- **`data/capture/`** — changes to the capture rig need a real RealSense
  device for smoke testing.
- **`data/mask_annotation/`** — mask generation changes need access to the
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

<sub>Paper: `paper-submission/` (local-only) · Code MIT · Dataset CC BY 4.0 · PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md)</sub>
