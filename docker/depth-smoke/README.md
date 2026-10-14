# Depth images (T1 and T2)

Pull `irvlutd/rpx:depth-zipdepth-latest` for every T1 model and every T2 model except DVD
(`irvlutd/rpx:depth-dvd-latest`) and GemDepth (build only). See the [image table](../README.md#which-image-to-pull).
It contains the 16 models built by this directory's Dockerfile plus the [FE2E](../depth-fe2e/README.md) and
[ZipDepth](../depth-zipdepth/README.md) overlays.

The image holds code, pinned Python environments and pinned upstream checkouts. Data, checkpoints, credentials and
results stay in mounted host directories.

<p align="center">
  <a href="#models">Models</a> ·
  <a href="#host-setup">Host setup</a> ·
  <a href="#build">Build</a> ·
  <a href="#gates">Gates</a> ·
  <a href="#run-micro-and-acceptance">Run</a> ·
  <a href="#results">Results</a> ·
  <a href="#adding-a-model">Adding a model</a> ·
  <a href="#troubleshooting">Troubleshooting</a>
</p>

## Models

The roster has 20 models: 10 image and 10 video. The Dockerfile's final stage, `depth_all_hyden`, builds 16 of them;
the FE2E, ZipDepth, DVD and GemDepth overlays add the rest. "Baked" means dependencies install and upstream imports
succeed. It does not mean the model has passed acceptance, so run micro and acceptance on the exact image you build.

| Model | Task | Environment | Status in this image |
|---|---|---|---|
| `da-v2-large` | image | `transformers-image` | baked, matrix-ready, acceptance passed |
| `da3-metric-l` | image | `da3` | baked, matrix-ready |
| `depth-pro` | image | `transformers-image` | baked, matrix-ready, acceptance passed |
| `fe2e` | image | [`depth-fe2e`](../depth-fe2e/README.md) | separate PyTorch 2.6 overlay |
| `hyden` | image | `metadepth` (alias `hyden`) | baked, matrix-ready; checkpoint may need access approval |
| `lotus-2` | image | `lotus2` | baked, matrix-ready |
| `metric3d-v2` | image | `metric3d` | baked, matrix-ready |
| `moge-2-vit-l` | image | `moge2` | baked, matrix-ready, acceptance passed |
| `unidepth-v2` | image | `unidepth2` | baked, matrix-ready, acceptance passed |
| `zipdepth` | image | [`depth-zipdepth`](../depth-zipdepth/README.md) | separate overlay |
| `chrono-depth` | video | `chrono` | baked, import-checked; no readiness marker yet |
| `da3-video` | video | `da3` | baked, matrix-ready |
| `depth-crafter` | video | `depthcrafter` | baked, import-checked; no readiness marker yet |
| `dvd` | video | [`depth-dvd`](../depth-dvd/README.md) | separate overlay (DVD v1.1 + Wan2.1) |
| `gem-depth` | video | [`depth-gemdepth`](../depth-gemdepth/README.md) | separate overlay, build only |
| `monst3r` | video | `monst3r` | baked, matrix-ready |
| `rolling-depth` | video | `rolling` (backed by `rollingdepth`) | baked, matrix-ready |
| `vggt-omega` | video | `vggt` (backed by `geometry-video`) | baked, matrix-ready |
| `video-da` | video | `video_da` (backed by `video-depth-anything`) | baked, matrix-ready |
| `vigeo` | video | `vigeo` (backed by `geometry-video`) | baked, matrix-ready |

<details>
<summary>Dockerfile stages</summary>

The Dockerfile is cumulative, with each stage extending the previous one:

| Stage | Adds |
|---|---|
| published four-model base | `da-v2-large`, `depth-pro`, `unidepth-v2`, `moge-2-vit-l` |
| `depth_all_da3` | `da3-metric-l`, `da3-video` |
| `depth_all_lotus2` | `lotus-2` |
| `depth_all_metric3d` | `metric3d-v2` |
| `depth_all_geometry_video` | `vggt-omega`, `vigeo` |
| `depth_all_diffusion_video` | `chrono-depth`, `depth-crafter` |
| `depth_all_rolling_vda_video` | `rolling-depth`, `video-da` |
| `depth_all_monst3r` | `monst3r` |
| `depth_all_hyden` | `hyden` |

</details>

`python3 benchmark/scripts/run_depth_smoke_matrix.py --list-models` prints the roster from code.

## Host setup

You need:

- Linux x86-64;
- Docker with BuildKit/buildx;
- an NVIDIA GPU with a driver that supports CUDA 12.8, plus the NVIDIA Container Toolkit;
- a Hugging Face token for gated checkpoints;
- more than 100 GB free for image layers, weights, the RPX cache and outputs.

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# If the GPU is not visible:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

export HF_HOME="$HOME/.cache/huggingface"
export RPX_SMOKE_ROOT="$PWD/rpx-smoke"
export HF_TOKEN="hf_..."            # keep it in the environment, never in the image
mkdir -p "$HF_HOME" "$RPX_SMOKE_ROOT"
```

## Build

```bash
git clone https://github.com/IRVLUTD/RPX.git rpx-depth-smoke
cd rpx-depth-smoke

export RPX_GIT_SHA="$(git rev-parse HEAD)"
export METADEPTH_SHA="$(tr -d '[:space:]' < docker/depth-smoke/metadepth.sha)"
export IMAGE="rpx-depth-smoke:sha-$(git rev-parse --short=12 HEAD)"
mkdir -p ../depth-smoke-envs/.wheelhouse

docker buildx build --load \
  --file docker/depth-smoke/Dockerfile \
  --target depth_all_hyden \
  --build-context \
    "torch_wheelhouse=../depth-smoke-envs/.wheelhouse" \
  --build-arg "RPX_GIT_SHA=$RPX_GIT_SHA" \
  --build-arg "METADEPTH_SHA=$METADEPTH_SHA" \
  --label "org.opencontainers.image.revision=$RPX_GIT_SHA" \
  --tag "$IMAGE" \
  .
```

The first stage starts from the published four-model image, pinned by digest. Each later stage creates its own
environment under `/opt/rpx-envs`, checks its pinned upstream imports during the build, and passes the environment
forward. To work on one family, build its stage with `--target`, for example `--target depth_all_rolling_vda_video`.

`build_and_push.sh` is the original four-model publishing helper. It refuses a dirty tree and does not yet forward
`METADEPTH_SHA`, so use the command above for the final stage.

Check the result:

```bash
# CUDA and the original four runtimes
docker run --rm --gpus all --ipc=host --shm-size=8g "$IMAGE" verify
# full roster
docker run --rm "$IMAGE" list
docker image inspect "$IMAGE" \
  --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
```

`verify` does not download weights or run samples, so it is a preflight, not a smoke pass.

## Gates

Run the gates in order. Passing one gate does not count as passing the next, and neither a successful build nor an
import check counts as a micro pass.

1. `verify` checks the container and CUDA only.
2. `micro` runs the smallest test with real data and a real checkpoint.
3. `acceptance` runs a larger real-data test.
4. `easy` runs the full Easy tier. Run it only after acceptance has passed and the adapter has been reviewed.

All gates run on CUDA on one idle NVIDIA GPU. They use `IRVLUTD/RPX` at revision
`2e2a387f7f93e98c177b2e039c141eacda94e5fc`, the Easy split, the official checkpoint and batch size 1 for image models.
Output stays local; gates never upload to Box or Hugging Face.

| Gate | Image models | Video models |
|:--|:--|:--|
| `micro` | 1 image | 1 clip, 8 frames by stride sampling |
| `acceptance` | 25 images | 1 clip, 25 frames |
| `easy` | all 24,750 Easy images | all 99 Easy clips, every frame (`sampling=all`) |

A gate passes when:

- image `micro`: 1 sample and 1 cell with finite metrics, valid GPU and precision fields in the SystemCard, and
  exactly one finite, non-degenerate `.npz` (the raw prediction is saved and FLOP profiling is skipped);
- image `acceptance`: 25 samples, 1 cell, finite metrics and 25 finite saved predictions;
- image `easy`: 24,750 samples and predictions, and a complete `comprehensive_metrics.json`;
- video `micro` and `acceptance`: 1 sample and 1 cell, a valid SystemCard, and finite `absrel`, `rmse`, `delta1`,
  `delta2` and `delta3`;
- video `easy`: 99 samples and cells with valid core metrics.

The temporal metrics `tae`, `opw`, `tgm` and `tcc` can be non-finite on clips too short to support them. The five core
metrics must always be finite.

Every gate writes `result.json`, `cells.parquet`, `summary.md`, `run_metadata.json`, `run.log` and `pip-freeze.txt`;
image gates also save predictions. `run_metadata.json` records the code identity and dataset
revision, the Python, PyTorch and GPU versions, the upstream SHAs, the command, the validation result and the failure
class.

## Run micro and acceptance

```bash
rpx_docker_run() {
  docker run --rm --gpus all --ipc=host --shm-size=8g \
    -e HF_TOKEN \
    -v "$HF_HOME:/cache/huggingface" \
    -v "$RPX_SMOKE_ROOT:/outputs" \
    "$@"
}
```

The `smoke` entrypoint runs micro and then acceptance for the four original image models. Set `RPX_MODELS` to pick
one. It handles image models only.

```bash
rpx_docker_run "$IMAGE" smoke
rpx_docker_run -e RPX_MODELS=depth-pro "$IMAGE" smoke
```

To run one gate for any model, call the gate script with that model family's Python. Pass `--gate micro` first, then
`--gate acceptance`. For a video model, use `--task video` with its own environment, for example
`/opt/rpx-envs/video_da/bin/python`, `--model video-da` and `--upstream-dir /opt/rpx-envs/sources/video-depth-anything`.

```bash
rpx_docker_run "$IMAGE" \
  /opt/rpx-envs/da3/bin/python \
  scripts/run_depth_smoke_gate.py \
  --task image \
  --model da3-metric-l \
  --gate micro \
  --cache-dir /cache/huggingface \
  --output-root /outputs \
  --upstream-dir /opt/rpx-envs/sources/da3
```

The matrix is the usual way to run several models. It runs one model at a time and requires micro before acceptance.
It skips gates that already passed for the same code identity and writes a single state file.

```bash
IMAGE_MODELS=da-v2-large,da3-metric-l,depth-pro,hyden,lotus-2
IMAGE_MODELS+=,metric3d-v2,moge-2-vit-l,unidepth-v2

rpx_docker_run "$IMAGE" \
  python scripts/run_depth_smoke_matrix.py \
  --task image \
  --models "$IMAGE_MODELS" \
  --gates micro,acceptance \
  --env-root /opt/rpx-envs \
  --cache-dir /cache/huggingface \
  --output-root /outputs

rpx_docker_run "$IMAGE" \
  python scripts/run_depth_smoke_matrix.py \
  --task video \
  --models da3-video,monst3r,rolling-depth,vggt-omega,video-da,vigeo \
  --gates micro,acceptance \
  --env-root /opt/rpx-envs \
  --cache-dir /cache/huggingface \
  --output-root /outputs
```

ChronoDepth and DepthCrafter are not in the matrix yet. Their stages don't write `rpx-environment.json`, the marker the
matrix requires. Run them with the single-gate command, using `/opt/rpx-envs/chrono/bin/python` and
`/opt/rpx-envs/depthcrafter/bin/python`. FE2E and GemDepth use their own overlays.

Pick a GPU with `--gpu-index N`. Don't use `--allow-busy-gpu` for acceptance or for any run you will publish.

### Resuming and retrying

Matrix state is keyed by code identity, so rerunning a command resumes it, while a new commit or a dirty diff starts
fresh.

- Passed gates are skipped unless you pass `--rerun-passed`.
- After fixing a code or dependency problem, pass `--retry-failed`. Code and dependency failures get two attempts.
  Access, download and out-of-memory failures get one.
- A CUDA OOM is a hardware limit. Rerun the same command on a GPU with more memory rather than changing the model.

## Results

Outputs land on the host through the `/outputs` mount:

```text
$RPX_SMOKE_ROOT/
├── matrix/<code-id>/<host>/matrix.json
└── <code-id>/<host>/<model>/<gate>/<UTC timestamp>/
    ├── result.json
    ├── cells.parquet
    ├── summary.md
    ├── run_metadata.json
    ├── run.log
    ├── pip-freeze.txt
    └── predictions/                 # image gates
```

Read pass or fail from `run_metadata.json` and `matrix.json`, not from console output. Keep failed runs: their logs
classify the failure as dependency, access, weight download, upstream API, shape or dtype, data or manifest, CUDA OOM,
or unknown.

To move the image to another machine, use `docker save` rather than `docker export`, so that layers, labels and the
entrypoint are kept:

```bash
docker save "$IMAGE" \
  | gzip > "rpx-depth-smoke-$(git rev-parse --short=12 HEAD).tar.gz"
gunzip -c rpx-depth-smoke-<sha>.tar.gz | docker load
```

<details>
<summary>Publishing an image</summary>

When publishing, push the immutable tag first. Move `latest` only after that image passes the gates, and point
consumers at the `sha-…` tag:

```bash
export REGISTRY_IMAGE="<registry>/<namespace>/rpx-depth-smoke"
export IMMUTABLE_TAG="sha-$(git rev-parse --short=12 HEAD)"

docker tag "$IMAGE" "$REGISTRY_IMAGE:$IMMUTABLE_TAG"
docker push "$REGISTRY_IMAGE:$IMMUTABLE_TAG"

docker tag "$IMAGE" "$REGISTRY_IMAGE:latest"
docker push "$REGISTRY_IMAGE:latest"
```

</details>

## Adding a model

1. Verify the official release. Find the authors' source repository, weights, checkpoint name, license and load
   procedure, and pin the source to a full Git SHA. Don't relax a safety check to accommodate an unverified community
   upload.

2. Write the adapter.
   - An image model goes in `benchmark/scripts/depth_models/<model_name>.py`. It returns float32 depth for an RGB
     `uint8` image and declares `native_alignment` (`none` for metric depth). Expose the Torch module for profiling if
     you can, and register the model in `depth_models/__init__.py`.
   - A video model goes in `benchmark/scripts/video_depth_models/<model_name>.py` with a `build(device=...)` factory.
     It takes a whole clip, returns `(T, H, W)` at the input size, and says whether its output is metric or relative.
     It must not fall back to per-frame inference unless the row is meant as a frame-as-video baseline.
   - Add the name to `IMAGE_MODELS` or `VIDEO_MODELS` in `benchmark/scripts/run_depth_smoke_gate.py`. If the weights
     are unverified, add it to `BLOCKED_MODELS` instead.

3. Define the environment family. In `benchmark/scripts/setup_depth_smoke_env.py`, fill in `MODEL_FAMILY`,
   `UPSTREAMS` (repository and SHA), `FAMILY_PACKAGES` and `FAMILY_IMPORTS`, the last of which should exercise the
   adapter's real load path. A host setup catches conflicts faster than a Docker rebuild:

   ```bash
   python3.11 benchmark/scripts/setup_depth_smoke_env.py \
     --model <canonical-name> \
     --env-root /data/rpx-envs
   ```

4. Lock dependencies. Add `requirements-<family>.in` with the direct versions and a reviewed
   `requirements-<family>.lock`. Leave Torch, torchvision and CUDA to the base image unless the family must replace
   them, and prefer `opencv-python-headless`. Check the lock for accidental upgrades to NumPy, Transformers, Diffusers,
   OpenCV, xFormers or Hugging Face Hub, which are the packages most likely to break an upstream API or ABI.

5. Add a Docker stage after the current last one. The stage should:
   - pin and fetch the upstream SHA;
   - create `/opt/rpx-envs/<family>`, using `--system-site-packages` when sharing the base CUDA/PyTorch;
   - install the lock and the source;
   - add path aliases if the family and checkout names differ;
   - write `/opt/rpx-envs/<family>/rpx-environment.json`;
   - run the imports through `/opt/rpx-envs/<family>/bin/python`.

   Without `rpx-environment.json` the matrix will not run the model. If the family changes benchmark code, copy
   `benchmark` into the stage and reinstall the package so the image doesn't keep a stale adapter.

6. Update the defaults in `entrypoint.sh`, `compose.yaml` and the tables on this page once the model is
   matrix-ready. Keep image and video selections separate, since `entrypoint.sh smoke` runs image models only.

7. Test in widening scope. Start with `pytest -q benchmark/tests/test_depth_smoke_tools.py`. Then:
   1. build the stage, which also runs its import check;
   2. run `verify`;
   3. run the micro gate;
   4. run acceptance;
   5. after review, run the Easy tier.

   Installation, imports, CPU inference, an OOM, a quantized diagnostic run or a micro pass alone does not make a model
   complete.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is false | GPU passthrough. Fix the driver or Container Toolkit; never fall back to CPU |
| GPU already has processes | Pick an idle GPU with `--gpu-index`; don't overlap acceptance runs |
| HTTP 401/403, gated repository | Request access with the same Hugging Face account; don't substitute checkpoints |
| CUDA OOM | Rerun the same command on a larger GPU |
| `No module named ...` | Add the dependency or source path to the family and rebuild |
| `has no attribute`, `from_pretrained` fails | Align the adapter with the pinned upstream API and checkpoint |
| Wrong shape or dtype | Return float32 `(H, W)` for images or `(T, H, W)` for video |
| Matrix reports `missing_environment` | Check that `<family>/bin/python` and `<family>/rpx-environment.json` exist under `/opt/rpx-envs` |
| A rebuilt image reruns passed gates | Expected: state is keyed by exact code identity |

To benchmark your own model, see the [benchmark README](../../benchmark/README.md).

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="../README.md">Model environments</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a></sub>
