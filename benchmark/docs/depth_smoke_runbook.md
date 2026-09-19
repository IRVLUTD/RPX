# Depth smoke validation and Easy pilot

This is the from-scratch operational companion to `adapter_status.md`. Every
command is GPU-only and pins dataset revision
`2e2a387f7f93e98c177b2e039c141eacda94e5fc`. The launcher refuses CPU fallback,
an occupied selected GPU and the unverified FE2E adapter.
It never uploads results.

## 1. Host and storage setup

Use a large local disk. The environments, checkpoints, extracted Easy split and
saved image predictions can collectively require well over 100 GB.

```bash
nvidia-smi
python3.11 --version

export RPX_WORK_ROOT=/data/shared                 # choose a large disk
export HF_HOME="$RPX_WORK_ROOT/hf-cache"           # shared data/weight cache
export RPX_SMOKE_ROOT="$RPX_WORK_ROOT/rpx-smoke"   # immutable run artefacts
mkdir -p "$HF_HOME" "$RPX_SMOKE_ROOT" "$RPX_WORK_ROOT/rpx-envs"

hf auth login   # required for gated checkpoints such as FLUX.1-dev
```

Clone or pull the reviewed RPX ref, then confirm the worktree is clean. Never
benchmark a silently modified checkout; the launcher records and hashes dirty
code, but that result is diagnostic only.

```bash
git clone https://github.com/IRVLUTD/RPX.git rpx
cd rpx
git checkout <reviewed-depth-smoke-ref>
git status --short
```

## 2. Environment boundaries

Use Python 3.11 and one environment per incompatible upstream family. The setup
helper installs CUDA-enabled PyTorch, the RPX package, exact upstream Git SHAs,
and records `pip-freeze.txt` plus `rpx-environment.json` inside the environment.
The defaults use PyTorch 2.10 CUDA 12.8 because that supports the RTX 5060
(sm_120). On a server whose NVIDIA driver cannot support CUDA 12.8, pass a
driver-compatible `--torch-version`, `--torchvision-version` and
`--torch-index`; never replace the CUDA wheel with a CPU wheel.

| Environment | Canonical models |
| --- | --- |
| `da3` | `da3-metric-l`, `da3-video` |
| `transformers-image` | `da-v2-large`, `depth-pro` |
| `metadepth` | `hyden` |
| `depthlm` | `depthlm` |
| `lotus2` | `lotus-2` |
| `metric3d` | `metric3d-v2` |
| `moge2` | `moge-2-vit-l` |
| `unidepth2` | `unidepth-v2` |
| `chrono` | `chrono-depth` |
| `depthcrafter` | `depth-crafter` |
| `monst3r` | `monst3r` |
| `rolling` | `rolling-depth` |
| `vggt` | `vggt-omega` |
| `video_da` | `video-da` |
| `vigeo` | `vigeo` |

Create an environment by canonical model name. Models in a shared family reuse
the same directory, so running the helper twice is safe.

```bash
python3.11 benchmark/scripts/setup_depth_smoke_env.py \
  --model depth-pro --env-root "$RPX_WORK_ROOT/rpx-envs"

# Repeat for each pending canonical model. Example for a video model:
python3.11 benchmark/scripts/setup_depth_smoke_env.py \
  --model video-da --env-root "$RPX_WORK_ROOT/rpx-envs"
```

FE2E is deliberately absent from the setup choices. GemDepth uses its
official-runtime overlay under ``docker/depth-gemdepth``; production DVD runs
use ``docker/depth-dvd`` (the setup helper also supports a host diagnostic).
The smoke launcher refuses the unverified FE2E model and never supplies
`--acknowledge-unverified`.

The setup helper refuses to start below 30 GiB free space and warns below
150 GiB. An interrupted environment is not considered ready until it contains
both `bin/python` and `rpx-environment.json`; rerunning the same setup command
resumes normal pip/Git installation and repeats the import/CUDA checks. Pip
temporary files and its cache are redirected under `--env-root`, avoiding small
system `/tmp` mounts. The helper also keeps one shared pinned CUDA/PyTorch
wheelhouse there, so later family environments reuse the large wheels without
redownloading or retaining a duplicate pip-cache copy.

The matrix persists setup attempts by upstream family. After a failed first
attempt, correct the recipe or dependency issue and add `--retry-failed`; a
second failure reaches the setup ceiling and requires a new reviewed code
identity rather than another automatic retry.

## 3. Gates

Run the launcher with the environment's Python. There is no need to activate
the environment. Pass its pinned source checkout with `--upstream-dir` when the
family has one; the exact SHA is then copied into the run metadata.

```bash
export ENV="$RPX_WORK_ROOT/rpx-envs/transformers-image"

# One image
"$ENV/bin/python" benchmark/scripts/run_depth_smoke_gate.py \
  --task image --model <model> --gate micro \
  --output-root "$RPX_SMOKE_ROOT"

# 25 images
"$ENV/bin/python" benchmark/scripts/run_depth_smoke_gate.py \
  --task image --model <model> --gate acceptance \
  --output-root "$RPX_SMOKE_ROOT"

# One clip, first with 8 frames and then with 25 frames
"$ENV/bin/python" benchmark/scripts/run_depth_smoke_gate.py \
  --task video --model <model> --gate micro \
  --upstream-dir "$RPX_WORK_ROOT/rpx-envs/sources/<family>"
"$ENV/bin/python" benchmark/scripts/run_depth_smoke_gate.py \
  --task video --model <model> --gate acceptance \
  --upstream-dir "$RPX_WORK_ROOT/rpx-envs/sources/<family>"
```

The micro gate means one image or one eight-frame clip. Acceptance means 25
images or one 25-frame clip. Only the selected GPU is checked for occupancy;
choose another with `--gpu-index N`. The launcher validates `result.json`,
`cells.parquet`, `summary.md`, exact sample counts, finite metrics and SystemCard
fields. Image smokes also save and inspect a raw finite, non-degenerate
prediction. Standard Hugging Face HTTP is used because Xet is less portable;
opt in with `--enable-xet` only on a host where it is known to work.

On an 8 GB RTX 5060, first try Depth Pro, HyDen, Lotus-2 and MoGe-2. Escalate an
unchanged command that reports CUDA OOM to the high-memory server. An 8 GB OOM
is a hardware classification, not an adapter failure. DepthLM must pass in its
intended FP16 configuration; a 4-bit result is diagnostic only. The other large
image models and all video models should normally be gated on the high-memory
host.

Every run lands under
`$RPX_SMOKE_ROOT/<code-id>/<host>/<model>/<gate>/<UTC timestamp>/`. Keep failed
runs: `run_metadata.json` classifies dependency, download, API, shape/data, OOM
or code failures and `run.log` contains the evidence needed for the fix loop.

### Sequential roster driver

The matrix driver invokes those same per-model gates one at a time. It resumes
passed gates, limits code/dependency failures to two attempts, never retries an
OOM or access failure, and writes one consolidated status JSON per code
identity.

```bash
# Confirm all 20 canonical rows (17 runnable, 3 blocked).
python benchmark/scripts/run_depth_smoke_matrix.py --list-models

# RTX 5060: build and test only the plausible 8 GB image candidates.
python benchmark/scripts/run_depth_smoke_matrix.py \
  --task image \
  --models depth-pro,hyden,moge-2-vit-l,unidepth-v2,lotus-2 \
  --gates micro,acceptance \
  --setup-missing \
  --setup-python python3.11 \
  --env-root "$RPX_WORK_ROOT/rpx-envs" \
  --cache-dir "$HF_HOME" \
  --output-root "$RPX_SMOKE_ROOT"

# High-memory server: prepare and gate every canonical row sequentially.
python benchmark/scripts/run_depth_smoke_matrix.py \
  --task all --gates micro,acceptance --setup-missing \
  --env-root "$RPX_WORK_ROOT/rpx-envs" \
  --cache-dir "$HF_HOME" \
  --output-root "$RPX_SMOKE_ROOT"
```

Use `--retry-failed` only after fixing a dependency/API/code defect. The driver
still enforces a one-attempt ceiling for `access`, `cuda_oom` and
`weight_download`. Use `--rerun-passed` only when deliberately repeating an
already-passed gate on the exact same code identity.

## 4. Validation snapshot and server handoff

The 2026-07-02 RTX 5060 (8 GB) run against the pinned dataset revision produced
these acceptance results. These are smoke metrics over 25 Easy frames, not the
publishable full-Easy benchmark:

| Model | Result | RMSE | AbsRel | delta1 | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `depth-pro` | 25 / 25 passed | 0.5981 | 0.1390 | 0.9092 | 1095.5 ms |
| `unidepth-v2` | 25 / 25 passed | 0.4978 | 0.0707 | 0.9479 | 134.4 ms |
| `moge-2-vit-l` | 25 / 25 passed | 0.6235 | 0.1769 | 0.7502 | 373.1 ms |

DA-V2 Large had already passed the team's real-GPU acceptance gate, bringing
the current total to 4/20 canonical rows (4/17 runnable rows). HyDen's setup,
import and CUDA checks pass, but its official gated checkpoint returned HTTP
403 for the current Hugging Face account; request access and retry once without
changing the checkpoint. Lotus-2 exposed an incompatible Transformers 5
resolution; the setup recipe now pins Transformers 4.46.3 for Diffusers 0.32.2.
Its two-attempt local ceiling was reached, so the corrected recipe must be
confirmed on the server. No checkpoint substitution or CPU fallback is valid.

On a clean checkout of the reviewed smoke branch, the high-memory server can
resume all unresolved rows with this exact command:

```bash
export RPX_WORK_ROOT=/data/shared
export HF_HOME="$RPX_WORK_ROOT/hf-cache"
export RPX_SMOKE_ROOT="$RPX_WORK_ROOT/rpx-smoke"

python3.11 benchmark/scripts/run_depth_smoke_matrix.py \
  --task all --gates micro,acceptance --setup-missing \
  --setup-python python3.11 \
  --env-root "$RPX_WORK_ROOT/rpx-envs" \
  --cache-dir "$HF_HOME" \
  --output-root "$RPX_SMOKE_ROOT"
```

The matrix skips already-passed gates only when they are recorded under that
exact code identity. Copying an old matrix JSON into a new identity is not a
valid shortcut. Smoke output stays local; this workflow neither changes the
source dataset nor uploads results.

## 5. Easy pilot

Only after every applicable acceptance gate passes and the smoke PR is
reviewed:

```bash
# All 24,750 Easy image frames, batch size 1, predictions + full metrics.
"$ENV/bin/python" benchmark/scripts/run_depth_smoke_gate.py \
  --task image --model <passed-image-model> --gate easy \
  --output-root "$RPX_SMOKE_ROOT" --upstream-dir /path/to/upstream

# All 99 Easy clips, all frames.
"$ENV/bin/python" benchmark/scripts/run_depth_smoke_gate.py \
  --task video --model <passed-video-model> --gate easy \
  --output-root "$RPX_SMOKE_ROOT" --upstream-dir /path/to/upstream
```

The Easy image gate requires exactly 24,750 frames, batch size 1, saved raw
predictions and comprehensive metrics. The Easy video gate requires exactly 99
clips and uses every frame (`sampling=all`). Use one common high-memory GPU for
comparable latency. Review the generated metrics and counts before any separate
Box upload. Medium and Hard remain out of scope until the Easy pilot is
reviewed.

Once the same matrix state shows acceptance passed for all 17 runnable rows,
the server can execute the full pilot sequentially:

```bash
python benchmark/scripts/run_depth_smoke_matrix.py \
  --task all --gates easy \
  --env-root "$RPX_WORK_ROOT/rpx-envs" \
  --cache-dir "$HF_HOME" \
  --output-root "$RPX_SMOKE_ROOT"
```
