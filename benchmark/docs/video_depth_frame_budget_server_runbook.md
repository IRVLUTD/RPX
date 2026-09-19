# RPX Video-Depth Frame-Budget Server Runbook

This is the operator runbook for evaluating the ten D1-V models at
50, 100, 150, 200, and 250 frames per RPX scene-phase clip. It uses the
pinned RPX dataset, uniform stride sampling, saved prediction containers,
and validated resume.

The frame-budget experiment asks whether the video model's spatial and
temporal metrics change as fewer uniformly distributed frames are available.
Keep the model, split, dataset revision, sampling method, and evaluation code
fixed. Only the frame budget changes.

## Fixed protocol

- Dataset: `IRVLUTD/RPX`
- Revision: `2e2a387f7f93e98c177b2e039c141eacda94e5fc`
- Splits: `easy`, `medium`, `hard`
- Expected scene-phase clips: 99, 99, and 102 (300 total)
- Budgets: 50, 100, 150, 200, and 250 frames
- Sampling: `stride`
- Prediction persistence: enabled
- Resume validation: enabled
- F-score: disabled (do not pass `--compute-fscore`)

`stride` samples uniformly over the original 250-frame clip with endpoint
anchoring. It is not equivalent to taking only the first N frames. Use this
same policy for every model and budget.

The runner writes one NPZ per scene-phase clip, not one NPZ per frame:

```text
predictions/<scene>/<phase>/depth.npz
```

Each NPZ records the predicted depth sequence and the exact selected frame
indices.

## 1. Enter the pinned checkout and export paths

Run these commands in every new SSH shell or tmux session:

```bash
cd /data/narendhiran_rpx/src/RPX-video-temporal-10763af

export HF_HOME="/data/narendhiran_rpx/hf-cache"
export RPX_OUTPUT="/data/narendhiran_rpx/docker-smoke/depth-paper-outputs"
export DATASET_REVISION="2e2a387f7f93e98c177b2e039c141eacda94e5fc"
export BUDGET_ROOT="$RPX_OUTPUT/video-depth-budget-sweep"

test -d "$HF_HOME" || { echo "Missing HF cache: $HF_HOME"; exit 1; }
mkdir -p "$BUDGET_ROOT" "$RPX_OUTPUT/logs/video-budget"
test -w "$RPX_OUTPUT" || { echo "Output is not writable: $RPX_OUTPUT"; exit 1; }

test -n "${HF_TOKEN:-}" || {
  read -rsp "Hugging Face token: " HF_TOKEN
  echo
  export HF_TOKEN
}
```

Never continue if `HF_HOME` or `RPX_OUTPUT` is empty. An empty variable causes
Docker errors such as `invalid spec: :/cache/huggingface`.

## 2. Select the immutable local images

These are the images used by the completed production runs on `cs94092`:

```bash
export VIDEO_IMAGE="rpx-video-depth:10763af29acb"
export GEM_IMAGE="rpx-video-gemdepth:fed54de7f09e"
export DVD_IMAGE="rpx-video-dvd:66c058783d69"

for image in "$VIDEO_IMAGE" "$GEM_IMAGE" "$DVD_IMAGE"; do
  docker image inspect "$image" >/dev/null || {
    echo "Missing Docker image: $image"
    exit 1
  }
done
```

Do not silently replace these with `latest`. If an image must change, record
the new tag and digest with the resulting metrics.

## 3. Ten-model execution map

| Model key | Image variable | Python inside image |
|---|---|---|
| `video-da` | `VIDEO_IMAGE` | `/opt/rpx-envs/video_da/bin/python` |
| `da3-video` | `VIDEO_IMAGE` | `/opt/rpx-envs/da3/bin/python` |
| `vggt-omega` | `VIDEO_IMAGE` | `/opt/rpx-envs/geometry-video/bin/python` |
| `depth-crafter` | `VIDEO_IMAGE` | `/opt/rpx-envs/depthcrafter/bin/python` |
| `monst3r` | `VIDEO_IMAGE` | `/opt/rpx-envs/monst3r/bin/python` |
| `vigeo` | `VIDEO_IMAGE` | `/opt/rpx-envs/geometry-video/bin/python` |
| `rolling-depth` | `VIDEO_IMAGE` | `/opt/rpx-envs/rollingdepth/bin/python` |
| `chrono-depth` | `VIDEO_IMAGE` | `/opt/rpx-envs/chrono/bin/python` |
| `gem-depth` | `GEM_IMAGE` | `/opt/rpx-envs/gemdepth/bin/python` |
| `dvd` | `DVD_IMAGE` | `/opt/rpx-envs/dvd/bin/python` |

The table is the source of truth for choosing an image and interpreter. Do
not run a dedicated adapter with `/usr/local/bin/python`.

## 4. Preflight the GPUs, cache, images, and storage

```bash
nvidia-smi
df -h "$HF_HOME" "$RPX_OUTPUT"

for image in "$VIDEO_IMAGE" "$GEM_IMAGE" "$DVD_IMAGE"; do
  printf '%s  ' "$image"
  docker image inspect --format 'id={{.Id}} size={{.Size}}' "$image"
done

find "$HF_HOME/rpx-resolved/IRVLUTD__RPX/manifests/video_depth" \
  -maxdepth 1 -type f -name '*.json' -print | sort
```

If another user is using a GPU, choose another GPU. Parallel models may share
the HF cache and output filesystem, but never write the same
`model/split/budget` directory concurrently.

## 5. Define the reusable launcher

Paste this once into the shell that will launch jobs:

```bash
video_model_runtime() {
  case "$1" in
    video-da)       printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/video_da/bin/python ;;
    da3-video)      printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/da3/bin/python ;;
    vggt-omega)     printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/geometry-video/bin/python ;;
    depth-crafter)  printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/depthcrafter/bin/python ;;
    monst3r)        printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/monst3r/bin/python ;;
    vigeo)          printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/geometry-video/bin/python ;;
    rolling-depth)  printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/rollingdepth/bin/python ;;
    chrono-depth)   printf '%s\t%s\n' "$VIDEO_IMAGE" /opt/rpx-envs/chrono/bin/python ;;
    gem-depth)      printf '%s\t%s\n' "$GEM_IMAGE" /opt/rpx-envs/gemdepth/bin/python ;;
    dvd)            printf '%s\t%s\n' "$DVD_IMAGE" /opt/rpx-envs/dvd/bin/python ;;
    *) echo "Unknown model: $1" >&2; return 2 ;;
  esac
}

run_video_budget() {
  model="$1"
  budget="$2"
  gpu="$3"

  runtime=$(video_model_runtime "$model") || return
  image=$(printf '%s\n' "$runtime" | cut -f1)
  python_bin=$(printf '%s\n' "$runtime" | cut -f2)

  for split in easy medium hard; do
    output="/outputs/video-depth-budget${budget}/${model}/${split}"
    name="rpx-${model}-b${budget}-${split}"

    echo "=================================================="
    echo "model=$model budget=$budget split=$split gpu=$gpu"
    echo "image=$image"
    echo "output=$output"
    echo "=================================================="

    docker rm -f "$name" >/dev/null 2>&1 || true

    docker run --rm \
      --name "$name" \
      --gpus "device=${gpu}" \
      --ipc=host \
      --shm-size=32g \
      -e HF_TOKEN \
      -e HF_HOME=/cache/huggingface \
      -e HF_HUB_DISABLE_XET=1 \
      -e PYTHONUNBUFFERED=1 \
      -v "$HF_HOME:/cache/huggingface" \
      -v "$RPX_OUTPUT:/outputs" \
      "$image" \
      "$python_bin" scripts/run_video_depth.py \
        --model "$model" \
        --split "$split" \
        --revision "$DATASET_REVISION" \
        --device cuda \
        --frame-budget "$budget" \
        --sampling stride \
        --save-predictions \
        --resume-predictions \
        --output-dir "$output"

    status=$?
    if test "$status" -ne 0; then
      echo "FAILED: model=$model budget=$budget split=$split status=$status"
      return "$status"
    fi
  done

  echo "COMPLETE: model=$model budget=$budget"
}
```

The function runs Easy, Medium, and Hard sequentially on one GPU. A rerun uses
validated prediction containers and only recomputes missing or corrupt clips.

### Run one missing 50-frame model

```bash
run_video_budget da3-video 50 0 2>&1 | \
  tee "$RPX_OUTPUT/logs/video-budget/da3-video-b50.log"
```

Use GPU 2 by changing only the final argument:

```bash
run_video_budget depth-crafter 50 2 2>&1 | \
  tee "$RPX_OUTPUT/logs/video-budget/depth-crafter-b50.log"
```

## 6. Preferred full five-budget sweep

When starting a model from scratch, `--budget-sweep` is more efficient because
the adapter is loaded once and reused for every budget. It stores each budget
under `budget_<N>`:

```bash
run_video_sweep() {
  model="$1"
  gpu="$2"

  runtime=$(video_model_runtime "$model") || return
  image=$(printf '%s\n' "$runtime" | cut -f1)
  python_bin=$(printf '%s\n' "$runtime" | cut -f2)

  for split in easy medium hard; do
    name="rpx-${model}-sweep-${split}"

    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run --rm \
      --name "$name" \
      --gpus "device=${gpu}" \
      --ipc=host \
      --shm-size=32g \
      -e HF_TOKEN \
      -e HF_HOME=/cache/huggingface \
      -e HF_HUB_DISABLE_XET=1 \
      -e PYTHONUNBUFFERED=1 \
      -v "$HF_HOME:/cache/huggingface" \
      -v "$RPX_OUTPUT:/outputs" \
      "$image" \
      "$python_bin" scripts/run_video_depth.py \
        --model "$model" \
        --split "$split" \
        --revision "$DATASET_REVISION" \
        --device cuda \
        --budget-sweep 50,100,150,200,250 \
        --sampling stride \
        --save-predictions \
        --resume-predictions \
        --output-dir "/outputs/video-depth-budget-sweep/${model}/${split}" || return
  done
}
```

Do not run `run_video_sweep` and `run_video_budget` against the same output
directory; their directory layouts intentionally differ.

## 7. Run safely in tmux

Create one session per active GPU. The left pane runs inference; the upper
right pane monitors GPU use; the lower right pane monitors completed clips.

```bash
tmux new-session -s rpx-vdbudget-gpu0 \
  -c /data/narendhiran_rpx/src/RPX-video-temporal-10763af
```

Inside tmux, create the monitoring panes:

```bash
tmux split-window -h 'watch -n 0.5 nvidia-smi -i 0'
tmux split-window -v \
  'watch -n 20 '\''find /data/narendhiran_rpx/docker-smoke/depth-paper-outputs -path "*/video-depth-budget*/predictions/*/*/depth.npz" -type f 2>/dev/null | wc -l'\'''
tmux select-pane -t 0
```

Paste the exports and function from §§1, 2, and 5 into the left pane, then run
one model. Detach with `Ctrl-b d`; reattach with:

```bash
tmux attach -t rpx-vdbudget-gpu0
```

For GPU 2, use a separate session and change both the session name and
`nvidia-smi -i 2`.

## 8. Validate one completed model-budget

This checks clip counts and required result artifacts:

```bash
validate_video_budget() {
  model="$1"
  budget="$2"
  root="$RPX_OUTPUT/video-depth-budget${budget}/${model}"
  failed=0

  for item in easy:99 medium:99 hard:102; do
    split=${item%:*}
    expected=${item#*:}
    dir="$root/$split"
    count=$(find "$dir/predictions" -type f -name depth.npz 2>/dev/null | wc -l)
    result=MISSING
    cells=MISSING
    test -s "$dir/result.json" && result=PRESENT
    test -s "$dir/cells.parquet" && cells=PRESENT

    printf '%-7s clips=%3s/%3s result=%s cells=%s\n' \
      "$split" "$count" "$expected" "$result" "$cells"

    test "$count" -eq "$expected" || failed=1
    test "$result" = PRESENT || failed=1
    test "$cells" = PRESENT || failed=1
  done

  test "$failed" -eq 0
}

validate_video_budget video-da 50
validate_video_budget gem-depth 50
```

Exit status 0 means the structural acceptance check passed. This does not
replace scientific review of metric values.

## 9. Audit all models and budgets

```bash
models='video-da da3-video vggt-omega depth-crafter monst3r vigeo gem-depth dvd rolling-depth chrono-depth'

for budget in 50 100 150 200 250; do
  for model in $models; do
    printf '%-14s budget=%3s  ' "$model" "$budget"
    if validate_video_budget "$model" "$budget" >/dev/null 2>&1; then
      echo PASS
    else
      echo INCOMPLETE
    fi
  done
done
```

An existing full 250-frame result under `$RPX_OUTPUT/video-depth/<model>` is
not automatically the same directory as the explicit budget-250 ablation.
It may be reused analytically only after `run_metadata.json` confirms the same
250-frame, `stride` protocol. Otherwise run the explicit budget-250 job.

## 10. Recovery rules

- If SSH disconnects but the tmux session remains, do nothing; reattach.
- If a container stops, rerun the identical command. Valid clips are cache
  hits; missing/corrupt clips are recomputed.
- Never delete a whole model directory merely because one split failed.
- Never run two writers against the same output directory.
- Do not add `--compute-fscore` to these ablation runs.
- Do not mix `stride` and `fps_se3` results in one comparison.
- Preserve `run_metadata.json`, `result.json`, `cells.parquet`, `summary.md`,
  and `predictions/` for every split and budget.

## 11. Metrics and downstream analysis

The authoritative cell logs are `cells.parquet`. The primary D1-V vector is:

```text
absrel, rmse, delta1, silog, tgm, tgse
```

Budget degradation analysis (TCV, AUDC, critical budget, and paired tests) is
post-processing over the completed cell logs. Do not infer degradation from
console summaries alone, and do not combine budgets until every compared model
has the same complete scene-phase cells.

