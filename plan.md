# DA-V2 Large End-to-End RPX Paper Pipeline

## Goal

Run `da-v2-large` over the pinned RPX D1-F dataset (75,000 frames), resume
validated predictions after interruption, compute the six headline depth
metrics, aggregate exactly 300 scene-phase cells, and generate the paper's
phase-robustness analysis. Supplementary metrics are intentionally excluded.

## Fixed protocol

- Dataset: `IRVLUTD/RPX` at
  `2e2a387f7f93e98c177b2e039c141eacda94e5fc`.
- Model: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`, metric
  output, no GT alignment, batch size 1, explicit FP16 weights on CUDA.
- Calibration: paper-declared RGB intrinsics at 640x480:
  `fx=fy=615`, `cx=320`, `cy=240`. The release has no per-device calibration
  artifact. D1-F needs no extrinsics because captured depth is RGB-aligned.
  This follows the [RealSense alignment example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)
  and [projection convention](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0).
- Valid GT: finite and strictly `0.3 < depth < 5.0` metres. Predictions must
  be finite and positive on every evaluated pixel.
- Metrics: AbsRel, RMSE, SILog, delta1, iRMSE and bidirectional point-cloud
  F-Score at a strict distance threshold of `< 0.05 m`, following the
  [Tanks and Temples definition](https://www.tanksandtemples.org/tutorial/).
- JEDI: implement the evaluator, but publish no numeric J without a versioned
  bounds file containing explicit bounds, directions, epsilon and provenance.

## Implementation checklist

- [x] Document the active Hugging Face path: manifest fetch, RGB/depth-only
      snapshot pull, cache reuse, checksum verification, idempotent extraction,
      and resolved local manifest.
- [x] Expose an explicit dataset cache directory and validate split contracts:
      Easy 24,750/99, Medium 24,750/99, Hard 25,500/102.
- [x] Save compressed float32 predictions atomically under
      `predictions/<scene>/<phase>/<frame>.npz`.
- [x] Add validated prediction resume; infer only missing/corrupt entries and
      record cache-hit/new/recomputed counts.
- [x] Keep prediction persistence outside measured model latency and remove the
      second full-dataset timing pass.
- [x] Compute the six metrics per frame, then average them per scene-phase.
- [x] Write `per_sample_metrics.parquet`, `cells.parquet`, `result.json`,
      `run_metadata.json` and logs for every split.
- [x] Merge exactly 100 scenes x 3 phases and compute repeated-measures Wilks
      Phi, Pillai diagnostic, paired Hotelling transitions with Holm correction,
      per-tier Phi and phase-by-difficulty interaction.
- [x] Emit combined cells, JSON/Markdown/CSV paper outputs and complete
      provenance. Emit `jedi_status: bounds_missing` when bounds are absent.
- [x] Add one resumable production launcher and a thin Docker overlay on the
      immutable cumulative image digest.
- [x] Cover dataset, resume, metrics, statistics, Docker imports and synthetic
      end-to-end behavior with tests.

## Server run sequence

The completed implementation must support this order without manual dataset
conversion:

1. Mount the existing Hugging Face cache at `/cache/huggingface` and persistent
   results at `/outputs`.
2. Run the CUDA/import preflight (not a smoke inference run).
3. Launch the production command in `tmux`; it resolves Easy, Medium and Hard,
   downloading only missing RGB/depth shards.
4. Re-run the identical command after interruption. Valid predictions are
   loaded; only missing/invalid predictions are inferred.
5. Verify 75,000 predictions, 75,000 metric rows, 300 complete cells and all
   non-JEDI paper statistics.
6. Supply a reviewed bounds file later to recompute JEDI from cell logs without
   rerunning the model.

## Copy-paste server commands

From the checked-out repository on the GPU server:

```bash
export RPX_GIT_SHA="$(git rev-parse HEAD)"
export IMAGE="rpx-depth-paper:${RPX_GIT_SHA:0:12}"
export HF_HOME="$HOME/.cache/huggingface"
export RPX_OUTPUT="$HOME/rpx-depth-paper"
mkdir -p "$HF_HOME" "$RPX_OUTPUT"

docker build --file docker/depth-paper/Dockerfile \
  --build-arg "RPX_GIT_SHA=$RPX_GIT_SHA" \
  --tag "$IMAGE" .
```

The overlay is pinned to cumulative image digest
`sha256:3b86de0e4e136587d6ea7d816a482af15cb115dc6a14f16ea8f7d9356ab220cc`.
Start the run in `tmux`:

```bash
tmux new -s rpx-depth
docker run --rm --gpus all --ipc=host --shm-size=8g \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_OUTPUT:/outputs" \
  "$IMAGE" python scripts/run_depth_paper.py \
    --cache-dir /cache/huggingface \
    --output-root /outputs/da-v2-large
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t rpx-depth`. Resume by
running the exact same Docker command. Once the three cached splits have been
successfully resolved, add `--offline` and omit `-e HF_TOKEN` to enforce
`HF_HUB_OFFLINE=1`.

Validate final counts on the host:

```bash
find "$RPX_OUTPUT/da-v2-large"/{easy,medium,hard}/predictions \
  -name '*.npz' | wc -l

docker run --rm -i -v "$RPX_OUTPUT:/outputs" "$IMAGE" python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/outputs/da-v2-large')
assert sum(pq.read_metadata(root / tier / 'per_sample_metrics.parquet').num_rows
           for tier in ('easy', 'medium', 'hard')) == 75_000
analysis = json.loads((root / 'paper' / 'paper_analysis.json').read_text())
assert analysis['n_cells'] == 300 and analysis['n_scenes'] == 100
assert analysis['jedi_status'] in {'bounds_missing', 'computed'}
print('RPX D1-F output validated')
PY
```

## Acceptance

- A second completed run performs zero DA-V2 forwards.
- No downsampling, masks, poses, optical flow or extrinsics enter D1-F scoring.
- No calibration, JEDI bound or epsilon is guessed.
- Missing, duplicate, non-finite or singular analysis inputs fail explicitly.
