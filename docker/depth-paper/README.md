# DA-V2 Large RPX paper run

This is a thin code overlay on the immutable cumulative HyDen image. It does
not contain data, model weights or results. Keep both mounted directories on
persistent server storage.

```bash
cd RPX
export RPX_GIT_SHA="$(git rev-parse HEAD)"
export IMAGE="rpx-depth-paper:${RPX_GIT_SHA:0:12}"
export HF_HOME="$HOME/.cache/huggingface"
export RPX_OUTPUT="$HOME/rpx-depth-paper"
mkdir -p "$HF_HOME" "$RPX_OUTPUT"

docker build --file docker/depth-paper/Dockerfile \
  --build-arg "RPX_GIT_SHA=$RPX_GIT_SHA" --tag "$IMAGE" .
```

The preflight checks CUDA, imports and free space; it does not run a smoke
inference. Start the complete 75,000-frame job in `tmux`:

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

Detach with `Ctrl-b d`; reattach with `tmux attach -t rpx-depth`. If the run is
interrupted, execute the identical Docker command. Every NPZ is CRC-loaded and
validated before reuse; only missing or invalid entries are inferred.

After the cache has completed and been validated once, the same command can be
made network-independent by adding `--offline` and omitting `HF_TOKEN`. This
sets `HF_HUB_OFFLINE=1` for manifest resolution and all child runs.

Final validation:

```bash
find "$RPX_OUTPUT/da-v2-large"/easy/predictions \
     "$RPX_OUTPUT/da-v2-large"/medium/predictions \
     "$RPX_OUTPUT/da-v2-large"/hard/predictions -name '*.npz' | wc -l

docker run --rm -i -v "$RPX_OUTPUT:/outputs" "$IMAGE" python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/outputs/da-v2-large")
assert sum(pq.read_metadata(root / s / "per_sample_metrics.parquet").num_rows
           for s in ("easy", "medium", "hard")) == 75_000
analysis = json.loads((root / "paper" / "paper_analysis.json").read_text())
assert analysis["n_scenes"] == 100 and analysis["n_cells"] == 300
assert analysis["jedi"]["status"] in {"bounds_missing", "computed"}
latency = json.loads((root / "latency.json").read_text())
assert latency["measured_forwards"] == 100 and latency["prediction_writes"] is False
print("validated", analysis["jedi"]["status"])
PY
```

To add JEDI later, mount a reviewed versioned bounds JSON and rerun only:

```bash
docker run --rm \
  -v "$RPX_OUTPUT:/outputs" -v "$PWD/bounds:/bounds:ro" \
  "$IMAGE" python scripts/analyze_depth_paper.py \
  /outputs/da-v2-large/{easy,medium,hard}/cells.parquet \
  --output-dir /outputs/da-v2-large/paper \
  --jedi-bounds /bounds/d1-jedi-v1.json
```
