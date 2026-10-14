<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/" aria-current="page"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>VQA</a>
<a href="../t6/"><span>T6</span>In-context VQA</a>
</div>

<p class="rpx-task-kicker">T2 WALKTHROUGH</p>

# Video depth

<div class="rpx-task-lead" markdown="1">
Takes an ordered `T × H × W × 3` RGB clip from one phase and returns a `T × H × W` depth sequence.
</div>

<div class="rpx-task-shortcuts"><a href="#run-the-benchmark">Run it <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Use local data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene049-rgb.png" alt="FRAME 00185: RGB clip · preserve frame order" width="640" height="480"><figcaption><span>FRAME 00185</span>RGB clip · preserve frame order</figcaption></figure>
<figure><img src="../../assets/scene049-next.png" alt="FRAME 00186: Adjacent frame · same capture" width="640" height="480"><figcaption><span>FRAME 00186</span>Adjacent frame · same capture</figcaption></figure>
</div>
<p class="rpx-image-note">Scene 049, Interaction, frames 00185 and 00186. Real clips are longer; their length comes from the manifest.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Ordered RGB clip</strong></div>
<div><span>PREDICTION</span><strong>Depth sequence</strong></div>
<div><span>EVALUATION</span><strong>Depth + temporal accuracy</strong></div>
</div>

## Connect your model

Put your prediction function in `my_model.py`. Load the model or API client once, at module level, rather than on every call.

```python title="my_model.py"
import numpy as np

def predict_video_depth(rgb):
    return np.asarray(model.predict_clip(rgb), dtype=np.float32)
```

<div class="rpx-task-note" markdown="1">
Return one depth map per input frame, in the order you received them. A constant map is rejected as degenerate.
</div>

## Run the benchmark

Install the toolkit with `pip install 'rpx-benchmark[hub,schemas]'`. The smoke test checks the install; the second tab scores your model.

=== "Smoke test (offline)"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T2 --smoke --output results/t2-smoke
    ```

    This runs the real evaluator on a few synthetic inputs with a placeholder model. Nothing is downloaded and no weights are needed. The output directory must be new or empty.


=== "Your model on RPX data"

    Run this from the directory that contains `my_model.py`. Without `--manifest`, the command downloads the data it needs first.

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T2 \
      --model my_model:predict_video_depth \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T2 \
      --device cuda \
      --split easy \
      --max-samples 10
    ```


    Ten samples are enough to check the wiring; drop `--max-samples` to score the whole split. To use data you already have on disk, add `--manifest path/to/manifest.json` (see [using local data](../README.md#move-to-real-data)).


## Read the results

The output directory holds `result.json`, `summary.md`, per-sample tables and cell logs. `run_config.json` records the model revision and the manifest hash. From Python, the `run_*` functions return `(result, report, paths)`.

Check that the number of scored samples matches what you asked for. Keep the per-sample records: [Φ and 𝒥<sub>min</sub>](../../analysis/README.md) are computed from them. Latency and memory are measured separately, with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

Your function must keep the input frame order and return the same `T`. Pass `--depth-output-kind relative` only for up-to-scale predictions; those get a per-clip scale and shift fit before scoring. Compare models with the same clip length, frame sampling and alignment.

Scores combine per-frame depth accuracy with two temporal metrics, `tgm` and `tgse`. The paper's metric vector leaves out the pose- and flow-dependent diagnostics. Here `--max-samples` counts clips, not frames. `VideoDepthRunConfig` has frame-budget and sampling options for ablations.

??? note "Before a full run"

    Check units, shapes, input order and sample IDs on a small run first. Pin the model revision, and keep the split and preprocessing the same for every model you compare. The smoke test only exercises the evaluator; your model still needs its own environment and weights.

??? info "Workflow figure"

    [T2 workflow (SVG)](../../assets/benchmark-t2.svg): the inputs, adapter and scoring stages for this task.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [The paper's models](https://github.com/IRVLUTD/RPX/tree/main/docker)
</div>

</div>
