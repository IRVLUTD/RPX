<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/" aria-current="page"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>VQA</a>
<a href="../t6/"><span>T6</span>In-context VQA</a>
</div>

<p class="rpx-task-kicker">T1 WALKTHROUGH</p>

# Image depth

<div class="rpx-task-lead" markdown="1">
Takes one `H × W × 3` RGB image and returns an `H × W` depth map in metres.
</div>

<div class="rpx-task-shortcuts"><a href="#run-the-benchmark">Run it <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Use local data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="MODEL INPUT: RGB · frame 00113" width="640" height="480"><figcaption><span>MODEL INPUT</span>RGB · frame 00113</figcaption></figure>
<figure><img src="../../assets/scene012-depth.png" alt="GROUND TRUTH: Measured depth · display ramp" width="640" height="480"><figcaption><span>GROUND TRUTH</span>Measured depth · display ramp</figcaption></figure>
</div>
<p class="rpx-image-note">Scene 012, Interaction, frame 00113. The right image is the released ground-truth depth, colorized over 0.3–5.0 m; black pixels have no valid depth.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>One RGB image</strong></div>
<div><span>PREDICTION</span><strong>Depth map in metres</strong></div>
<div><span>EVALUATION</span><strong>Depth accuracy</strong></div>
</div>

## Connect your model

Put your prediction function in `my_model.py`. Load the model or API client once, at module level, rather than on every call.

```python title="my_model.py"
import numpy as np

def predict_depth(rgb):
    # model is yours, loaded once above; it must return metres
    depth_metres = model.predict(rgb)
    return np.asarray(depth_metres, dtype=np.float32)
```

<div class="rpx-task-note" markdown="1">
Return one float per pixel, in metres. If your model only predicts relative (up-to-scale) depth, say so with `--depth-output-kind relative`; don't rescale it yourself.
</div>

## Run the benchmark

Install the toolkit with `pip install 'rpx-benchmark[hub,schemas]'`. The smoke test checks the install; the second tab scores your model.

=== "Smoke test (offline)"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T1 --smoke --output results/t1-smoke
    ```

    This runs the real evaluator on a few synthetic inputs with a placeholder model. Nothing is downloaded and no weights are needed. The output directory must be new or empty.


=== "Your model on RPX data"

    Run this from the directory that contains `my_model.py`. Without `--manifest`, the command downloads the data it needs first.

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T1 \
      --model my_model:predict_depth \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T1 \
      --device cuda \
      --split easy \
      --max-samples 10
    ```


    Ten samples are enough to check the wiring; drop `--max-samples` to score the whole split. To use data you already have on disk, add `--manifest path/to/manifest.json` (see [using local data](../README.md#move-to-real-data)).


## Read the results

The output directory holds `result.json`, `summary.md`, per-sample tables and cell logs. `run_config.json` records the model revision and the manifest hash. From Python, the `run_*` functions return `(result, report, paths)`.

Check that the number of scored samples matches what you asked for. Keep the per-sample records: [Φ and 𝒥<sub>min</sub>](../../analysis/README.md) are computed from them. Latency and memory are measured separately, with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

`model` in the example is your own estimator; RPX doesn't provide one. Metric depth is the default (`--depth-output-kind metric`). With `--depth-output-kind relative`, predictions are fitted to ground truth with a least-squares scale and shift (`ls_affine`) before scoring.

The runner writes raw depth errors and per-cell records. Pixels without valid ground-truth depth are excluded, not treated as zero. The paper's F-score uses fixed intrinsics, so it needs the original image size and camera calibration; the small synthetic images in the smoke test can't exercise it.

??? note "Before a full run"

    Check units, shapes, input order and sample IDs on a small run first. Pin the model revision, and keep the split and preprocessing the same for every model you compare. The smoke test only exercises the evaluator; your model still needs its own environment and weights.

??? info "Workflow figure"

    [T1 workflow (SVG)](../../assets/benchmark-t1.svg): the inputs, adapter and scoring stages for this task.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [The paper's models](https://github.com/IRVLUTD/RPX/tree/main/docker)
</div>

</div>
