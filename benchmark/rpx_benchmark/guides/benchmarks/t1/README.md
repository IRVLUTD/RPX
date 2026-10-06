<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/" aria-current="page"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>Visual grounding</a>
<a href="../t6/"><span>T6</span>In-context grounding</a>
</div>

<p class="rpx-task-kicker">BENCHMARK WALKTHROUGH / T1</p>

# Image depth

<div class="rpx-task-lead" markdown="1">
One `H × W × 3` RGB image → an `H × W` floating depth map in **metres**.
</div>

<div class="rpx-task-shortcuts"><a href="#02-run-the-task">Run the example <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Prepare data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="MODEL INPUT: RGB · frame 00113" width="640" height="480"><figcaption><span>MODEL INPUT</span>RGB · frame 00113</figcaption></figure>
<figure><img src="../../assets/scene012-depth.png" alt="GROUND TRUTH: Measured depth · display ramp" width="640" height="480"><figcaption><span>GROUND TRUTH</span>Measured depth · display ramp</figcaption></figure>
</div>
<p class="rpx-image-note">Scene012 · Interaction · frame 00113. Depth is released ground truth, colorized over 0.3–5.0 m; black pixels are invalid. No model predictions are shown.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>One RGB image</strong></div>
<div><span>PREDICTION</span><strong>Depth map in metres</strong></div>
<div><span>EVALUATION</span><strong>Depth accuracy</strong></div>
</div>

## 01 / Connect your model

Save your callable in **`my_model.py`**. Initialize your own model or API client once in that module; the example below shows the adapter boundary.

```python title="my_model.py · adapter interface"
import numpy as np

def predict_depth(rgb):
    # Replace with actual model inference; preserve metre units.
    depth_metres = model.predict(rgb)
    return np.asarray(depth_metres, dtype=np.float32)
```

<div class="rpx-task-note" markdown="1">
**Check resolution and units.** Return one floating depth value per pixel. Declare relative depth explicitly; retain the released validity mask and calibration.
</div>

## 02 / Run the task

Choose an offline integration check or run your callable on released RPX data. Install the toolkit first with `python -m pip install 'rpx-benchmark[hub,schemas]'`.

=== "Offline smoke test"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T1 --smoke --output results/t1-smoke
    ```

    This runs a tiny synthetic fixture with a toy callable. Use a fresh output directory; no model weights or dataset downloads are required.


=== "Your model + RPX data"

    Prepare a [task manifest](../README.md#move-to-real-data) and make `my_model.py` importable from your working directory.

    ```bash title="Terminal · real-data run"
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T1 \
      --manifest manifests/T1/hard.json \
      --model my_model:predict_depth \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T1 \
      --device cuda \
      --split hard \
      --max-samples 10
    ```


    For an installed Hub manifest, download a supported task split with `download_split`, then pass its returned local JSON path. `max_samples=10` is an integration subset; omit it for the full protocol.


## 03 / Inspect the results

Open the generated reports and cell logs under your output directory. `run_config.json` records the model revision and manifest hash; the frame/clip runner returns `(result, deployment_report, paths)` with the saved artifact paths.

Confirm that every requested sample is accounted for before aggregating. Keep raw scene/phase records for [Φ and JEDI](../../analysis/README.md), and measure inference separately with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

`model` is your initialized estimator, not an RPX-provided object. If the model emits relative depth, pass `--depth-output-kind relative` and preserve the task's pooled scene/phase alignment protocol. Metric outputs should use `metric` (the default).

The public runner emits raw depth errors and cell records. GT validity is task defined; do not count invalid depth pixels as zero-depth targets. The fixed-intrinsics paper F-score needs the canonical image size and camera calibration. Small synthetic images validate plumbing rather than every geometric diagnostic.

??? note "Before a full benchmark"

    Verify units and shapes, input ordering, frame/object/sample IDs, and that every requested prediction is accounted for. Load the model once per process, pin its revision, and keep the same data split and preprocessing across comparisons. Real pretrained inference requires that model's environment and weights; the packaged smoke fixture does not replace it.

??? info "Download the workflow figure"

    [Open the T1 vector workflow](../../assets/benchmark-t1.svg). It describes the interface and scoring stages, not a pretrained prediction.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [Reference models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/README.md#reference-models)
</div>

</div>
