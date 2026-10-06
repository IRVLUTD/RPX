<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/" aria-current="page"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>Visual grounding</a>
<a href="../t6/"><span>T6</span>In-context grounding</a>
</div>

<p class="rpx-task-kicker">BENCHMARK WALKTHROUGH / T2</p>

# Video depth

<div class="rpx-task-lead" markdown="1">
One `T × H × W × 3` phase RGB clip → a `T × H × W` depth sequence.
</div>

<div class="rpx-task-shortcuts"><a href="#02-run-the-task">Run the example <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Prepare data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="FRAME 00113: RGB clip · preserve frame order" width="640" height="480"><figcaption><span>FRAME 00113</span>RGB clip · preserve frame order</figcaption></figure>
<figure><img src="../../assets/scene012-next.png" alt="FRAME 00114: Adjacent frame · same capture" width="640" height="480"><figcaption><span>FRAME 00114</span>Adjacent frame · same capture</figcaption></figure>
</div>
<p class="rpx-image-note">Scene012 · Interaction · adjacent frames 00113–00114 illustrate the sequence input. The actual clip and frame budget come from your manifest.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Ordered RGB clip</strong></div>
<div><span>PREDICTION</span><strong>Depth sequence</strong></div>
<div><span>EVALUATION</span><strong>Depth + temporal accuracy</strong></div>
</div>

## 01 / Connect your model

Save your callable in **`my_model.py`**. Initialize your own model or API client once in that module; the example below shows the adapter boundary.

```python title="my_model.py · adapter interface"
import numpy as np

def predict_video_depth(rgb):
    return np.asarray(model.predict_clip(rgb), dtype=np.float32)
```

<div class="rpx-task-note" markdown="1">
**Check order and clip length.** Return one depth map per input frame. Keep sampling, clip length and alignment identical across comparisons.
</div>

## 02 / Run the task

Choose an offline integration check or run your callable on released RPX data. Install the toolkit first with `python -m pip install 'rpx-benchmark[hub,schemas]'`.

=== "Offline smoke test"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T2 --smoke --output results/t2-smoke
    ```

    This runs a tiny synthetic fixture with a toy callable. Use a fresh output directory; no model weights or dataset downloads are required.


=== "Your model + RPX data"

    Prepare a [task manifest](../README.md#move-to-real-data) and make `my_model.py` importable from your working directory.

    ```bash title="Terminal · real-data run"
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T2 \
      --manifest manifests/T2/hard.json \
      --model my_model:predict_video_depth \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T2 \
      --device cuda \
      --split hard \
      --max-samples 10
    ```


    For an installed Hub manifest, download a supported task split with `download_split`, then pass its returned local JSON path. `max_samples=10` is an integration subset; omit it for the full protocol.


## 03 / Inspect the results

Open the generated reports and cell logs under your output directory. `run_config.json` records the model revision and manifest hash; the frame/clip runner returns `(result, deployment_report, paths)` with the saved artifact paths.

Confirm that every requested sample is accounted for before aggregating. Keep raw scene/phase records for [Φ and JEDI](../../analysis/README.md), and measure inference separately with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

The callable must preserve input frame order and return the same `T`. Use `--depth-output-kind relative` only for relative predictions; those receive the documented per-clip scale/shift fit. Compare models with the same clip length, frame sampling and alignment protocol.

Video scoring combines framewise depth accuracy and RGB-D temporal metrics (`tgm`, `tgse`). The paper headline vector excludes pose- and flow-dependent diagnostics. In this runner, `--max-samples` limits **clips**, not individual frames. The `VideoDepthRunConfig` API exposes frame-budget and sampling controls for ablations.

??? note "Before a full benchmark"

    Verify units and shapes, input ordering, frame/object/sample IDs, and that every requested prediction is accounted for. Load the model once per process, pin its revision, and keep the same data split and preprocessing across comparisons. Real pretrained inference requires that model's environment and weights; the packaged smoke fixture does not replace it.

??? info "Download the workflow figure"

    [Open the T2 vector workflow](../../assets/benchmark-t2.svg). It describes the interface and scoring stages, not a pretrained prediction.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [Reference models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/README.md#reference-models)
</div>

</div>
