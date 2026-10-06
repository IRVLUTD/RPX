<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/" aria-current="page"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>Visual grounding</a>
<a href="../t6/"><span>T6</span>In-context grounding</a>
</div>

<p class="rpx-task-kicker">BENCHMARK WALKTHROUGH / T4</p>

# Relative camera pose

<div class="rpx-task-lead" markdown="1">
Two RGB frames → relative `3 × 3` rotation and `3`-vector translation in the declared coordinate convention.
</div>

<div class="rpx-task-shortcuts"><a href="#02-run-the-task">Run the example <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Prepare data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="IMAGE A: RGB · frame 00113" width="640" height="480"><figcaption><span>IMAGE A</span>RGB · frame 00113</figcaption></figure>
<figure><img src="../../assets/scene012-next.png" alt="IMAGE B: RGB · frame 00114" width="640" height="480"><figcaption><span>IMAGE B</span>RGB · frame 00114</figcaption></figure>
</div>
<p class="rpx-image-note">Scene012 · Interaction · adjacent frames illustrate the two-image interface. Benchmark pairs are selected by the pose protocol, not by this illustration.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Two RGB frames</strong></div>
<div><span>PREDICTION</span><strong>Relative rotation + translation</strong></div>
<div><span>EVALUATION</span><strong>Rotation + translation error</strong></div>
</div>

## 01 / Connect your model

Save your callable in **`my_model.py`**. Initialize your own model or API client once in that module; the example below shows the adapter boundary.

```python title="my_model.py · adapter interface"
import numpy as np

def predict_pose(rgb_a, rgb_b):
    rotation, translation = model.predict_pair(rgb_a, rgb_b)
    return {'rotation': np.asarray(rotation), 'translation': np.asarray(translation)}
```

<div class="rpx-task-note" markdown="1">
**Check the pair convention.** Preserve image A/B order, coordinate frames, translation units and pair eligibility. Zero translation has no valid direction.
</div>

## 02 / Run the task

Choose an offline integration check or run your callable on released RPX data. Install the toolkit first with `python -m pip install 'rpx-benchmark[hub,schemas]'`.

=== "Offline smoke test"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T4 --smoke --output results/t4-smoke
    ```

    This runs a tiny synthetic fixture with a toy callable. Use a fresh output directory; no model weights or dataset downloads are required.


=== "Your model + RPX data"

    Prepare a [task manifest](../README.md#move-to-real-data) and make `my_model.py` importable from your working directory.

    ```bash title="Terminal · real-data run"
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T4 \
      --manifest manifests/T4/hard.json \
      --model my_model:predict_pose \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T4 \
      --device cuda \
      --split hard \
      --max-samples 10
    ```


    For an installed Hub manifest, download a supported task split with `download_split`, then pass its returned local JSON path. `max_samples=10` is an integration subset; omit it for the full protocol.


## 03 / Inspect the results

Open the generated reports and cell logs under your output directory. `run_config.json` records the model revision and manifest hash; the frame/clip runner returns `(result, deployment_report, paths)` with the saved artifact paths.

Confirm that every requested sample is accounted for before aggregating. Keep raw scene/phase records for [Φ and JEDI](../../analysis/README.md), and measure inference separately with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

The manifest must identify both images and their corresponding pose GT. Camera frame conventions, translation units/direction, timestamps, and eligible phase pairs are protocol choices, not inferred from a model's output. A zero translation can have undefined direction and must not be interpreted as a valid directional estimate.

Use [`pose_pairs`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_pairs.html) and [`pose_metrics`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_metrics.html) for the full pair selection and threshold AUC definitions. The generic smoke runner checks the adapter/reporting boundary; reproducing paper results requires the corresponding pair protocol.

??? note "Before a full benchmark"

    Verify units and shapes, input ordering, frame/object/sample IDs, and that every requested prediction is accounted for. Load the model once per process, pin its revision, and keep the same data split and preprocessing across comparisons. Real pretrained inference requires that model's environment and weights; the packaged smoke fixture does not replace it.

??? info "Download the workflow figure"

    [Open the T4 vector workflow](../../assets/benchmark-t4.svg). It describes the interface and scoring stages, not a pretrained prediction.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [Reference models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/README.md#reference-models)
</div>

</div>
