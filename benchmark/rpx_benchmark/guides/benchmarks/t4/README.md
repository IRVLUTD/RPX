<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/" aria-current="page"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>VQA</a>
<a href="../t6/"><span>T6</span>In-context VQA</a>
</div>

<p class="rpx-task-kicker">T4 WALKTHROUGH</p>

# Relative camera pose

<div class="rpx-task-lead" markdown="1">
Takes two RGB frames and returns the relative pose: a `3 × 3` rotation and a translation 3-vector, in the manifest's coordinate convention.
</div>

<div class="rpx-task-shortcuts"><a href="#run-the-benchmark">Run it <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Use local data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="IMAGE A: RGB · frame 00113" width="640" height="480"><figcaption><span>IMAGE A</span>RGB · frame 00113</figcaption></figure>
<figure><img src="../../assets/scene012-next.png" alt="IMAGE B: RGB · frame 00114" width="640" height="480"><figcaption><span>IMAGE B</span>RGB · frame 00114</figcaption></figure>
</div>
<p class="rpx-image-note">Scene 012, Interaction, two adjacent frames. The benchmark's actual pairs are chosen by the pose protocol.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Two RGB frames</strong></div>
<div><span>PREDICTION</span><strong>Relative rotation + translation</strong></div>
<div><span>EVALUATION</span><strong>Rotation + translation error</strong></div>
</div>

## Connect your model

Put your prediction function in `my_model.py`. Load the model or API client once, at module level, rather than on every call.

```python title="my_model.py"
import numpy as np

def predict_pose(rgb_a, rgb_b):
    rotation, translation = model.predict_pair(rgb_a, rgb_b)
    return {'rotation': np.asarray(rotation), 'translation': np.asarray(translation)}
```

<div class="rpx-task-note" markdown="1">
Return the pose of camera B relative to camera A, in metres. Keep the A/B order; a zero translation has no direction and can't be scored for it.
</div>

## Run the benchmark

Install the toolkit with `pip install 'rpx-benchmark[hub,schemas]'`. The smoke test checks the install; the second tab scores your model.

=== "Smoke test (offline)"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T4 --smoke --output results/t4-smoke
    ```

    This runs the real evaluator on a few synthetic inputs with a placeholder model. Nothing is downloaded and no weights are needed. The output directory must be new or empty.


=== "Your model on RPX data"

    Run this from the directory that contains `my_model.py`. Without `--manifest`, the command downloads the data it needs first.

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T4 \
      --model my_model:predict_pose \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T4 \
      --device cuda \
      --split easy \
      --max-samples 10
    ```


    Ten samples are enough to check the wiring; drop `--max-samples` to score the whole split. To use data you already have on disk, add `--manifest path/to/manifest.json` (see [using local data](../README.md#move-to-real-data)).


## Read the results

The output directory holds `result.json`, `summary.md`, per-sample tables and cell logs. `run_config.json` records the model revision and the manifest hash. From Python, the `run_*` functions return `(result, report, paths)`.

Check that the number of scored samples matches what you asked for. Keep the per-sample records: [Φ and 𝒥<sub>min</sub>](../../analysis/README.md) are computed from them. Latency and memory are measured separately, with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

The manifest names both images and their ground-truth poses. Frame conventions, translation units, timestamps and which phase pairs are eligible are fixed by the protocol; nothing is inferred from your model's output.

Pair selection and the threshold AUC are defined in [`pose_pairs`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_pairs.html) and [`pose_metrics`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_metrics.html). The smoke test only checks the adapter and reports; reproducing the paper's numbers needs its pair protocol.

??? note "Before a full run"

    Check units, shapes, input order and sample IDs on a small run first. Pin the model revision, and keep the split and preprocessing the same for every model you compare. The smoke test only exercises the evaluator; your model still needs its own environment and weights.

??? info "Workflow figure"

    [T4 workflow (SVG)](../../assets/benchmark-t4.svg): the inputs, adapter and scoring stages for this task.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [The paper's models](https://github.com/IRVLUTD/RPX/tree/main/docker)
</div>

</div>
