<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/" aria-current="page"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>Visual grounding</a>
<a href="../t6/"><span>T6</span>In-context grounding</a>
</div>

<p class="rpx-task-kicker">BENCHMARK WALKTHROUGH / T3</p>

# Object tracking

<div class="rpx-task-lead" markdown="1">
RGB frames, sequence boundaries and protocol-specific initialization → boxes with persistent track IDs.
</div>

<div class="rpx-task-shortcuts"><a href="#02-run-the-task">Run the example <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Prepare data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="MODEL INPUT: RGB · frame 00113" width="640" height="480"><figcaption><span>MODEL INPUT</span>RGB · frame 00113</figcaption></figure>
<figure><img src="../../assets/hero-interaction.jpg" alt="GROUND TRUTH: Released instance-mask overlay" width="640" height="480"><figcaption><span>GROUND TRUTH</span>Released instance-mask overlay</figcaption></figure>
</div>
<p class="rpx-image-note">Scene012 · Interaction · frame 00113. The overlay shows released ground-truth masks, not tracker predictions. Persistent identities must survive the sequence.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Frames + initialization</strong></div>
<div><span>PREDICTION</span><strong>Boxes + persistent IDs</strong></div>
<div><span>EVALUATION</span><strong>Tracking consistency</strong></div>
</div>

## 01 / Connect your model

Save your callable in **`my_model.py`**. Initialize your own model or API client once in that module; the example below shows the adapter boundary.

```python title="my_model.py · adapter interface"
# Stateful object: initialize once per sequence, then update per frame.
def predict_tracks(rgb):
    return tracker.update(rgb)  # [{"track_id": "1", "boxes": boxes_xyxy}, ...]
```

<div class="rpx-task-note" markdown="1">
**Check state and identities.** Initialize once per sequence, reset at sequence boundaries, and preserve track IDs. Use the sequence evaluator for paper HOTA/IDF1.
</div>

## 02 / Run the task

Choose an offline integration check or run your callable on released RPX data. Install the toolkit first with `python -m pip install 'rpx-benchmark[hub,schemas]'`.

=== "Offline smoke test"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T3 --smoke --output results/t3-smoke
    ```

    This runs a tiny synthetic fixture with a toy callable. Use a fresh output directory; no model weights or dataset downloads are required.


=== "Your model + RPX data"

    Prepare a [task manifest](../README.md#move-to-real-data) and make `my_model.py` importable from your working directory.

    ```bash title="Terminal · real-data run"
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T3 \
      --manifest manifests/T3/hard.json \
      --model my_model:predict_tracks \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T3 \
      --device cuda \
      --split hard \
      --max-samples 10
    ```


    For an installed Hub manifest, download a supported task split with `download_split`, then pass its returned local JSON path. `max_samples=10` is an integration subset; omit it for the full protocol.


## 03 / Inspect the results

Open the generated reports and cell logs under your output directory. `run_config.json` records the model revision and manifest hash; the frame/clip runner returns `(result, deployment_report, paths)` with the saved artifact paths.

Confirm that every requested sample is accounted for before aggregating. Keep raw scene/phase records for [Φ and JEDI](../../analysis/README.md), and measure inference separately with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

The simple callable path invokes the per-frame tracking adapter and registered tracking metrics. It expects instance identities consistent with the local GT contract; it is an integration path, **not the full sequence-level paper evaluator**. Provide state/initialization in your module and reset it at sequence boundaries. Do not treat each frame as an independent detector when evaluating a tracker.

For paper-model tracking runs and sequence-level metrics, use [`run_tracking_paper.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/scripts/run_tracking_paper.py), [`run_tracking_gate.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/scripts/run_tracking_gate.py), and the pinned evaluator protocol. Their `--help` defines the model-specific initialization, mask outputs and sequence settings. HOTA/IDF1 values from a sequence evaluator should not be substituted with framewise smoke scores.

??? note "Before a full benchmark"

    Verify units and shapes, input ordering, frame/object/sample IDs, and that every requested prediction is accounted for. Load the model once per process, pin its revision, and keep the same data split and preprocessing across comparisons. Real pretrained inference requires that model's environment and weights; the packaged smoke fixture does not replace it.

??? info "Download the workflow figure"

    [Open the T3 vector workflow](../../assets/benchmark-t3.svg). It describes the interface and scoring stages, not a pretrained prediction.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [Reference models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/README.md#reference-models)
</div>

</div>
