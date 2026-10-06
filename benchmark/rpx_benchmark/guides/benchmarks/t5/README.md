<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/" aria-current="page"><span>T5</span>Visual grounding</a>
<a href="../t6/"><span>T6</span>In-context grounding</a>
</div>

<p class="rpx-task-kicker">BENCHMARK WALKTHROUGH / T5</p>

# Visual grounding

<div class="rpx-task-lead" markdown="1">
One target image + canonical question → raw text containing a JSON `label` and `bbox`.
</div>

<div class="rpx-task-shortcuts"><a href="#02-run-the-task">Run the example <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Prepare data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-rgb.png" alt="MODEL INPUT: Target RGB + canonical question" width="640" height="480"><figcaption><span>MODEL INPUT</span>Target RGB + canonical question</figcaption></figure>
<figure><img src="../../assets/hero-interaction.jpg" alt="ANNOTATION CONTEXT: Instance masks stay with the evaluator" width="640" height="480"><figcaption><span>ANNOTATION CONTEXT</span>Instance masks stay with the evaluator</figcaption></figure>
</div>
<p class="rpx-image-note">Scene012 · Interaction · frame 00113. The annotation overlay explains the scoring context; the model receives the RGB image and public question only.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Target image + question</strong></div>
<div><span>PREDICTION</span><strong>Target box or binary answer</strong></div>
<div><span>EVALUATION</span><strong>Localization + response validity</strong></div>
</div>

## 01 / Connect your model

Save your callable in **`my_model.py`**. Initialize your own model or API client once in that module; the example below shows the adapter boundary.

```python title="my_model.py · adapter interface"
def predict_vqa(image_paths, prompt, max_new_tokens, output_kind):
    # One target image; use your own local or API client and image template.
    return client.generate(images=image_paths, prompt=prompt,
                           max_new_tokens=max_new_tokens)
```

<div class="rpx-task-note" markdown="1">
**Check every response.** Use the canonical question type and response schema. Parsing or inference failures remain in the scored denominator.
</div>

## 02 / Run the task

Choose an offline integration check or run your callable on released RPX data. Install the toolkit first with `python -m pip install 'rpx-benchmark[hub,schemas]'`.

=== "Offline smoke test"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T5 --smoke --output results/t5-smoke
    ```

    This runs a tiny synthetic fixture with a toy callable. Use a fresh output directory; no model weights or dataset downloads are required.


=== "Your model + RPX data"

    Prepare a [task manifest](../README.md#move-to-real-data) and make `my_model.py` importable from your working directory.

    ```bash title="Terminal · real-data run"
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T5 \
      --manifest manifests/T5/hard.jsonl \
      --model my_model:predict_vqa \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T5 \
      --image-cache .cache/rpx-vqa
    ```


    Every row in the VQA manifest is scored. Select a smaller canonical manifest beforehand for a smoke run. Keep its question, bbox, image locator and reference metadata intact.


## 03 / Inspect the results

<div class="rpx-result-files">
<div><code>run_config.json</code><span>Model and evaluation settings</span></div>
<div><code>predictions.jsonl</code><span>Raw responses for each question</span></div>
<div><code>result.json</code><span>Scored metrics and failure counts</span></div>
</div>

Confirm that every requested sample is accounted for before aggregating. Keep raw scene/phase records for [Φ and JEDI](../../analysis/README.md), and measure inference separately with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

The model sees images and the public prompt. Target boxes and GT answers stay inside the evaluator. Box output uses **XYXY normalized to 0–1000**: `{"label":"cup","bbox":[100,200,600,800]}`. Binary questions instead require `yes` or `no`. These numbers illustrate the response schema; they are not scene012 GT.

Scoring includes bbox IoU, localization at IoU thresholds, parser validity and per-question-type diagnostics. Malformed responses and inference exceptions count as failures; dataset-fetch errors abort. Label diagnostics are reported separately from the primary localization score. Request wall time includes API transport when the callable uses an API.

??? note "Before a full benchmark"

    Verify units and shapes, input ordering, frame/object/sample IDs, and that every requested prediction is accounted for. Load the model once per process, pin its revision, and keep the same data split and preprocessing across comparisons. Real pretrained inference requires that model's environment and weights; the packaged smoke fixture does not replace it.

??? info "Download the workflow figure"

    [Open the T5 vector workflow](../../assets/benchmark-t5.svg). It describes the interface and scoring stages, not a pretrained prediction.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [Reference models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/README.md#reference-models)
</div>

</div>
