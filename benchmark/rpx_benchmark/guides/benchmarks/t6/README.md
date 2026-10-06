<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/"><span>T5</span>Visual grounding</a>
<a href="../t6/" aria-current="page"><span>T6</span>In-context grounding</a>
</div>

<p class="rpx-task-kicker">BENCHMARK WALKTHROUGH / T6</p>

# In-context grounding

<div class="rpx-task-lead" markdown="1">
Verified SOS reference crop **first**, target RGB **second**, plus a canonical question → a box in the target image.
</div>

<div class="rpx-task-shortcuts"><a href="#02-run-the-task">Run the example <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Prepare data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene012-sos-reference.png" alt="IMAGE 1 / REFERENCE: Matched SOS teapot · global ID 65" width="108" height="139" class="rpx-reference-crop"><figcaption><span>IMAGE 1 / REFERENCE</span>Matched SOS teapot · global ID 65</figcaption></figure>
<figure><img src="../../assets/scene012-rgb.png" alt="IMAGE 2 / TARGET: Return coordinates in this image" width="640" height="480"><figcaption><span>IMAGE 2 / TARGET</span>Return coordinates in this image</figcaption></figure>
</div>
<p class="rpx-image-note">Same scene012 target, plus a matched SOS teapot reference. This crop illustrates the interface; actual requests use the manifest’s stored crop, pinned revision and hash.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>SOS reference + target + question</strong></div>
<div><span>PREDICTION</span><strong>Target box or binary answer</strong></div>
<div><span>EVALUATION</span><strong>Localization + response validity</strong></div>
</div>

## 01 / Connect your model

Save your callable in **`my_model.py`**. Initialize your own model or API client once in that module; the example below shows the adapter boundary.

```python title="my_model.py · adapter interface"
def predict_vqa(image_paths, prompt, max_new_tokens, output_kind):
    assert len(image_paths) == 2  # verified reference, then target
    return client.generate(images=image_paths, prompt=prompt,
                           max_new_tokens=max_new_tokens)
```

<div class="rpx-task-note" markdown="1">
**Check both images.** Send the verified reference crop first and target second. Boxes use target-image coordinates; a one-image endpoint cannot run T6.
</div>

## 02 / Run the task

Choose an offline integration check or run your callable on released RPX data. Install the toolkit first with `python -m pip install 'rpx-benchmark[hub,schemas]'`.

=== "Offline smoke test"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T6 --smoke --output results/t6-smoke
    ```

    This runs a tiny synthetic fixture with a toy callable. Use a fresh output directory; no model weights or dataset downloads are required.


=== "Your model + RPX data"

    Prepare a [task manifest](../README.md#move-to-real-data) and make `my_model.py` importable from your working directory.

    ```bash title="Terminal · real-data run"
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T6 \
      --manifest manifests/T6/hard.jsonl \
      --model my_model:predict_vqa \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T6 \
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

Use a canonical **in-context** JSONL manifest. The evaluator reconstructs the SOS crop using the stored inclusive crop box and verifies its SHA256. It preserves the manifest's immutable reference and target revisions. Do not substitute a full SOS frame, an unrelated reference object, or swap the two images.

The returned box uses XYXY normalized to 0–1000 in the **target** image. The same localization parser and metrics as T5 apply, with the in-context question types and two-image inputs. A one-image/text-only endpoint cannot satisfy this task. The SOS reference is the necessary exception to the common scene012 target walkthrough.

??? note "Before a full benchmark"

    Verify units and shapes, input ordering, frame/object/sample IDs, and that every requested prediction is accounted for. Load the model once per process, pin its revision, and keep the same data split and preprocessing across comparisons. Real pretrained inference requires that model's environment and weights; the packaged smoke fixture does not replace it.

??? info "Download the workflow figure"

    [Open the T6 vector workflow](../../assets/benchmark-t6.svg). It describes the interface and scoring stages, not a pretrained prediction.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [Reference models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/README.md#reference-models)
</div>

</div>
