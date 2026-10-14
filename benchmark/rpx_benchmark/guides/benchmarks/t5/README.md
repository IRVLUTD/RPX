<div class="rpx-benchmark-page" markdown="1">

<div class="rpx-task-switcher" role="navigation" aria-label="Benchmark tasks">
<a href="../t1/"><span>T1</span>Image depth</a>
<a href="../t2/"><span>T2</span>Video depth</a>
<a href="../t3/"><span>T3</span>Object tracking</a>
<a href="../t4/"><span>T4</span>Camera pose</a>
<a href="../t5/" aria-current="page"><span>T5</span>VQA</a>
<a href="../t6/"><span>T6</span>In-context VQA</a>
</div>

<p class="rpx-task-kicker">T5 WALKTHROUGH</p>

# VQA

<div class="rpx-task-lead" markdown="1">
Takes one image and a question and returns the model's raw text, which should contain a JSON object with a `label` and a `bbox`.
</div>

<div class="rpx-task-shortcuts"><a href="#run-the-benchmark">Run it <span aria-hidden="true">↓</span></a><a href="../#move-to-real-data">Use local data <span aria-hidden="true">↗</span></a></div>

<div class="rpx-task-visual">
<figure><img src="../../assets/scene049-rgb.png" alt="MODEL INPUT: Target RGB + canonical question" width="640" height="480"><figcaption><span>MODEL INPUT</span>Target RGB + canonical question</figcaption></figure>
<figure><img src="../../assets/hero-interaction.jpg" alt="ANNOTATION CONTEXT: Instance masks stay with the evaluator" width="640" height="480"><figcaption><span>ANNOTATION CONTEXT</span>Instance masks stay with the evaluator</figcaption></figure>
</div>
<p class="rpx-image-note">Scene 049, Interaction, frame 00185. The masks on the right are what the evaluator scores against; the model only gets the RGB image and the question.</p>

<div class="rpx-task-contract">
<div><span>INPUT</span><strong>Target image + question</strong></div>
<div><span>PREDICTION</span><strong>Box or yes/no answer</strong></div>
<div><span>EVALUATION</span><strong>Localization + response validity</strong></div>
</div>

## Connect your model

Put your prediction function in `my_model.py`. Load the model or API client once, at module level, rather than on every call.

```python title="my_model.py"
def predict_vqa(image_paths, prompt, max_new_tokens, output_kind):
    # image_paths holds the target image; client is your local model or API client
    return client.generate(images=image_paths, prompt=prompt,
                           max_new_tokens=max_new_tokens)
```

<div class="rpx-task-note" markdown="1">
Return the model's text as is; RPX parses it. Responses that fail to parse, and calls that raise, are scored as failures rather than skipped.
</div>

## Run the benchmark

Install the toolkit with `pip install 'rpx-benchmark[hub,schemas]'`. The smoke test checks the install; the second tab scores your model.

=== "Smoke test (offline)"

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T5 --smoke --output results/t5-smoke
    ```

    This runs the real evaluator on a few synthetic inputs with a placeholder model. Nothing is downloaded and no weights are needed. The output directory must be new or empty.


=== "Your model on RPX data"

    Run this from the directory that contains `my_model.py`. Without `--manifest`, it scores the 58 T5 questions packaged with the toolkit, pinned to the dataset release, and downloads only their images.

    ```bash
    python -m rpx_benchmark.examples.benchmark_tasks \
      --task T5 \
      --model my_model:predict_vqa \
      --model-name my-model \
      --model-revision exact-checkpoint-or-api-version \
      --output results/my-model/T5
    ```


    The packaged questions are a quick-start set, not the paper's full split. To score more, build a canonical manifest ([how](../README.md#prepare-canonical-vqa-samples)) and pass it with `--manifest` and `--image-cache`. Every row of a manifest you pass is scored, and `--max-samples` is refused, so cut the manifest itself for a short run. Keep each row's question, box, image locator and reference metadata intact.


## Read the results

<div class="rpx-result-files">
<div><code>run_config.json</code><span>Model and evaluation settings</span></div>
<div><code>predictions.jsonl</code><span>Raw responses for each question</span></div>
<div><code>result.json</code><span>Scored metrics and failure counts</span></div>
</div>

Check that the number of scored samples matches what you asked for. Keep the per-sample records: [Φ and 𝒥<sub>min</sub>](../../analysis/README.md) are computed from them. Latency and memory are measured separately, with the [hardware profiler](../../profiling/README.md).

### Scoring and protocol

The model sees only the image and the prompt; boxes and answers stay with the evaluator. Boxes are `[x1, y1, x2, y2]` normalized to 0–1000, for example `{"label":"cup","bbox":[100,200,600,800]}` (an illustration, not scene049's ground truth). Yes/no questions expect `yes` or `no`.

Scores include box IoU, accuracy at IoU thresholds, parse validity and a breakdown by question type. Malformed responses and inference errors count as failures; an error fetching the dataset stops the run. Label accuracy is reported separately from localization. For API models, the recorded wall time includes network transport.

??? note "Before a full run"

    Check units, shapes, input order and sample IDs on a small run first. Pin the model revision, and keep the split and preprocessing the same for every model you compare. The smoke test only exercises the evaluator; your model still needs its own environment and weights.

??? info "Workflow figure"

    [T5 workflow (SVG)](../../assets/benchmark-t5.svg): the inputs, adapter and scoring stages for this task.

<div class="rpx-task-next" markdown="1">
[All six tasks](../README.md) · [Bring your own model](../../models/README.md) · [The paper's models](https://github.com/IRVLUTD/RPX/tree/main/docker)
</div>

</div>
