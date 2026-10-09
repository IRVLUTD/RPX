---
hide:
  - toc
---

<div class="rpx-home" markdown>

<div class="rpx-hero" markdown>

<div class="rpx-hero-copy" markdown>

<p class="rpx-eyebrow">RPX / ROBOT PERCEPTION TOOLKIT</p>

# Same scene.<br>Different story.

<p class="rpx-lead">Score any perception model on real scenes before, during and after manipulation, and see where it breaks.</p>

Six tasks, one evaluator, and the paper's robustness diagnosis. Wrap your model or API in a few lines; RPX handles the data, splits and metrics.

<div class="rpx-actions"><a class="rpx-primary" href="getting-started/">Score your model <span aria-hidden="true">↗</span></a><a class="rpx-secondary" href="benchmarks/">See the six tasks <span aria-hidden="true">→</span></a></div>

</div>

<figure class="rpx-hero-figure"><img src="assets/hero-interaction.jpg" width="640" height="480" alt="Indoor RPX scene 012 during human interaction, with instance masks overlaid on everyday objects."><figcaption><span>SCENE 012 / INTERACTION</span><span>Released instance-mask overlay</span></figcaption></figure>

</div>

<div class="rpx-numbers" aria-label="Dataset at a glance"><div><strong>100</strong><span>multi-object scenes</span></div><div><strong>3</strong><span>capture phases</span></div><div><strong>6</strong><span>paper tasks</span></div><div><strong>70</strong><span>single-object captures</span></div></div>

## Try it now

```bash
pip install 'rpx-benchmark[hub,schemas]'
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke --output rpx-check
```

All six evaluators run on synthetic inputs in about a second, on CPU. [Score your own model on real scenes →](getting-started/README.md)

<p class="rpx-eyebrow">01 / THE FINDING</p>

## One scene. A changing world.

Each scene is recorded before manipulation (Clutter), while hands move objects (Interaction) and afterwards (Clean), plus an egocentric view. Here is what the paper found:

<figure class="rpx-workflow-figure rpx-finding"><img src="assets/rpx-finding-light.webp#only-light" width="2400" height="1482" alt="Scene 012 in Clutter, Interaction, Clean and egocentric views with mean quality per view for video depth, tracking and VQA; Interaction is lowest. A map of all 65 evaluations by worst-phase quality and phase robustness shows none in the golden zone."><img src="assets/rpx-finding-dark.webp#only-dark" width="2400" height="1482" alt="Scene 012 in Clutter, Interaction, Clean and egocentric views with mean quality per view for video depth, tracking and VQA; Interaction is lowest. A map of all 65 evaluations by worst-phase quality and phase robustness shows none in the golden zone."><figcaption>Bars: mean quality 𝒥 over all evaluated models, from the paper's Tables III–VI. Map: all 65 evaluations from the paper's Fig. 4. The four views of indoor scene 012 are not frame-synchronized.</figcaption></figure>





<p class="rpx-eyebrow">02 / THE TOOLKIT</p>

## From predictions to evidence.

<figure class="rpx-workflow-figure"><a href="assets/toolkit-overview.svg"><img src="assets/toolkit-overview.svg" alt="RPX data with scene012 feeds six task model interfaces, evaluation, Phi/JEDI calculation and hardware evidence."></a><figcaption>Bring your model to RPX task data. Adapters connect inference to shared scoring; annotation, profiling and extension tools complete the workflow. Open the vector figure to zoom.</figcaption></figure>

A versioned manifest selects the samples. The loader resolves inputs and ground truth, an adapter connects your model, and task-specific metrics produce per-sample records and aggregate reports. Model weights remain separate from the toolkit.

<div class="rpx-flow" aria-label="Benchmark workflow"><span><b>01</b> Manifest</span><i aria-hidden="true">→</i><span><b>02</b> Loader</span><i aria-hidden="true">→</i><span><b>03</b> Your model</span><i aria-hidden="true">→</i><span><b>04</b> Metrics</span><i aria-hidden="true">→</i><span><b>05</b> Reports</span></div>

<div class="rpx-routes">
<a href="getting-started/"><span class="rpx-route-number">01</span><span><strong>Run a benchmark</strong><small>Install, load your manifest, and inspect results.</small></span><span aria-hidden="true">↗</span></a>
<a href="models/"><span class="rpx-route-number">02</span><span><strong>Bring your own model</strong><small>Connect a local callable or an API-backed model.</small></span><span aria-hidden="true">↗</span></a>
<a href="metrics/"><span class="rpx-route-number">03</span><span><strong>Add a metric</strong><small>Register a calculator with a tested scoring contract.</small></span><span aria-hidden="true">↗</span></a>
<a href="tasks/"><span class="rpx-route-number">04</span><span><strong>Extend the tasks</strong><small>Define inputs, ground truth, adapters, and evaluation.</small></span><span aria-hidden="true">↗</span></a>
</div>

## Tools you can use today.

| Workflow | What it gives you |
| --- | --- |
| [Mask annotation pipeline](annotation/README.md) | Box curation, SAM2 propagation, review, refinement and identity mapping |
| [Hardware profiler](profiling/README.md) | Warmup-excluded synchronized latency, memory and system cards |
| [Φ/JEDI calculator](analysis/README.md) | Robustness, joint desirability and worst-phase quality from raw metrics |
| [Six benchmark samples](benchmarks/README.md) | Installed smoke commands and real manifest/callable examples for every paper task |
| [Bring your own model](models/README.md) | Local models and API clients through task-specific contracts |
| [All capabilities](capabilities/README.md) | Data tools, ESD, batch adapters, visualization, additional tasks and extensions |

The figures and task walkthroughs use **scene012** as a common target. The matched SOS reference required by T6 is the exception to the common target scene.

## Six tasks. Shared scenes.

| Task | Input | Evaluation |
| --- | --- | --- |
| T1 · Image depth | RGB image | Metric-depth accuracy |
| T2 · Video depth | RGB sequence | Depth accuracy and temporal consistency |
| T3 · Object tracking | Video and initialization | Multi-object tracking and identity consistency |
| T4 · Relative camera pose | Image pairs | Relative-pose errors and threshold AUC |
| T5 · Visual grounding | Image and text question | Localization accuracy |
| T6 · In-context grounding | Image, question, and reference image | Reference-conditioned localization |

T5 and T6 use visual-grounding tooling with distinct protocols. The toolkit also exposes detection, segmentation, sparse depth, and keypoint matching; `available_tasks()` lists the registered runners.

<p class="rpx-eyebrow">03 / THE MODALITIES</p>

## More than an RGB frame.

Exocentric captures combine RGB, metric depth, instance masks, calibrated fisheye stereo, and camera pose. Each modality plays a different role in loading, evaluation, and checking predictions.

<div class="rpx-modalities">
<figure><img src="assets/rgb.jpg" width="640" height="480" loading="lazy" alt="Scene012 Interaction RGB frame00113."><figcaption><span>RGB</span>Appearance and model input</figcaption></figure>
<figure><img src="assets/depth.png" width="640" height="480" loading="lazy" alt="Colorized metric-depth visualization of the same scene012 Interaction capture."><figcaption><span>METRIC DEPTH</span>Geometry and depth ground truth</figcaption></figure>
<figure><img src="assets/stereo-left.jpg" width="848" height="800" loading="lazy" alt="Left T265 fisheye view of the scene012 Interaction capture."><figcaption><span>FISHEYE / LEFT</span>Calibrated stereo input</figcaption></figure>
<figure><img src="assets/stereo-right.jpg" width="848" height="800" loading="lazy" alt="Right T265 fisheye view paired with the left view."><figcaption><span>FISHEYE / RIGHT</span>Paired viewpoint</figcaption></figure>
</div>

<p class="rpx-image-note">RGB, measured depth, and fisheye stereo from scene012 Interaction frame00113. The depth visualization is illustrative; scoring uses the released metric values, not image colors. See the <a href="data/">data guide</a> for annotation and modality details.</p>

## Concepts and acronyms

- **MOS / SOS:** multi-object scenes / single-object captures.
- **Phase:** Clutter, Interaction, or Clean. Egocentric Interaction is a separate view.
- **Easy / Medium / Hard:** difficulty tiers of 33, 33 and 34 scenes, stratified by how much correction each scene's masks needed.
- **JEDI / J / Jmin:** Joint Empirical Desirability Index; overall mean cell desirability and the lowest phase-mean quality.
- **Φ:** phase robustness from repeated-measures analysis of paired raw metrics. High robustness must be considered alongside prediction quality.
- **GT:** ground truth. Keep units, coordinates, and object identities consistent across predictions and GT.

<div class="rpx-closing" markdown>

## Build on the benchmark.

Start with a small integration run. Scale to the full protocol once shapes, units, IDs, and sample coverage agree.

[Get started →](getting-started/README.md) · [API reference](api/README.md) · [GitHub](https://github.com/IRVLUTD/RPX) · [PyPI](https://pypi.org/project/rpx-benchmark/)

</div>

## Cite RPX

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and Kamath, Srinanditha and
            Allu, Sai Haneesh and Rangappa, Govind Tyagi and Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

[Licenses and citation](citation/README.md). Toolkit code is MIT; dataset and pretrained-model licenses are separate.

</div>
