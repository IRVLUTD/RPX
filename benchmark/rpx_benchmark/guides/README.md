---
hide:
  - toc
---

<div class="rpx-home" markdown>

<div class="rpx-hero" markdown>

<div class="rpx-hero-copy" markdown>

<p class="rpx-eyebrow">RPX / ROBOT PERCEPTION TOOLKIT</p>

# Perception,<br>under interaction.

<p class="rpx-lead">The same objects. A changing scene.<br>Understand how your model holds up.</p>

Evaluate pretrained models on real RGB-D captures before, during, and after manipulation. Bring your own model, extend the metrics, and keep every comparison grounded in the same data.

<div class="rpx-actions"><a class="rpx-primary" href="getting-started/">Start benchmarking <span aria-hidden="true">↗</span></a><a class="rpx-secondary" href="api/">Explore the API <span aria-hidden="true">→</span></a></div>

</div>

<figure class="rpx-hero-figure"><img src="assets/hero-interaction.jpg" width="640" height="480" alt="Indoor RPX scene 012 during human interaction, with instance masks overlaid on everyday objects."><figcaption><span>SCENE 012 / INTERACTION</span><span>Released instance-mask overlay</span></figcaption></figure>

</div>

<div class="rpx-numbers" aria-label="Dataset at a glance"><div><strong>100</strong><span>multi-object scenes</span></div><div><strong>3</strong><span>capture phases</span></div><div><strong>6</strong><span>paper tasks</span></div><div><strong>70</strong><span>single-object captures</span></div></div>

<p class="rpx-eyebrow">01 / THE DATA</p>

## One scene. A changing world.

Clutter captures the scene before manipulation; Interaction introduces hands, occlusion, and motion; Clean records the scene afterwards. Egocentric Interaction provides another viewpoint on an independent clock. Object identities connect the views and phases.

<div class="rpx-filmstrip">
<figure><img src="assets/clutter.jpg" width="640" height="480" loading="lazy" alt="Scene 091 before manipulation with colored instance-mask overlays."><figcaption><span>01 / CLUTTER</span>Before manipulation</figcaption></figure>
<figure><img src="assets/interaction.jpg" width="640" height="480" loading="lazy" alt="Scene 091 during manipulation with a hand occluding objects and mask overlays."><figcaption><span>02 / INTERACTION</span>Hands, motion, occlusion</figcaption></figure>
<figure><img src="assets/clean.jpg" width="640" height="480" loading="lazy" alt="Scene 091 after manipulation with persistent object-mask colors."><figcaption><span>03 / CLEAN</span>After manipulation</figcaption></figure>
<figure><img src="assets/ego.jpg" width="960" height="720" loading="lazy" alt="Egocentric view of scene 091 during interaction with object-mask overlays."><figcaption><span>+ / EGOCENTRIC</span>A separate viewpoint</figcaption></figure>
</div>

<p class="rpx-image-note">Representative frames from outdoor scene 091, shown with released mask overlays. The four views are not frame-synchronized.</p>

<p class="rpx-eyebrow">02 / THE TOOLKIT</p>

## From predictions to evidence.

A versioned manifest selects the samples. The loader resolves inputs and ground truth, an adapter connects your model, and task-specific metrics produce per-sample records and aggregate reports. Model weights remain separate from the toolkit.

<div class="rpx-flow" aria-label="Benchmark workflow"><span><b>01</b> Manifest</span><i aria-hidden="true">→</i><span><b>02</b> Loader</span><i aria-hidden="true">→</i><span><b>03</b> Your model</span><i aria-hidden="true">→</i><span><b>04</b> Metrics</span><i aria-hidden="true">→</i><span><b>05</b> Reports</span></div>

<div class="rpx-routes">
<a href="getting-started/"><span class="rpx-route-number">01</span><span><strong>Run a benchmark</strong><small>Install, load your manifest, and inspect results.</small></span><span aria-hidden="true">↗</span></a>
<a href="models/"><span class="rpx-route-number">02</span><span><strong>Bring your own model</strong><small>Connect a local callable or an API-backed model.</small></span><span aria-hidden="true">↗</span></a>
<a href="metrics/"><span class="rpx-route-number">03</span><span><strong>Add a metric</strong><small>Register a calculator with a tested scoring contract.</small></span><span aria-hidden="true">↗</span></a>
<a href="tasks/"><span class="rpx-route-number">04</span><span><strong>Extend the tasks</strong><small>Define inputs, ground truth, adapters, and evaluation.</small></span><span aria-hidden="true">↗</span></a>
</div>

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
<figure><img src="assets/rgb.jpg" width="640" height="480" loading="lazy" alt="Single-object RGB capture of a dark handled object on the capture board."><figcaption><span>RGB</span>Appearance and model input</figcaption></figure>
<figure><img src="assets/depth.png" width="640" height="480" loading="lazy" alt="Colorized metric-depth visualization of the same single-object capture."><figcaption><span>METRIC DEPTH</span>Geometry and depth ground truth</figcaption></figure>
<figure><img src="assets/stereo-left.jpg" width="848" height="800" loading="lazy" alt="Left T265 fisheye view of the single-object capture board."><figcaption><span>FISHEYE / LEFT</span>Calibrated stereo input</figcaption></figure>
<figure><img src="assets/stereo-right.jpg" width="848" height="800" loading="lazy" alt="Right T265 fisheye view paired with the left view."><figcaption><span>FISHEYE / RIGHT</span>Paired viewpoint</figcaption></figure>
</div>

<p class="rpx-image-note">RGB, colorized depth, and stereo examples from the same SOS capture. The depth visualization is illustrative; scoring uses the released metric values, not image colors. See the <a href="data/">data guide</a> for annotation and modality details.</p>

## Concepts and acronyms

- **MOS / SOS:** multi-object scenes / single-object captures.
- **Phase:** Clutter, Interaction, or Clean. Egocentric Interaction is a separate view.
- **ESD:** empirical scene difficulty, used to group scenes by difficulty.
- **J / Jmin:** normalized desirability and worst-phase quality.
- **Φ:** phase robustness from the paper analysis tools; it is not the mean of raw metric values.
- **GT:** ground truth. Keep units, coordinates, and object identities consistent across predictions and GT.

<div class="rpx-closing" markdown>

## Build on the benchmark.

Start with a small integration run. Scale to the full protocol once shapes, units, IDs, and sample coverage agree.

[Get started →](getting-started/README.md) · [API reference](api/README.md) · [GitHub](https://github.com/IRVLUTD/RPX) · [PyPI](https://pypi.org/project/rpx-benchmark/)

</div>

The toolkit is released under the MIT license. Check dataset and pretrained-model licenses separately.

</div>
