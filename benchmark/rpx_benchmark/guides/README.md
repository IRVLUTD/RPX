---
hide:
  - toc
---

<div class="rpx-home" markdown>

<div class="rpx-hero rpx-hero-single" markdown>

<div class="rpx-hero-copy" markdown>

<p class="rpx-eyebrow">RPX / ROBOT PERCEPTION TOOLKIT</p>

# Same scene.<br>Different story.

<p class="rpx-lead">Score any perception model on real scenes before, during and after manipulation, and see where it breaks.</p>

Six tasks, one evaluator, and the paper's robustness diagnosis. Wrap your model or API in a few lines; RPX handles the data, splits and metrics.

<div class="rpx-actions"><a class="rpx-primary" href="getting-started/">Score your model <span aria-hidden="true">↗</span></a><a class="rpx-secondary" href="benchmarks/">See the six tasks <span aria-hidden="true">→</span></a></div>

</div>

</div>

<figure class="rpx-overview-figure"><img src="assets/rpx-overview.webp" width="2400" height="1393" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires."><figcaption>RPX at a glance: today each task is benchmarked on its own data; RPX scores six tasks on the same real scenes, before, during and after manipulation.</figcaption></figure>

## Try it now

```bash
pip install 'rpx-benchmark[hub,schemas]'
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke --output rpx-check
```

All six evaluators run on synthetic inputs in a few seconds, on CPU, with no download. [Score your own model on real scenes →](getting-started/README.md)

<p class="rpx-eyebrow">01 / THE FINDING</p>

## One scene. A changing world.

Each scene is recorded before manipulation (Clutter), while hands move objects (Interaction) and afterwards (Clean), plus an egocentric view. Here is what the paper found:

<figure class="rpx-workflow-figure rpx-finding"><img src="assets/rpx-finding-light.webp#only-light" width="2400" height="1482" alt="Scene 012 in Clutter, Interaction, Clean and egocentric views with mean quality per view for video depth, tracking and VQA; Interaction is lowest. A map of all 65 evaluations by worst-phase quality and phase robustness shows none in the golden zone."><img src="assets/rpx-finding-dark.webp#only-dark" width="2400" height="1482" alt="Scene 012 in Clutter, Interaction, Clean and egocentric views with mean quality per view for video depth, tracking and VQA; Interaction is lowest. A map of all 65 evaluations by worst-phase quality and phase robustness shows none in the golden zone."><figcaption>Bars: mean quality 𝒥 over all evaluated models, from the paper's Tables III–VI. Map: all 65 evaluations from the paper's Fig. 4. The four views of indoor scene 012 are not frame-synchronized.</figcaption></figure>

<p class="rpx-eyebrow">02 / THE TOOLKIT</p>

## From predictions to evidence.

<figure class="rpx-workflow-figure"><a href="assets/toolkit-overview.svg"><img src="assets/toolkit-overview.svg" alt="RPX data with scene012 feeds six task model interfaces, evaluation, Phi/JEDI calculation and hardware evidence."></a><figcaption>Bring your model to RPX task data. Adapters connect inference to shared scoring; annotation, profiling and extension tools complete the workflow. Open the vector figure to zoom.</figcaption></figure>

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

<p class="rpx-eyebrow">04 / NEXT STEPS</p>

## Where to go next

<div class="rpx-routes">
<a href="getting-started/"><span class="rpx-route-number">01</span><span><strong>Run a benchmark</strong><small>Install, score a model on real scenes, read the results.</small></span><span aria-hidden="true">↗</span></a>
<a href="models/"><span class="rpx-route-number">02</span><span><strong>Bring your own model</strong><small>Wrap a local callable or an API client.</small></span><span aria-hidden="true">↗</span></a>
<a href="benchmarks/"><span class="rpx-route-number">03</span><span><strong>Six task walkthroughs</strong><small>Smoke commands and real examples for every paper task.</small></span><span aria-hidden="true">↗</span></a>
<a href="analysis/"><span class="rpx-route-number">04</span><span><strong>Φ/JEDI calculator</strong><small>Robustness and worst-phase quality from raw metrics.</small></span><span aria-hidden="true">↗</span></a>
<a href="profiling/"><span class="rpx-route-number">05</span><span><strong>Hardware profiler</strong><small>Latency, memory and system cards, warmup excluded.</small></span><span aria-hidden="true">↗</span></a>
<a href="annotation/"><span class="rpx-route-number">06</span><span><strong>Mask annotation pipeline</strong><small>Box curation, SAM2 propagation, review and identity mapping.</small></span><span aria-hidden="true">↗</span></a>
<a href="metrics/"><span class="rpx-route-number">07</span><span><strong>Add a metric</strong><small>Register a calculator with a tested scoring contract.</small></span><span aria-hidden="true">↗</span></a>
<a href="tasks/"><span class="rpx-route-number">08</span><span><strong>Extend the tasks</strong><small>Define inputs, ground truth, adapters and evaluation.</small></span><span aria-hidden="true">↗</span></a>
</div>

Everything else, including data tools, batch adapters and visualization, is under [all capabilities](capabilities/README.md).

??? info "Concepts and acronyms"

    - **MOS / SOS:** multi-object scenes / single-object captures.
    - **Phase:** Clutter, Interaction, or Clean. Egocentric Interaction is a separate view.
    - **Easy / Medium / Hard:** difficulty tiers of 33, 33 and 34 scenes, stratified by how much correction each scene's masks needed.
    - **JEDI / J / Jmin:** Joint Empirical Desirability Index; overall mean cell desirability and the lowest phase-mean quality.
    - **Φ:** phase robustness from repeated-measures analysis of paired raw metrics. High robustness must be considered alongside prediction quality.
    - **GT:** ground truth. Keep units, coordinates, and object identities consistent across predictions and GT.

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
