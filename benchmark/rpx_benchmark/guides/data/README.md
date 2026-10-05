# Data and annotations

RPX follows the same objects across scene phases. Its inputs and annotations support several perception tasks on shared captures, rather than unrelated task datasets.

## Capture views

Exocentric Clutter, Interaction, and Clean provide RGB-D, masks, stereo, and pose. Egocentric RGB is captured during Interaction on an independent clock. Do not assume frame-level synchronization between the ego and exo streams.

<figure class="rpx-wide-figure"><img src="../assets/interaction.jpg" width="640" height="480" alt="Exocentric scene 091 during interaction, with released instance-mask overlays." loading="lazy"><figcaption>Scene 091 / Interaction. Instance-mask overlays show the object identities that tracking and grounding rely on.</figcaption></figure>

## Metric depth and instance masks

The RGB and colorized-depth examples below are from a single-object capture. Colorized depth is a display aid; a benchmark must load the metric ground truth and honor its validity rules. Mask values encode instance identities, not depth or confidence.

<div class="rpx-modalities rpx-three">
<figure><img src="../assets/rgb.jpg" width="640" height="480" loading="lazy" alt="RGB single-object capture."><figcaption><span>RGB</span>Appearance</figcaption></figure>
<figure><img src="../assets/depth.png" width="640" height="480" loading="lazy" alt="Colorized metric-depth example."><figcaption><span>DEPTH</span>Metric geometry, visualized</figcaption></figure>
<figure><img src="../assets/mask.png" width="640" height="480" loading="lazy" alt="Binary instance-mask visualization of the object in this single-object capture."><figcaption><span>INSTANCE MASK</span>Object pixels and identity</figcaption></figure>
</div>

Masks connect local frame instances to persistent/global identities. The overlays on the [overview](../README.md) make these labels visible; the benchmark consumes the released mask IDs rather than rendered colors.

## How masks are annotated

<figure class="rpx-wide-figure"><a href="../assets/mask-annotation.jpg"><img src="../assets/mask-annotation.jpg" width="3757" height="1519" loading="lazy" alt="Seven-stage mask annotation workflow: generate boxes, verify initial masks, propagate with SAM2, verify frame by frame, automatically refine, manually relabel, and release verified masks."></a><figcaption>RPX's existing annotation-pipeline figure. Open the image for the full-resolution labels.</figcaption></figure>

Automatic propagation is combined with human verification and correction. This figure describes dataset annotation, not the inference pipeline of every benchmark model.

## Stereo and pose

<div class="rpx-modalities">
<figure><img src="../assets/stereo-left.jpg" width="848" height="800" loading="lazy" alt="Left calibrated T265 fisheye image."><figcaption><span>LEFT</span>Fisheye stereo</figcaption></figure>
<figure><img src="../assets/stereo-right.jpg" width="848" height="800" loading="lazy" alt="Right calibrated T265 fisheye image."><figcaption><span>RIGHT</span>Fisheye stereo</figcaption></figure>
</div>

Relative-pose evaluation uses the task-specific image pairing and pose ground-truth protocol. Having a stereo or pose modality in a capture does not by itself define the evaluation split or eligible phase pairs.

## Check a dataset before benchmarking

Verify modality paths, shape, depth units, camera-coordinate conventions, and joins between scene IDs, frame IDs, mask IDs, and object IDs. Use the same versioned split and validity rules across models. The dataset loader and task APIs define how those contracts enter a run.

[Get started](../getting-started/README.md) · [Model integration](../models/README.md) · [Hub API](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/hub.html)

The images on these pages come from the existing RPX appendix assets. Their source files and hashes are recorded in [asset provenance](../assets/provenance.json).
