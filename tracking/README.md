# Tracking (T3)

Released inputs for the object-tracking task that are not part of the dataset itself.

| Folder | Contents |
|:--|:--|
| [`metadata/text_initialization_v1/`](metadata/text_initialization_v1/) | Fixed text prompts for text-prompted trackers |

The prompts (`black hammer`, …) are fixed, one per object per clip.

The tracking evaluator and reference runners are in [`benchmark/`](../benchmark/README.md). Docker images
for the paper's trackers are listed in the [Docker guide](../docker/README.md#which-image-to-pull).

---

<sub>Part of <a href="../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>
