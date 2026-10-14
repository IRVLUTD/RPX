# Tracking (T3)

Released inputs for the object-tracking task that are not part of the dataset itself.

| Folder | Contents |
|:--|:--|
| [`metadata/text_initialization_v1/`](metadata/text_initialization_v1/) | Fixed text prompts (`black hammer`, …) for text-prompted trackers, one per object per clip |

The tracking evaluator and reference runners are in [`benchmark/`](../benchmark/README.md). Docker images for the
paper's trackers are listed in the [Docker guide](../docker/README.md#which-image-to-pull).
