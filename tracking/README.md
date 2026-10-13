# Tracking (T3) assets

Released inputs for RPX's object-tracking task beyond the dataset itself.

| Folder | What it holds |
|:--|:--|
| [`metadata/text_initialization_v1/`](metadata/text_initialization_v1/) | Fixed text prompts (`black hammer`, …) for text-prompted trackers, one per object per clip |

The tracking evaluator and reference runners live in [`benchmark/`](../benchmark/README.md); the paper's tracker
images are in the [Docker guide](../docker/README.md#which-image-to-pull).
