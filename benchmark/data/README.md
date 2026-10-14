# Dataset metadata

The RGB-D data and VQA Parquets live on [IRVLUTD/RPX](https://huggingface.co/datasets/IRVLUTD/RPX).
This directory keeps the small metadata files the toolkit and its provenance depend on.

| Path | Contents |
|:--|:--|
| `splits/scene_splits.json` | Easy, Medium and Hard scene tiers |
| `release/manifest/` | metadata snapshot of the release |
| `release/phase_splits/` | original per-phase splits, provenance only |
| `release/objects_meta/` | object metadata and questionnaires |
| [`vqa_smoke/v1/`](vqa_smoke/v1/README.md) | small fixed VQA sample for mechanical tests |

The tiers hold 33, 33 and 34 scenes by released `sceneNNN` ID; the `.txt` files next to
`scene_splits.json` list the same scenes for loaders. The release manifest snapshot holds the frame
indexes, the mapping from old to released scene names, the object catalog and the mask-to-object
joins, and records its schema and label versions. The per-phase splits differ from the scene tiers;
the benchmark doesn't read them.

Tracking's text prompts are versioned separately, with their own checksums, in
[`tracking/metadata/`](../../tracking/metadata/).

These files describe the dataset snapshot. They don't replace the manifests a run downloads at its
pinned Hub revision, which the benchmark records with every result. To generate new task manifests,
see `python -m rpx_benchmark.dataset_hub.cli manifest --help` and pass explicit dataset and staging
paths. Don't write predictions into this directory.

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>
