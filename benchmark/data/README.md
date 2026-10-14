# Dataset metadata

The RGB-D data and VQA Parquets live on [IRVLUTD/RPX](https://huggingface.co/datasets/IRVLUTD/RPX). This directory
keeps the small metadata files the toolkit and its provenance depend on.

| Path | Contents |
|---|---|
| `splits/scene_splits.json` | The easy, medium and hard scene tiers (33, 33 and 34 scenes) by released `sceneNNN` ID. The `.txt` files list the same scenes for loaders. |
| `release/manifest/` | Metadata snapshot of the release: frame indexes, the mapping from old to released scene names, the object catalog, and mask-to-object joins. The dataset manifest records its schema and label versions. |
| `release/phase_splits/` | The original per-phase splits. They differ from the scene tiers and are kept for provenance only; the benchmark doesn't read them. |
| `release/objects_meta/` | Metadata and questionnaires for the selected objects. |
| `vqa_smoke/v1/` | A small fixed VQA sample for mechanical tests ([details](vqa_smoke/v1/README.md)). |

Tracking's text prompts are versioned separately, with their own checksums, in
[`tracking/metadata/`](../../tracking/metadata/).

These files describe the dataset snapshot. They don't replace the manifests a run downloads at its pinned Hub
revision, which the benchmark records with every result. To generate new task manifests, see
`python -m rpx_benchmark.dataset_hub.cli manifest --help` and pass explicit dataset and staging paths. Don't write
predictions into this directory.
