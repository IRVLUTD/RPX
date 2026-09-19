# Dataset metadata

Large RGB-D assets and VQA Parquets are distributed through
[IRVLUTD/RPX](https://huggingface.co/datasets/IRVLUTD/RPX), not Git.

- `splits/scene_splits.json` is the canonical scene-level easy/medium/hard
  assignment: 33/33/34 scenes, using released `sceneNNN` identifiers. The text
  files are loader-compatible exports of exactly those lists.
- `release/manifest/` contains the retained dataset metadata snapshot: frame
  indexes, old-to-released scene-name mapping, object catalog and mask/object
  joins. The original dataset manifest records its schema and label versions.
- `release/phase_splits/` retains the original per-phase ESD assignments. These
  differ from the scene-level split and are provenance, not benchmark split inputs.
- `release/objects_meta/` stores the selected objects' metadata and questionnaires.
- Tracking's separately versioned vocabulary and its checksums live in
  [`tracking/metadata/`](../../tracking/metadata/).

These metadata files describe the consolidated dataset snapshot; they do not
replace the manifests downloaded at a run's pinned Hub revision. The benchmark
records that revision with outputs. Generate new task manifests using
`python -m rpx_benchmark.dataset_hub.cli manifest --help`; use explicit dataset
and staging paths. Do not write prediction outputs into this directory.
