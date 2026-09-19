# Local RPX workspace snapshot

This directory preserves standalone files from the local dataset workspace.
The maintained, integrated toolkit is in `../benchmark/` and its runtimes are in
`../docker/`. See [the consolidation inventory](../benchmark/docs/naren_push_consolidation.md).

- `ego_release/`: ego capture packing and release preparation.
- `rerun_check/`: download, structural validation and Rerun visualization helpers.
- `docker_mask_annotation_test/`: local annotation-image test scripts and notes.
- Root Python scripts: benchmark reports and depth/NVS visualizations.
- `manifest/`, `splits/`, `objects_meta/`: release metadata snapshots.
- `paper-metrics/`, `smoke-results*/` and root reports: historical result summaries.
  Bulk prediction arrays, result Parquets and dataset shards are not included.
- `legacy_vqa_roster/`: the original uncommitted VQA proposal, preserved with its
  original relative layout. Its model-matrix schema differs from the current
  VQA pipeline; it is a historical draft, not a runtime configuration.
- `legacy_molmo_runtime/`: the superseded Molmo runtime and isolation fix;
  the current VQA roster uses InternVL 3.5 in its place.
- `README.md`: the original Hugging Face dataset card, preserved unchanged.

These are snapshots, not a self-contained dataset checkout. Dataset helpers
expect real shards and captures at their configured dataset root. Pass their
`--repo-root`/input options where available. Several ad-hoc reporting and Docker
scripts contain the original `/media/naren/...` or `/home/naren/...` paths; review
and adjust those paths before running them on a different host. Commands in the
historical notes may refer to earlier branches or runtime states.

`imported_files.json` provides provenance and SHA-256 checksums. The folder and
branch inventories account for the original workspace and fetched remote refs.
