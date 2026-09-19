# RPX consolidation: `naren/push`

Snapshot date: 2026-09-19. Base: `origin/main` at `719d0da`.

This branch consolidates the code in the local `rpx` workspace and the current
remote descendants of its active development branches. The parent directory is
an approximately 717 GiB Hugging Face dataset checkout, not the GitHub code
repository. Its Git history, shards, model weights, environments and caches do
not belong in the GitHub code tree.

## Repository structure

- `benchmark/`: the integrated Python library, model adapters, evaluation runners,
  tests, runbooks, and local SOS pose-audit results.
- `docker/`: depth, video-depth, NVS, RCPE, tracking and VQA runtime images and gates.
- `data/`: existing capture, annotation and VQA ground-truth tooling.
- `tracking/metadata/text_initialization_v1/`: the versioned text vocabulary release.
- `workspace/`: standalone local release/QA scripts, historical reports, lightweight
  dataset metadata, sample VQA tables and presentation assets. See
  [workspace overview](../../workspace/OVERVIEW.md).

## Branch inventory

Full SHAs and inclusion decisions are recorded in
[`workspace/branch_inventory.json`](../../workspace/branch_inventory.json).

| Remote branch | Revision | Disposition |
| --- | --- | --- |
| `docker/depth-all-hyden` | `1a1949c65506` | Included through merge ancestry |
| `docker/tracking-all-yoloe` | `2b90d59d26a8` | Included through merge ancestry |
| `itay/ego-dataset-hub-support` | `f6d76c7c0dd3` | Included through merge ancestry |
| `itay/maskgen` | `d53285395550` | Historical branch; functionality subsequently imported/refactored on main; not reintroduced wholesale |
| `itay/vqa` | `4f768e7136ef` | Included through merge ancestry |
| `itay/vqa-incontext` | `de59a5111a21` | Included through merge ancestry |
| `jikai/relative_camera_pose` | `5ce24d55bb45` | Included through merge ancestry |
| `jishnu/benchmark-code-skeleton` | `0506ab9e6b6c` | Historical branch; functionality subsequently imported/refactored on main; not reintroduced wholesale |
| `jishnu/video-temporal-framework` | `5951f2efae1c` | Included through merge ancestry |
| `main` | `719d0daf351b` | Included through merge ancestry |
| `naren/depth_pipeline_check` | `74e00b2233e4` | Included through merge ancestry |
| `naren/nvs-pipeline-additions` | `c8aebe4fa8f0` | Included through merge ancestry |
| `naren/rcpe-pipeline-additions` | `5ebd50f699f5` | Included through merge ancestry |
| `naren/tracking-bbox-init` | `0be831d15391` | Included through merge ancestry |
| `naren/tracking-sam31-empty-points` | `2bd95e5499f7` | Included through merge ancestry |
| `naren/tracking-text-models` | `1f8043e089dd` | Included through merge ancestry |
| `naren/tracking-text-vocab-v1` | `7646e1a1ae5e` | Included through merge ancestry |
| `naren/vqa-deepseek-vl2` | `78e831f7a439` | Included through merge ancestry |
| `naren/vqa-deepseek-vl2-full-config` | `4e3b48361caa` | Included through merge ancestry |
| `naren/vqa-force-bbox` | `c7155bec4b37` | Included through merge ancestry |
| `naren/vqa-internvl1b-server1-dual-gpu` | `786388806a2a` | Included through merge ancestry |
| `naren/vqa-molmo` | `c2e57c88c258` | Included through merge ancestry |
| `naren/vqa-molmo-no-tensorflow` | `715213b1ca62` | Included through merge ancestry |
| `smoke/depth-models-2026-06-29` | `cb77db974ce0` | Included through merge ancestry |

The historical `itay/maskgen` and `jishnu/benchmark-code-skeleton` branches predate
main's squash imports and later cleanup/refactors (including PRs #23, #33, #34,
#35 and #36). Merging their old trees would reintroduce removed vendor code,
duplicate packages and obsolete layouts. They remain available on the remote;
this branch retains main's current annotation and dataset-hub implementations.

## Local worktrees

| Source folder | Local branch | Local work incorporated |
| --- | --- | --- |
| `github_rpx` | `naren/depth_pipeline_check` | Four untracked VQA draft files preserved under `workspace/legacy_vqa_roster/` |
| `github_rpx_depth_smoke` | `docker/depth-all-hyden` | Committed history included through pipeline descendants |
| `github_rpx_tracking` | `docker/tracking-all-yoloe` | Committed history plus newer remote fixes included |
| `github_rpx_tracking_text` | `naren/tracking-text-models` | Committed history plus bbox and empty-point fixes included |
| `github_rpx_video_temporal` | `naren/nvs-pipeline-additions` | Nine modified source/docs files, new audit/visualization tools, NVS protocol tests, frame-budget runbook and audit reports |
| `github_rpx_vqa_prompt_fix` | `naren/vqa-force-bbox` | Current remote descendants including in-context, Molmo, DeepSeek, InternVL and dual-GPU support |

Two capture NPZ samples are missing in all six original worktrees. Their deletion
was not propagated: they are existing upstream samples, not source changes.
The original worktrees and dataset checkout were left untouched.

## Conflict resolutions

- Combined FE2E/ZipDepth with DVD/GemDepth runtime additions. The generic depth
  environment inventory excludes FE2E and GemDepth, which have dedicated overlays.
  No canonical model is marked weight-blocked by the merged branches.
- Retained `ConfigTypeError` for invalid system cards (also a `ConfigError`).
- Retained the current InternVL/dual-GPU runtime. Commit `68da9f7` explicitly
  replaced Molmo in the active roster; the divergent older Molmo isolation
  recipe, entrypoint, backend and tests are preserved under
  `workspace/legacy_molmo_runtime/` instead of reactivating superseded models.
- Combined ego-aware tracking paths with the tracking pipeline's removal of the
  nonexistent `tracklets/v1.json` requirement.
- Applied uncommitted NVS/video work using a three-way patch, preserving newer
  RCPE changes.
- Archived the old uncommitted VQA roster separately because its schema and
  ten-model proposal conflict with the current runtime matrix. Its original
  tests and files remain byte-for-byte snapshots, not active benchmark tests.

## External files and exclusions

[`workspace/imported_files.json`](../../workspace/imported_files.json) records
source/destination paths, sizes and SHA-256 hashes for copied files.
[`workspace/folder_inventory.json`](../../workspace/folder_inventory.json)
accounts for every original top-level entry. Unmodified downloaded third-party
repositories are pinned in
[`workspace/third_party_sources.json`](../../workspace/third_party_sources.json).

Dataset tar shards, raw ego captures, checkpoint weights, Python environments,
Docker backups, cache directories, build logs, Rerun recordings, per-frame
prediction binaries and bulk result Parquet files remain local. Lightweight
manifests, splits, object metadata, VQA sample tables, text/JSON/CSV/TSV reports,
and selected SOS audit figures are included. Existing Hugging Face LFS rules
were not copied into the GitHub repository.

Historical reports are preserved as recorded; their reported experiments were
not rerun during consolidation. Some standalone scripts contain original host
paths. See the workspace overview before executing them.

## Validation

Checks ran locally on Python 3.13. GPU inference, image builds, and the project's
Python 3.10–3.12 CI matrix were not run.

- **Integrity:** 638 copied files match their recorded SHA-256 hashes; all nine
  locally modified NVS/video source files match their original working copies.
- **Syntax:** all 517 staged Python files parse and all 61 shell scripts pass
  `bash -n`. No unresolved merge markers in source. The existing vendored
  `munkres.py` emits a Python escape-sequence warning.
- **Focused integration:** 178 passed, 2 skipped, 1 deselected. The deselected
  RollingDepth test hits a local ImageIO/PyAV dependency mismatch; skipped checks
  need OpenCV or a generated VQA smoke manifest.
- **Broader pytest:** 1,271 passed, 4 failed, 14 skipped, 3 deselected, with the
  exclusions below. This is not a fully passing test suite.
- **Lint:** the source files manually edited to resolve conflicts pass Ruff.
  The full combined benchmark tree reports 52 lint findings, mostly inherited
  unused imports, import ordering, and loop/zip warnings.
- **Typing:** mypy 1.20.2 reports 54 errors in 13 files in the combined package.
- **Credential/size scan:** no matches for the checked private-key, GitHub,
  Hugging Face, AWS access-key or OpenAI token patterns; no staged file exceeds
  50 MiB. This is a pattern check, not a security audit.
- **Whitespace:** copied historical files preserve their original bytes,
  including CSV CRLF endings, Markdown hard breaks and a few trailing blank
  lines; these produce `git diff --check` findings in the snapshots.

The initial full test collection fails because `DepthLogMetrics`,
`DepthRangeStratified` and `video_depth_cell_row` are absent from the imported
implementations. They are also absent from the corresponding `main`,
video-temporal and RCPE source branches. `test_depth_log_range.py` and
`test_phase_cell_recording.py` were therefore excluded from the broader run.
`TestDepthGrasp3D` and `test_pipeline.py` exercise the existing quadratic,
50,000-point distance implementation and were excluded after prolonged runs.
No scientific metric formulas were changed to make the tests pass.

The four completed failures are:

1. RollingDepth's video-encoding test: ImageIO imports `av.VideoFormat`, missing
   in the local optional PyAV module.
2. Exception-style policy: the imported pose/audit modules raise standard
   `ValueError`/`RuntimeError` rather than RPX exception subclasses.
3. Paper aliases: the imported video-depth module lacks `evaluate_d1v_clip`
   (and the related clip/cell convenience APIs).
4. DA3 video adapter: the inherited test expects metric depth while the current
   adapter declares relative depth.

Detailed output is preserved under
[`validation/naren_push/`](validation/naren_push/). The completed broad check was:

```bash
cd benchmark
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m pytest tests -q --strict-markers --tb=short \
  --ignore=tests/test_depth_log_range.py \
  --ignore=tests/test_phase_cell_recording.py \
  --ignore=tests/test_pipeline.py -k 'not TestDepthGrasp3D'
```

These findings are retained for follow-up; this branch preserves and integrates
the requested code without silently changing evaluation protocols.
