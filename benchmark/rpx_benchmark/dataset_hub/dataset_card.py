"""Generate the HuggingFace dataset card (`README.md`) at the repo root.

The dataset card is what HuggingFace renders on the dataset page and what
the Dataset Viewer parses for tags, task categories, and split sizes.
HF requires it as ``README.md`` at the repo root with YAML frontmatter.

Inputs to the generator:

* :class:`ScanResult` — for scene counts, modality file counts, byte totals.
* Optional :class:`PackResult` — for shard counts (more accurate than the
  scan when the packer collapses categories like sam2/* into masks_aux).
* Optional ``splits`` mapping — to populate the per-tier scene counts.
* Optional ``label_versions`` — to render the label versioning section.

Output: ``<staging>/README.md``. Idempotent (overwrites with ``overwrite=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from ..exceptions import DatasetError
from ..logging_utils import get_logger
from .recipes import (
    DEFAULT_REPO_ID,
    MULTI_OBJECT_TASK_RECIPES,
    SINGLE_OBJECT_TASK_RECIPES,
    SceneType,
)
from .scanner import ScanResult

log = get_logger(__name__)


DEFAULT_LICENSE = "cc-by-4.0"
DEFAULT_PRETTY_NAME = "RPX: Robot Perception X"

DEFAULT_TASK_CATEGORIES = (
    "image-segmentation",
    "depth-estimation",
    "object-detection",
    "visual-question-answering",
)
DEFAULT_TAGS = (
    "robotics",
    "embodied-ai",
    "rgb-d",
    "benchmark",
    "perception",
    "manipulation",
    "stereo",
)


@dataclass(frozen=True)
class CardSpec:
    """Knobs for the card generator."""

    repo_id: str = DEFAULT_REPO_ID
    pretty_name: str = DEFAULT_PRETTY_NAME
    license: str = DEFAULT_LICENSE
    task_categories: tuple[str, ...] = DEFAULT_TASK_CATEGORIES
    tags: tuple[str, ...] = DEFAULT_TAGS
    paper_url: Optional[str] = None
    code_url: Optional[str] = None


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


def _size_category(total_bytes: int) -> str:
    """HF size_categories tag bucket (used for search filters)."""
    gb = total_bytes / (1024**3)
    if gb < 1:
        return "100M<n<1B"
    if gb < 10:
        return "1B<n<10B"
    if gb < 100:
        return "10B<n<100B"
    if gb < 1024:
        return "100B<n<1T"
    return "n>1T"


def _yaml_list(values) -> str:
    return "\n".join(f"- {v}" for v in values)


def _frontmatter(spec: CardSpec, scan: ScanResult, splits: Mapping[str, str] | None = None) -> str:
    n_multi = len(scan.by_type(SceneType.MULTI_OBJECT))
    n_single = len(scan.by_type(SceneType.SINGLE_OBJECT))

    lines = [
        "---",
        f"license: {spec.license}",
        f'pretty_name: "{spec.pretty_name}"',
        "task_categories:",
        _yaml_list(spec.task_categories),
        "language:",
        "- en",
        "tags:",
        _yaml_list(spec.tags),
        "size_categories:",
        f"- {_size_category(scan.total_bytes)}",
    ]

    # If splits are provided, advertise them so HF Datasets viewer
    # surfaces tier filters in the navigator.
    if splits and n_multi > 0:
        tier_counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        for s in scan.by_type(SceneType.MULTI_OBJECT):
            tier = splits.get(s.scene_id)
            if tier in tier_counts:
                tier_counts[tier] += 1
        lines.append("configs:")
        lines.append("- config_name: multi_object")
        lines.append(
            '  description: "MOS scenes (3 phases each), '
            'tertile-cut by Effort-Stratified Difficulty."'
        )
        lines.append("  data_files:")
        for tier, n in tier_counts.items():
            if n:
                lines.append(f'  - split: "{tier}"')
                lines.append(f'    path: "splits/{tier}.txt"')
        lines.append("- config_name: single_object")
        lines.append(
            '  description: "SOS scenes (one 360° collection per object); no difficulty split."'
        )
    lines.append("---")
    return "\n".join(lines)


def _body(
    spec: CardSpec,
    scan: ScanResult,
    splits: Mapping[str, str] | None = None,
    label_versions: Mapping[str, str] | None = None,
) -> str:
    n_multi = len(scan.by_type(SceneType.MULTI_OBJECT))
    n_single = len(scan.by_type(SceneType.SINGLE_OBJECT))
    modalities = scan.modality_totals()

    mod_table_rows = []
    for name, inv in modalities.items():
        mod_table_rows.append(
            f"| `{name}` | {inv.file_count:,} | {_human_bytes(inv.total_bytes)} |"
        )
    mod_table = "\n".join(mod_table_rows) if mod_table_rows else "*(none scanned)*"

    multi_recipe_rows = "\n".join(
        f"| `{name}` | {sorted(r.inputs)} → {sorted(r.labels)} |"
        for name, r in MULTI_OBJECT_TASK_RECIPES.items()
    )
    single_recipe_rows = "\n".join(
        f"| `{name}` | {sorted(r.inputs)} → {sorted(r.labels)} |"
        for name, r in SINGLE_OBJECT_TASK_RECIPES.items()
    )
    label_v_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in (label_versions or {}).items())
    label_v_block = (
        "| modality | current version |\n|---|---|\n" + label_v_rows
        if label_v_rows
        else "*(default versions are in `manifest/current.json`)*"
    )

    paper_line = f"\n* Paper: {spec.paper_url}\n" if spec.paper_url else ""
    code_line = f"* Code: {spec.code_url}\n" if spec.code_url else ""

    return f"""
# {spec.pretty_name}

A real-world RGB-D benchmark for evaluating robot perception under
embodied deployment conditions.

{paper_line}{code_line}

## Dataset at a glance

| | |
|---|---|
| Multi-object scenes (MOS) | **{n_multi}** (3 phases each: clutter / interaction / clean) |
| Single-object scenes (SOS) | **{n_single}** (one 360° collection per object) |
| Total files | **{scan.file_count:,}** |
| Total bytes | **{_human_bytes(scan.total_bytes)}** |

## Modality inventory

Three lossless space-saving steps were applied to this release; each is
**bit-identical to the source** at the decoded pixel/value level
(verified per-frame at conversion time):

* `rgb/`, `fisheye/`, `ego/rgb/` are stored as **WebP-lossless**
  (~45-55% smaller than the source PNGs on real-world photo content).
* `depth/` and `sam2/masks/` are stored as **PNG re-encoded at
  `compress_level=9`** (~20% smaller; same PNG file format, palette /
  16-bit modes preserved exactly).
* `cam_pose/` per-frame `.npz` files are stored as per-frame `.npy`
  (~68% smaller; the per-file zip wrapper is removed and the position
  and orientation are concatenated into one `(7,) float64` vector
  packing `[x, y, z, qx, qy, qz, qw]`).

Loaders see identical numpy arrays in every modality.

| modality | files | bytes |
|---|---:|---:|
{mod_table}

## Quick start

```bash
pip install "rpx-benchmark[hub]"
hf auth login
```

```python
from rpx_benchmark.dataset_hub import download_for_task

# Pull just RGB + masks for the Easy difficulty tier — never the whole repo.
res = download_for_task(task="segmentation", split="easy",
                          repo_id="{spec.repo_id}")
print(res.local_dir, res.matched_scenes)
```

```bash
# Or from the CLI:
python -m rpx_benchmark.dataset_hub.cli download \\
    --task segmentation --split easy \\
    --repo-id {spec.repo_id}
```

A subsequent call for a different task on the same split (e.g.
`relative_pose`) reuses the cached RGB tars and only fetches the new
modality (`cam_pose`) as the delta.

## Repo layout

```
{spec.repo_id}/
├── manifest/
│   ├── frames_v1.parquet     # per-frame metadata (always pulled, ~30 MB)
│   └── current.json          # default version per label modality
├── splits/
│   ├── scene_splits.json
│   ├── easy.txt  medium.txt  hard.txt
├── scenes/<scene_id>/<phase>/                     # MOS
│   ├── rgb.tar  depth.tar  fisheye.tar
│   └── labels/{{cam_pose,masks,masks_aux,sam2_meta,vqa}}/v1.tar
├── objects/<object_id>/0/                         # SOS
│   └── (same modality structure)
├── objects_meta/                                  # questionnaire dedup
│   ├── _index.json
│   └── <object_id>/questionnaire.json
└── README.md   ←  this file
```

## Tasks

### Multi-object (use a difficulty split)

| recipe | inputs → labels |
|---|---|
{multi_recipe_rows}

### Single-object (no split — these are object templates)

| recipe | inputs → labels |
|---|---|
{single_recipe_rows}

## Label versioning

Labels live at `labels/<name>/v<N>.tar`. Newer versions land at new
paths; old versions stay reachable for reproducibility.

{label_v_block}

To pin to a specific version:

```python
download_for_task(
    task="relative_pose", split="easy", repo_id="{spec.repo_id}",
    label_versions={{"cam_pose": "v1"}},   # don't auto-upgrade to v2
)
```

## Citation

```bibtex
@misc{{rpx2026,
    title  = {{RPX: Robot Perception X — A real-world RGB-D benchmark for
              embodied perception}},
    author = {{Anonymous Institution}},
    year   = 2026,
    url    = {{https://huggingface.co/datasets/{spec.repo_id}}},
}}
```

## License

Released under the **{spec.license}** license.
"""


def write_dataset_card(
    staging_root: Path,
    scan: ScanResult,
    spec: Optional[CardSpec] = None,
    splits: Optional[Mapping[str, str]] = None,
    label_versions: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
) -> Path:
    """Render and write ``<staging_root>/README.md``.

    Returns the destination path.
    """
    spec = spec or CardSpec()
    out_path = Path(staging_root) / "README.md"
    if out_path.exists() and not overwrite:
        raise DatasetError(
            f"refusing to overwrite existing dataset card: {out_path}",
            hint="Pass overwrite=True or remove the file first.",
        )

    contents = (
        _frontmatter(spec, scan, splits)
        + "\n"
        + _body(
            spec,
            scan,
            splits,
            label_versions,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(contents, encoding="utf-8")
    log.info("wrote dataset card → %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path
