"""File-staging helpers — non-tar artefacts that ship to the HF repo root.

The packer/manifest/uploader chain handles per-(scene, phase, modality)
tar shards and the per-frame Parquet manifest. This module covers the
small companion files HF Datasets needs alongside them:

* ``splits/{scene_splits.json, easy.txt, medium.txt, hard.txt}`` —
  copied from ``benchmark/data/splits/`` so downloaders can resolve a
  split to a scene list before fetching the manifest.
* ``rpx_croissant.json`` — copied from
  ``paper-submission/neurips-2026/croissant/`` with path reconciliation
  against the new tar layout (TODO 3 — handled in a sibling helper).
* ``README.md`` (HF dataset card) — generated from the scan totals
  (TODO 2 — handled in :mod:`.dataset_card`).
* ``preview/`` (HF Dataset Viewer thumbnails) — handled in
  :mod:`.preview` (TODO 4).

Each helper returns a list of :class:`StagedFile` records so the upload
step can verify what landed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..exceptions import ConfigError, DatasetError
from ..logging_utils import get_logger

log = get_logger(__name__)


# Default location of the splits files relative to the repo root. Resolved
# lazily so the helper still works when called from a non-repo cwd.
def _default_splits_src() -> Path:
    """``<repo_root>/benchmark/data/splits/``."""
    here = Path(__file__).resolve()
    # __file__ -> .../benchmark/rpx_benchmark/dataset_hub/staging.py
    benchmark_root = here.parents[2]
    return benchmark_root / "data" / "splits"


SPLIT_TIER_FILES = ("easy.txt", "medium.txt", "hard.txt")
SCENE_SPLITS_JSON = "scene_splits.json"


@dataclass(frozen=True)
class StagedFile:
    """One file the staging step copied / wrote."""

    repo_path: str  # path relative to staging root, forward-slashes
    src: Optional[Path]
    bytes_: int


def stage_splits(
    staging_root: Path,
    splits_src: Optional[Path] = None,
    overwrite: bool = False,
    require_all: bool = True,
) -> List[StagedFile]:
    """Copy ``splits/{scene_splits.json, easy.txt, medium.txt, hard.txt}``
    into ``<staging_root>/splits/``.

    Parameters
    ----------
    staging_root : Path
        Where the packed/upload-ready tree lives.
    splits_src : Path, optional
        Source dir holding the splits files. Defaults to
        ``benchmark/data/splits/`` next to this package.
    overwrite : bool
        If False (default), raises when a destination file already exists.
    require_all : bool
        If True (default), raises when any of the four expected files is
        missing from ``splits_src``. Set False for partial-stage scenarios
        (e.g., shipping with only the JSON manifest).

    Returns
    -------
    List[StagedFile]
        One entry per copied file.
    """
    src_root = Path(splits_src) if splits_src else _default_splits_src()
    if not src_root.is_dir():
        raise ConfigError(
            f"splits source dir does not exist: {src_root}",
            hint="Run experiments/scripts/build_difficulty_splits.py first, "
            "or pass splits_src= explicitly.",
        )

    staging_root = Path(staging_root)
    out_dir = staging_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = (SCENE_SPLITS_JSON, *SPLIT_TIER_FILES)
    missing = [name for name in expected if not (src_root / name).is_file()]
    if missing and require_all:
        raise DatasetError(
            f"missing splits files in {src_root}: {missing}",
            hint=(
                "Re-run build_difficulty_splits.py to regenerate them, or "
                "pass require_all=False to ship a partial splits dir."
            ),
        )

    staged: List[StagedFile] = []
    for name in expected:
        src = src_root / name
        if not src.is_file():
            log.warning("splits source missing: %s (skipping)", src)
            continue
        dst = out_dir / name
        if dst.exists() and not overwrite:
            raise DatasetError(
                f"refusing to overwrite existing staged file: {dst}",
                hint="Pass overwrite=True or remove the file first.",
            )
        shutil.copy2(src, dst)
        staged.append(
            StagedFile(
                repo_path=f"splits/{name}",
                src=src,
                bytes_=dst.stat().st_size,
            )
        )
        log.info("staged splits/%s (%d bytes)", name, dst.stat().st_size)
    return staged


def stage_paths(
    paths: Sequence[Path],
    staging_root: Path,
    repo_subdir: str = "",
    overwrite: bool = False,
) -> List[StagedFile]:
    """Generic helper: copy a list of files into ``<staging>/<repo_subdir>/``.

    Useful for ad-hoc staging of small files (Croissant JSON, custom
    docs) without writing a dedicated module. ``repo_subdir`` may be
    empty to land files at the staging root.
    """
    staging_root = Path(staging_root)
    target = staging_root / repo_subdir if repo_subdir else staging_root
    target.mkdir(parents=True, exist_ok=True)

    staged: List[StagedFile] = []
    for src in paths:
        src = Path(src)
        if not src.is_file():
            raise ConfigError(
                f"source file does not exist: {src}",
                hint="Check the path; staging does not create files.",
            )
        dst = target / src.name
        if dst.exists() and not overwrite:
            raise DatasetError(
                f"refusing to overwrite existing staged file: {dst}",
                hint="Pass overwrite=True or remove the file first.",
            )
        shutil.copy2(src, dst)
        rel = (Path(repo_subdir) / src.name).as_posix() if repo_subdir else src.name
        staged.append(
            StagedFile(
                repo_path=rel,
                src=src,
                bytes_=dst.stat().st_size,
            )
        )
        log.info("staged %s (%d bytes)", rel, dst.stat().st_size)
    return staged


def load_scene_splits(staging_root: Path) -> dict[str, str]:
    """Read ``<staging_root>/splits/scene_splits.json`` and return a
    flat ``{scene_id: tier}`` mapping. Convenience for downstream steps
    (manifest builder, dataset card) that need to look up a scene's tier.

    Accepts both the canonical schema with a top-level ``"splits"`` key
    holding ``{tier: [scene_id, ...]}`` and a flat ``{scene_id: tier}``
    shape.
    """
    path = Path(staging_root) / "splits" / SCENE_SPLITS_JSON
    if not path.is_file():
        raise ConfigError(
            f"scene_splits.json not found at {path}",
            hint="Run stage_splits() before load_scene_splits().",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in payload and isinstance(payload["splits"], dict):
        return {sid: tier for tier, ids in payload["splits"].items() for sid in ids}
    return dict(payload)
