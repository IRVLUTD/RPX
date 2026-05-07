"""Stage the Croissant metadata JSON into the upload tree.

Croissant (https://mlcommons.org/working-groups/croissant/) is the ML
metadata standard HuggingFace renders alongside dataset cards and that
search platforms (Kaggle, Papers-with-Code) parse. We already authored
``paper-submission/neurips-2026/croissant/rpx_croissant.json`` against
an earlier dataset URL; this helper:

1. **Copies** the source JSON into ``<staging>/rpx_croissant.json``.
2. **Patches** the `url`, `version`, and `citeAs` fields to point at the
   HF repo (the source pre-dates the HF repo decision).
3. **Annotates** the `description` so consumers know that per-frame
   records now live inside per-modality tar shards rather than as loose
   files (the field-level path descriptions in the JSON still describe
   the logical type, but the on-disk shape changed under our packer).

A full path-by-path schema reconciliation against the new tar layout is
left to the dataset owner — it's a deeper edit that depends on what
loaders downstream consumers actually use, and is not blocking for the
v1 upload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..exceptions import ConfigError, DatasetError
from ..logging_utils import get_logger
from .recipes import DEFAULT_REPO_ID

log = get_logger(__name__)


# Default location of the source JSON, resolved relative to the repo root.
def _default_croissant_src() -> Path:
    """``<repo_root>/paper-submission/neurips-2026/croissant/rpx_croissant.json``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # .../benchmark/rpx_benchmark/dataset_hub/croissant.py
    return repo_root / "paper-submission" / "neurips-2026" / "croissant" / "rpx_croissant.json"


@dataclass(frozen=True)
class CroissantPatch:
    """Knobs for the staging step."""

    repo_id: str = DEFAULT_REPO_ID
    version: str = "1.0.0"
    cite_as: Optional[str] = None
    annotate_layout: bool = True


def _hf_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


_LAYOUT_NOTE = (
    "\n\nNote on on-disk shape: per-frame records are physically packed "
    "into per-(scene, phase, modality) tar shards on the HF repo "
    "(scenes/<scene>/<phase>/<modality>.tar for raw modalities; "
    "labels/<modality>/v<N>.tar for versioned labels). The "
    "rpx_benchmark.dataset_hub downloader resolves selective task+split "
    "downloads from these shards transparently."
)


def stage_croissant(
    staging_root: Path,
    src: Optional[Path] = None,
    patch: Optional[CroissantPatch] = None,
    overwrite: bool = False,
) -> Path:
    """Copy + patch the Croissant JSON into ``<staging>/rpx_croissant.json``.

    Returns the destination path.
    """
    src = Path(src) if src else _default_croissant_src()
    patch = patch or CroissantPatch()
    if not src.is_file():
        raise ConfigError(
            f"croissant source JSON does not exist: {src}",
            hint=(
                "Pass src= explicitly, or check that "
                "paper-submission/neurips-2026/croissant/ is present "
                "in the repo."
            ),
        )

    out_path = Path(staging_root) / "rpx_croissant.json"
    if out_path.exists() and not overwrite:
        raise DatasetError(
            f"refusing to overwrite existing croissant: {out_path}",
            hint="Pass overwrite=True or remove the file first.",
        )

    payload = json.loads(src.read_text(encoding="utf-8"))

    payload["url"] = _hf_url(patch.repo_id)
    payload["version"] = patch.version
    if patch.cite_as is not None:
        payload["citeAs"] = patch.cite_as

    if patch.annotate_layout:
        desc = payload.get("description", "")
        if _LAYOUT_NOTE.strip() not in desc:
            payload["description"] = desc + _LAYOUT_NOTE

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("staged croissant → %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path
