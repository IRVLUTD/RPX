"""Build the per-frame Parquet manifest the dataset hub ships.

The manifest is the *single source of truth* for what's in the dataset:
one row per frame, with boolean columns marking which modalities are
present, plus split assignment, scene type, and pointers to the tar
shards that contain the actual bytes. It is shipped at
``manifest/frames_<schema_version>.parquet`` on the HF repo and is
small enough (~30 MB for 750k frames) that the downloader pulls it
eagerly before deciding what else to fetch.

Schema (column / dtype / meaning)::

    scene_id        str          "scene1.library.fountain" or "object001.mug"
    scene_type      str          "multi_object" | "single_object"
    phase           int32        0/1/2 for multi, 0 for single
    frame_idx       int32        0-based frame index inside the phase
    frame_filename  str          the source filename (e.g. "00000.png")
    split           str | null   "easy" | "medium" | "hard" (null if no split)
    has_rgb         bool         True iff the frame appears in rgb.tar
    has_depth       bool
    has_fisheye     bool
    has_cam_pose    bool
    has_masks       bool
    has_masks_aux   bool
    has_sam2_meta   bool
    shard_rgb       str | null   repo-relative path to rgb.tar (or null)
    shard_depth     str | null
    shard_fisheye   str | null
    shard_cam_pose  str | null
    shard_masks     str | null
    shard_masks_aux str | null
    shard_sam2_meta str | null

Plus a sibling ``manifest/current.json`` that maps logical label names
to the version currently considered "default" (e.g. ``masks → v1``).
The downloader reads this first to resolve recipe modalities to actual
shard paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ..exceptions import DatasetError
from ..logging_utils import get_logger
from .packer import PackedShard, PackResult
from .recipes import (
    CAM_POSE,
    DEPTH,
    FISHEYE,
    MASKS,
    MASKS_AUX,
    RGB,
    SceneType,
)  # noqa: F401  CAM_POSE used in default label_versions
from .scanner import ScanResult

log = get_logger(__name__)


SCHEMA_VERSION = "v1"

# Modality → ("has_X", "shard_X") column-name pair. Order is the canonical
# column order in the Parquet table.
_COLUMNS: Dict[str, tuple[str, str]] = {
    RGB: ("has_rgb", "shard_rgb"),
    DEPTH: ("has_depth", "shard_depth"),
    FISHEYE: ("has_fisheye", "shard_fisheye"),
    CAM_POSE: ("has_cam_pose", "shard_cam_pose"),
    MASKS: ("has_masks", "shard_masks"),
    MASKS_AUX: ("has_masks_aux", "shard_masks_aux"),
    "sam2_meta": ("has_sam2_meta", "shard_sam2_meta"),
}


@dataclass(frozen=True)
class ManifestPaths:
    """Where the manifest writer dropped its outputs."""

    parquet_path: Path
    current_json_path: Path


def _arrow():
    """Lazy-import pyarrow with a friendly error message."""
    try:
        import pyarrow as pa  # noqa: WPS433
        import pyarrow.parquet as pq  # noqa: WPS433
    except ImportError as e:
        raise DatasetError(
            "pyarrow is required to write the dataset manifest.",
            hint="Install with: pip install 'rpx-benchmark[hub]'",
        ) from e
    return pa, pq


def _index_shards_by_phase(
    pack: PackResult,
) -> Dict[tuple[str, int], Dict[str, PackedShard]]:
    """Group shards by (scene_id, phase) for fast lookup while walking frames."""
    out: Dict[tuple[str, int], Dict[str, PackedShard]] = {}
    for s in pack.shards:
        key = (s.scene_id, s.phase)
        out.setdefault(key, {})[s.modality] = s
    return out


def _frame_filenames_for(
    scan: ScanResult,
    scene_id: str,
    phase_index: int,
) -> List[str]:
    """Pick one modality (rgb, then depth, then anything) and return its
    sorted filenames as the canonical frame list for that phase.
    """
    scene = next(s for s in scan.scenes if s.scene_id == scene_id)
    phase = next(p for p in scene.phases if p.phase_index == phase_index)
    sub = "mos" if scene.scene_type is SceneType.MULTI_OBJECT else "sos"
    src = scan.root / sub / scene_id / str(phase_index)
    for prefer in ("rgb", "depth", "fisheye"):
        if prefer in phase.modalities:
            return sorted(p.name for p in (src / prefer).iterdir() if p.is_file())
    return []


def build_frame_manifest(
    scan: ScanResult,
    pack: PackResult,
    out_dir: Path,
    splits: Optional[Mapping[str, str]] = None,
    label_versions: Optional[Mapping[str, str]] = None,
) -> ManifestPaths:
    """Write ``manifest/frames_<v>.parquet`` and ``manifest/current.json``.

    Parameters
    ----------
    scan : ScanResult
        Output of :func:`scan_capture_root`.
    pack : PackResult
        Output of :func:`pack_capture_tree`.
    out_dir : Path
        Where to write ``frames_<v>.parquet`` and ``current.json``. Both
        end up under ``out_dir / "manifest" / ...``.
    splits : Mapping[str, str], optional
        ``{scene_id: "easy" | "medium" | "hard"}`` for multi-object scenes.
        Single-object scenes are always assigned ``None``.
    label_versions : Mapping[str, str], optional
        Logical-label → version map written verbatim to ``current.json``.
        Defaults to ``{"masks": "v1", "masks_aux": "v1", "sam2_meta": "v1"}``.
    """
    pa, pq = _arrow()
    splits = dict(splits or {})
    label_versions = dict(
        label_versions
        or {
            MASKS: "v1",
            MASKS_AUX: "v1",
            "sam2_meta": "v1",
            CAM_POSE: "v1",
        }
    )
    out_dir = Path(out_dir)
    (out_dir / "manifest").mkdir(parents=True, exist_ok=True)

    shard_index = _index_shards_by_phase(pack)

    # Build the table column-by-column for memory efficiency.
    cols: Dict[str, list] = {
        "scene_id": [],
        "scene_type": [],
        "phase": [],
        "frame_idx": [],
        "frame_filename": [],
        "split": [],
    }
    for has_col, shard_col in _COLUMNS.values():
        cols[has_col] = []
        cols[shard_col] = []

    for scene in scan.scenes:
        for phase in scene.phases:
            shards_here = shard_index.get((scene.scene_id, phase.phase_index), {})
            filenames = _frame_filenames_for(
                scan,
                scene.scene_id,
                phase.phase_index,
            )
            split_label = (
                splits.get(scene.scene_id) if scene.scene_type is SceneType.MULTI_OBJECT else None
            )
            for idx, fname in enumerate(filenames):
                cols["scene_id"].append(scene.scene_id)
                cols["scene_type"].append(scene.scene_type.value)
                cols["phase"].append(phase.phase_index)
                cols["frame_idx"].append(idx)
                cols["frame_filename"].append(fname)
                cols["split"].append(split_label)
                for modality, (has_col, shard_col) in _COLUMNS.items():
                    shard = shards_here.get(modality)
                    cols[has_col].append(shard is not None)
                    cols[shard_col].append(shard.repo_path if shard else None)

    schema = pa.schema(
        [
            ("scene_id", pa.string()),
            ("scene_type", pa.string()),
            ("phase", pa.int32()),
            ("frame_idx", pa.int32()),
            ("frame_filename", pa.string()),
            ("split", pa.string()),
            *[(c, pa.bool_()) for (c, _) in _COLUMNS.values()],
            *[(c, pa.string()) for (_, c) in _COLUMNS.values()],
        ]
    )
    table = pa.table({k: cols[k] for k in schema.names}, schema=schema)

    parquet_path = out_dir / "manifest" / f"frames_{SCHEMA_VERSION}.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    current_path = out_dir / "manifest" / "current.json"
    current_path.write_text(
        json.dumps({"label_versions": label_versions, "schema_version": SCHEMA_VERSION}, indent=2),
        encoding="utf-8",
    )
    log.info("wrote manifest: %d rows → %s", len(table), parquet_path)
    return ManifestPaths(parquet_path=parquet_path, current_json_path=current_path)


def read_frame_manifest(parquet_path: Path):
    """Read a manifest Parquet back as a pyarrow Table (test/debug aid)."""
    _, pq = _arrow()
    return pq.read_table(parquet_path)
