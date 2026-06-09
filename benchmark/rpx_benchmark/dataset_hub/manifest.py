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
from typing import Any, Dict, List, Mapping, Optional

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


# Logical-modality → on-disk sub-path relative to a phase root. These map
# from the split-manifest's modality vocabulary onto where the actual
# files live in the scanned tree. Used by :func:`_detect_modality_extensions`
# to sniff per-modality file extensions for ``current.json``.
_MODALITY_SUBPATH_FOR_DETECTION: Dict[str, str] = {
    "rgb": "rgb",
    "depth": "depth",
    "fisheye": "fisheye",
    "fisheye_left": "fisheye/left",
    "fisheye_right": "fisheye/right",
    "masks": "sam2/masks",
    "cam_pose": "cam_pose",
}


def _detect_modality_extensions(scan: ScanResult) -> Dict[str, str]:
    """Sniff one file per modality directory in the scanned tree and
    return ``{modality_name: ".png" | ".webp" | ".npz" | ".npy"}``.

    Records the dataset's *on-disk* extension per modality so the
    split-manifest writer (and any other downstream consumer) can build
    paths that resolve after extraction without hardcoded assumptions.

    Backward-compatible: if a modality's directory is missing or empty
    in the scanned tree, that modality is omitted from the returned
    dict, and downstream code falls back to its built-in defaults.
    """
    found: Dict[str, str] = {}
    for scene in scan.scenes:
        if not scene.phases:
            continue
        sub = "mos" if scene.scene_type is SceneType.MULTI_OBJECT else "sos"
        phase_root = scan.root / sub / scene.scene_id / str(scene.phases[0].phase_index)
        for modality, subpath in _MODALITY_SUBPATH_FOR_DETECTION.items():
            if modality in found:
                continue
            mod_dir = phase_root / subpath
            if not mod_dir.is_dir():
                continue
            # Pick the first regular file under this modality's subdir;
            # rglob handles the case where files live one level deeper
            # (e.g. ``fisheye/`` has ``left/`` and ``right/`` subdirs but
            # no files directly inside).
            for entry in sorted(mod_dir.rglob("*")):
                if entry.is_file():
                    found[modality] = entry.suffix
                    break
        if len(found) == len(_MODALITY_SUBPATH_FOR_DETECTION):
            break
    return found


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
    modality_extensions = _detect_modality_extensions(scan)

    # Preserve any pre-existing current.json keys we do not own. The live
    # IRVLUTD/RPX repo's current.json carries additional top-level blocks
    # (``manifests``, ``sos``, ``mos``, ``metadata_versions``, …) that
    # the teammate's separate manifest tooling writes. A naive overwrite
    # would strip them and break downstream consumers; instead we read
    # the existing payload (if any), update only the keys we are
    # authoritative for, and write the merged result back.
    #
    # The keys we own (and will overwrite on every run):
    #
    #   * ``label_versions``       — from the ``--label-version`` flag
    #   * ``schema_version``       — fixed at this module's SCHEMA_VERSION
    #   * ``modality_extensions``  — sniffed from the scanned source tree
    #
    # Everything else is left untouched.
    OWNED_KEYS = {"label_versions", "schema_version", "modality_extensions"}
    if current_path.is_file():
        try:
            current_payload: Dict[str, Any] = json.loads(
                current_path.read_text(encoding="utf-8"),
            )
            if not isinstance(current_payload, dict):
                current_payload = {}
        except json.JSONDecodeError:
            current_payload = {}
    else:
        current_payload = {}
    current_payload["label_versions"] = label_versions
    current_payload["schema_version"] = SCHEMA_VERSION
    if modality_extensions:
        current_payload["modality_extensions"] = modality_extensions
    # Don't carry stale modality_extensions if it has nothing to say
    # AND no prior writer set it.
    elif "modality_extensions" in current_payload and not current_payload["modality_extensions"]:
        current_payload.pop("modality_extensions")
    preserved = sorted(k for k in current_payload if k not in OWNED_KEYS)
    if preserved:
        log.info(
            "current.json: preserved %d unmanaged top-level key(s): %s",
            len(preserved),
            ", ".join(preserved),
        )
    current_path.write_text(
        json.dumps(current_payload, indent=2),
        encoding="utf-8",
    )

    # Persist per-tar SHA-256 to manifest/checksums.json. The packer
    # computes these as part of pack_capture_tree; the downstream
    # download-side verification in hub._extract_snapshot_tars reads
    # this file and aborts on any tar whose bytes don't match. Tar
    # entries whose sha256 is None (rare — only when re-running
    # manifest without a fresh pack) are still written with a null
    # value so the absence is explicit, not silent.
    checksums_path = out_dir / "manifest" / "checksums.json"
    checksums = {s.repo_path: s.sha256 for s in pack.shards}
    checksums_path.write_text(
        json.dumps(
            {"sha256": checksums, "algorithm": "sha256"},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    log.info("wrote manifest: %d rows → %s", len(table), parquet_path)
    n_with_sha = sum(1 for v in checksums.values() if v)
    log.info(
        "wrote checksums.json: %d/%d tar shards have SHA-256 → %s",
        n_with_sha,
        len(checksums),
        checksums_path,
    )

    # Per-file SHA-256: walk every packed tar, stream each member through
    # SHA-256, and persist the mapping at manifest/file_checksums.json.
    # This is the foundation for ``verify_dataset(snapshot_root)`` to
    # detect local-disk corruption between extract and load — the case
    # the tar-level SHA-256 cannot guard against.
    file_checksums = _compute_per_file_checksums(out_dir, pack)
    file_checksums_path = out_dir / "manifest" / "file_checksums.json"
    file_checksums_path.write_text(
        json.dumps(
            {"algorithm": "sha256", "sha256": file_checksums},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log.info(
        "wrote file_checksums.json: %d extracted-file SHA-256 entries → %s",
        len(file_checksums),
        file_checksums_path,
    )

    if modality_extensions:
        log.info(
            "current.json modality_extensions: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(modality_extensions.items())),
        )
    return ManifestPaths(parquet_path=parquet_path, current_json_path=current_path)


def _compute_per_file_checksums(
    out_dir: Path,
    pack: PackResult,
) -> Dict[str, str]:
    """For every member of every packed tar, compute SHA-256 and key it
    by the *extracted* repo path the user will see after
    ``_extract_snapshot_tars`` runs.

    Returns ``{ "extracted/scenes/<scene>/<phase>/<member>": sha256_hex }``.

    The per-tar SHA-256 (in ``checksums.json``) guards transport
    corruption from operator → HF → user. This per-file map guards
    everything *after* extraction: bit rot on the user's disk, OS-level
    file corruption, accidental overwrites.
    """
    import hashlib
    import tarfile

    out: Dict[str, str] = {}
    for shard in pack.shards:
        tar_path = out_dir / shard.repo_path
        if not tar_path.is_file():
            continue
        # The extracted layout mirrors the tar's nesting: a tar at
        # ``scenes/scene_x/0/rgb.tar`` whose members are ``rgb/<frame>.webp``
        # extracts to ``extracted/scenes/scene_x/0/rgb/<frame>.webp``.
        # Strip the ``.tar`` filename to get the extracted prefix.
        rel_parts = shard.repo_path.split("/")
        if not rel_parts or not rel_parts[-1].endswith(".tar"):
            continue
        prefix_parts = rel_parts[:-1]
        # labels/<modality>/<version>.tar extracts straight to
        # extracted/.../<modality>'s file layout — drop the ``labels/`` and
        # ``<version>`` segments so the result lives under the modality.
        if "labels" in prefix_parts:
            li = prefix_parts.index("labels")
            prefix_parts = prefix_parts[:li]
        extracted_prefix = "extracted/" + "/".join(prefix_parts) if prefix_parts else "extracted"
        try:
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    h = hashlib.sha256()
                    for chunk in iter(lambda f=f: f.read(1 << 20), b""):
                        h.update(chunk)
                    extracted_path = f"{extracted_prefix}/{member.name}"
                    out[extracted_path] = h.hexdigest()
        except tarfile.TarError:
            continue
    return out


def read_frame_manifest(parquet_path: Path):
    """Read a manifest Parquet back as a pyarrow Table (test/debug aid)."""
    _, pq = _arrow()
    return pq.read_table(parquet_path)
