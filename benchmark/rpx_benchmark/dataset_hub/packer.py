"""Pack an on-disk capture tree into per-modality tar shards for HF upload.

We deliberately tar each ``(scene, phase, modality)`` triple instead of
shipping loose files. Three reasons:

1. **Upload speed.** The full RPX dataset is ~890 GB across an estimated
   750k loose frames. HF (and git LFS underneath) handles thousands of
   ~400 MB tars far faster than hundreds of thousands of loose files.

2. **Selective download granularity.** ``snapshot_download`` filters at the
   *file* level. With per-modality tars, a single ``allow_pattern`` like
   ``scenes/*/*/rgb.tar`` pulls exactly the bytes a task needs and
   nothing more — the cache deduplicates across tasks automatically.

3. **Atomic re-issue of label updates.** Bumping a label version is a
   one-line edit (re-run the packer for that modality with
   ``--label-version v2``) plus an upload of the new tars; older versions
   stay in the repo for reproducibility.

Output layout (under the staging directory)::

    staging/
    ├── scenes/                                # multi-object captures
    │   └── <scene_id>/<phase>/
    │       ├── rgb.tar
    │       ├── depth.tar
    │       ├── fisheye.tar
    │       ├── cam_pose.tar              # treated as raw (sensor pipeline output)
    │       └── labels/
    │           ├── masks/v1.tar
    │           ├── masks_aux/v1.tar
    │           └── sam2_meta/v1.tar
    └── objects/                               # single-object captures (one phase)
        └── <object_id>/0/
            (same modality structure)

The packer is *content-stable*: tar member order is sorted, mtime is
zeroed, ownership is normalised — so re-running on the same source
produces byte-identical tars (and therefore identical SHA-256 hashes).
This matters for the manifest's integrity field and for HF's
content-addressed cache to deduplicate across re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..exceptions import ConfigError, DatasetError
from ..logging_utils import get_logger
from .recipes import (
    CAM_POSE,
    DEPTH,
    FISHEYE,
    MASKS,
    MASKS_AUX,
    QUESTIONNAIRE,
    RGB,
    SceneType,
)
from .scanner import ScanResult


log = get_logger(__name__)


# Modalities considered "raw" (immutable, no version embedded in path).
RAW_MODALITIES = frozenset({RGB, DEPTH, FISHEYE, CAM_POSE})

# Mapping from scanner modality keys → packer output modality + label flag.
# Scanner keys come straight from the on-disk dir names, with sam2/ split
# into sam2/<sub>. The packer collapses these into a small, named set.
_MODALITY_PLAN: Dict[str, tuple[str, bool]] = {
    # scanner key                 (output modality name, is_label)
    "rgb":                         (RGB,           False),
    "depth":                       (DEPTH,         False),
    "fisheye":                     (FISHEYE,       False),
    "cam_pose":                    (CAM_POSE,      False),
    "sam2/masks":                  (MASKS,         True),
    "sam2/_meta":                  ("sam2_meta",   True),
    # The remaining sam2/* aux directories all collapse into masks_aux.
    "sam2/bbox_overlay":           (MASKS_AUX,     True),
    "sam2/contour_gt_masks":       (MASKS_AUX,     True),
    "sam2/dino_output":            (MASKS_AUX,     True),
    "sam2/masks_contour_with_hidden": (MASKS_AUX,  True),
    "sam2/palette":                (MASKS_AUX,     True),
    "sam2/rgb_and_mask":           (MASKS_AUX,     True),
}


@dataclass(frozen=True)
class PackPlan:
    """Knobs for the packer."""

    src_root:      Path
    staging_root:  Path
    label_version: str  = "v1"
    overwrite:     bool = False
    compute_hash:  bool = True


@dataclass(frozen=True)
class PackedShard:
    """One tar shard the packer produced."""

    repo_path:    str           # path relative to staging_root, forward-slashes
    scene_id:     str
    phase:        int
    modality:     str
    is_label:     bool
    file_count:   int
    total_bytes:  int
    sha256:       Optional[str] = None


@dataclass(frozen=True)
class PackResult:
    """Aggregate result of one packer run."""

    shards:        List[PackedShard]
    skipped_keys:  List[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(s.total_bytes for s in self.shards)

    @property
    def total_files(self) -> int:
        return sum(s.file_count for s in self.shards)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _scene_root_for(scene_type: SceneType) -> str:
    return "scenes" if scene_type is SceneType.MULTI_OBJECT else "objects"


def _shard_repo_path(
    scene_type: SceneType, scene_id: str, phase: int,
    modality: str, is_label: bool, label_version: str,
) -> str:
    base = f"{_scene_root_for(scene_type)}/{scene_id}/{phase}"
    if is_label:
        return f"{base}/labels/{modality}/{label_version}.tar"
    return f"{base}/{modality}.tar"


def _scene_src_root(src_root: Path, scene_type: SceneType,
                     scene_id: str) -> Path:
    """Where a scene actually lives on disk: ``<src>/{mos,sos}/<scene_id>``."""
    sub = "mos" if scene_type is SceneType.MULTI_OBJECT else "sos"
    return src_root / sub / scene_id


def _src_dir_for(src_root: Path, scene_type: SceneType, scene_id: str,
                  phase: int, scanner_key: str) -> Path:
    """Translate a scanner modality key back to its on-disk source dir."""
    return _scene_src_root(src_root, scene_type, scene_id) / str(phase) / scanner_key


def _enumerate_files_for_modality(
    src_root: Path, scene_type: SceneType, scene_id: str,
    phase: int, scanner_key: str,
) -> List[Path]:
    """All files under one modality dir, sorted, recursive.

    For ``sam2/_meta`` the source is the ``sam2/`` dir itself, but only
    its *immediate* loose files (not subdirs) are taken — the subdirs
    are owned by the per-aux-dir scanner keys.
    """
    if scanner_key == "sam2/_meta":
        sam2_root = _scene_src_root(src_root, scene_type, scene_id) / str(phase) / "sam2"
        if not sam2_root.is_dir():
            return []
        return sorted(p for p in sam2_root.iterdir() if p.is_file())

    src_dir = _src_dir_for(src_root, scene_type, scene_id, phase, scanner_key)
    if not src_dir.is_dir():
        return []
    return sorted(p for p in src_dir.rglob("*") if p.is_file())


def _write_tar(
    out_path: Path, files: List[Path], rel_to: Path,
    compute_hash: bool,
) -> tuple[int, int, Optional[str]]:
    """Write ``files`` to a tar at ``out_path``. Returns
    ``(file_count, total_bytes, sha256)``. Tar entries are deterministic.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    h = hashlib.sha256() if compute_hash else None

    # Write to a temp path then rename — never expose a half-written tar.
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with tarfile.open(tmp, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for f in files:
            arcname = str(f.relative_to(rel_to)).replace(os.sep, "/")
            ti = tf.gettarinfo(name=str(f), arcname=arcname)
            # Normalise so re-runs produce byte-identical tars.
            ti.mtime = 0
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            with f.open("rb") as fh:
                tf.addfile(ti, fh)
            total += f.stat().st_size

    if h is not None:
        with tmp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)

    os.replace(tmp, out_path)
    return len(files), total, (h.hexdigest() if h else None)


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #

def pack_capture_tree(plan: PackPlan, scan: ScanResult) -> PackResult:
    """Pack each ``(scene, phase, modality)`` triple in ``scan`` into one tar.

    Reusing :class:`ScanResult` from the scanner avoids re-walking the
    source tree — the scan already has the file counts, byte totals, and
    structure we need. The packer just translates each (scene, phase,
    modality) entry into a tar at the right repo path.
    """
    if plan.src_root != scan.root:
        raise ConfigError(
            f"plan.src_root ({plan.src_root}) does not match the root the "
            f"scan was taken at ({scan.root}).",
            hint="Pass the same root to scan_capture_root() and PackPlan.",
        )
    plan.staging_root.mkdir(parents=True, exist_ok=True)

    shards: List[PackedShard] = []
    skipped: List[str] = []

    for scene in scan.scenes:
        for phase in scene.phases:
            # Group source modalities by output (modality, is_label) so
            # masks_aux/* all flow into the same masks_aux.tar.
            grouped: Dict[tuple[str, bool], list[str]] = {}
            for scanner_key in phase.modalities:
                target = _MODALITY_PLAN.get(scanner_key)
                if target is None:
                    skipped.append(
                        f"{scene.scene_id}/{phase.phase_index}/{scanner_key}"
                    )
                    continue
                grouped.setdefault(target, []).append(scanner_key)

            for (out_modality, is_label), keys in grouped.items():
                files: List[Path] = []
                for k in keys:
                    files.extend(_enumerate_files_for_modality(
                        plan.src_root, scene.scene_type,
                        scene.scene_id, phase.phase_index, k,
                    ))
                if not files:
                    continue

                repo_path = _shard_repo_path(
                    scene.scene_type, scene.scene_id, phase.phase_index,
                    out_modality, is_label, plan.label_version,
                )
                out_path = plan.staging_root / repo_path
                if out_path.exists() and not plan.overwrite:
                    raise DatasetError(
                        f"refusing to overwrite existing shard: {out_path}",
                        hint="Pass overwrite=True or remove the staging dir.",
                    )

                fc, bs, sha = _write_tar(
                    out_path, files,
                    rel_to=(_scene_src_root(plan.src_root, scene.scene_type,
                                              scene.scene_id)
                             / str(phase.phase_index)),
                    compute_hash=plan.compute_hash,
                )
                shards.append(PackedShard(
                    repo_path=repo_path,
                    scene_id=scene.scene_id,
                    phase=phase.phase_index,
                    modality=out_modality,
                    is_label=is_label,
                    file_count=fc,
                    total_bytes=bs,
                    sha256=sha,
                ))
                log.info("packed %s (%d files, %d bytes)",
                          repo_path, fc, bs)

    return PackResult(shards=shards, skipped_keys=skipped)


# --------------------------------------------------------------------- #
# Per-object shared artefacts (objects_meta/) — questionnaire dedup
# --------------------------------------------------------------------- #

# FewSOL questionnaire is "<n>. <question>" then a comma-separated answer
# block on the following non-blank line. This regex captures one Q-A pair.
_QUESTION_RE = re.compile(r"^\s*\d+\.\s*(?P<q>.+\?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SharedArtefact:
    """One file written under ``objects_meta/<object_id>/``."""

    object_id:   str
    repo_path:   str
    total_bytes: int


def _parse_questionnaire(text: str) -> Dict[str, list[str]]:
    """Parse the FewSOL-style questionnaire into a structured dict.

    The text shape is::

        # comments are ignored
        1. What is the name of the object in these images?
        tape and tape holder, tape, support and adhesive tape

        2. What is the category of the object in these images?
        ...

    Returns ``{question_text: [answer1, answer2, ...]}`` with answers
    split on commas. Questions are kept verbatim (warts and all) so the
    schema survives changes to the answer-template wording.
    """
    out: Dict[str, list[str]] = {}
    lines = [ln for ln in text.splitlines()
              if not ln.lstrip().startswith("#")]
    cleaned = "\n".join(lines)
    matches = list(_QUESTION_RE.finditer(cleaned))
    for i, m in enumerate(matches):
        q = m.group("q").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = cleaned[start:end].strip()
        # Take the first non-blank line as the answer block.
        first_line = next(
            (ln.strip() for ln in body.splitlines() if ln.strip()),
            "",
        )
        answers = [a.strip() for a in first_line.split(",") if a.strip()]
        out[q] = answers
    return out


def pack_objects_meta(plan: PackPlan, scan: ScanResult) -> List[SharedArtefact]:
    """Build the ``objects_meta/`` layer.

    For every SOS scene with a ``questionnaire.txt`` at its scene root,
    parse the file and write a normalised
    ``objects_meta/<object_id>/questionnaire.json`` under the staging
    dir. ``object_id`` is exactly the SOS scene directory name (the
    canonical key MOS scenes use in ``mask_to_object.json``).

    Also write ``objects_meta/_index.json``: a small lookup table
    listing every known object_id. The downloader uses this to verify
    that mask_to_object references resolve.
    """
    out: List[SharedArtefact] = []
    objects_meta = plan.staging_root / "objects_meta"
    objects_meta.mkdir(parents=True, exist_ok=True)

    object_ids: list[str] = []
    for scene in scan.scenes:
        if scene.scene_type is not SceneType.SINGLE_OBJECT:
            continue
        sos_root = (plan.src_root / "sos" / scene.scene_id)
        questionnaire_src = sos_root / "questionnaire.txt"
        if not questionnaire_src.is_file():
            log.warning("SOS scene %s has no questionnaire.txt; skipping",
                         scene.scene_id)
            continue

        parsed = _parse_questionnaire(
            questionnaire_src.read_text(encoding="utf-8"),
        )
        payload = {
            "object_id": scene.scene_id,
            "source": "sos/" + scene.scene_id + "/questionnaire.txt",
            "questions": parsed,
        }
        repo_path = f"objects_meta/{scene.scene_id}/questionnaire.json"
        out_path = plan.staging_root / repo_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not plan.overwrite:
            raise DatasetError(
                f"refusing to overwrite existing object meta: {out_path}",
                hint="Pass overwrite=True or remove the staging dir.",
            )
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        out_path.write_text(rendered, encoding="utf-8")
        out.append(SharedArtefact(
            object_id=scene.scene_id, repo_path=repo_path,
            total_bytes=out_path.stat().st_size,
        ))
        object_ids.append(scene.scene_id)
        log.info("wrote %s (%d questions)", repo_path, len(parsed))

    index_path = objects_meta / "_index.json"
    index_path.write_text(
        json.dumps({"object_ids": sorted(object_ids)}, indent=2,
                    sort_keys=True),
        encoding="utf-8",
    )
    return out
