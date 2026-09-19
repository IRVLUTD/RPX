"""Arrange raw ego (GoPro) captures into the layout scan_capture_root expects.

Ego captures land on disk as ``<ego_root>/scene<NNN>/ego/{rgb,sam2}`` — one
continuous, non-phased video per scene (see the ego annotation pipeline,
``~/Desktop/EGO_ANNOTATION_PIPELINE.md``). That does not match dataset_hub's
``<DATA>/{mos,sos}/<scene_id>/<phase>/<modality>`` shape: no top-level
``ego/`` scene family recognised before this module, and no numeric phase
subdirectory (ego has one continuous clip, not three).

This module bridges the two, non-destructively (symlinks by default — nothing
here ever writes into the source ego captures):

    <DATA>/ego/<scene_id>/0/{rgb,sam2/...}

``<scene_id>`` is ego's own bare, zero-padded name (``scene020``) —
VERIFIED against the live IRVLUTD/RPX repo, which publishes mos/ scenes as
bare ``scene001``..``scene100`` (a local working copy may use a
location-suffixed directory name like ``scene20.su.checkerboard.``; the
frame/mask bytes underneath are identical to the live repo, only that
local directory name differs). Using the bare form means
``manifest.py``'s split/difficulty lookup (an exact-string
``splits.get(scene.scene_id)``) works against the live splits file with
zero extra code — same physical scene, same scene number, captured on
independent cameras (rig D435 vs. head-worn GoPro).

A few loose files/dirs are intentionally left OUT of the arranged tree —
internal to the ego labeling pipeline, not something a downstream benchmark
user needs:

    ref_frame.txt              which frame the DINOv3 correspondence used
                                as its reference (pipeline provenance only)
    mask_to_object_multi.json  multi-anchor labeling metadata (only exists
                                on a handful of experimentally-relabeled
                                scenes; not part of the shipped label set)
    jpg/                       secondary JPEG copies of the rgb/ frames

``masks_verified/`` has no MOS precedent either — MOS only ships the
``verified_masks.txt`` *file* (which both ego and MOS already carry via the
existing ``sam2/_meta`` bundle), never a directory of rendered frames.
Excluded by default; pass ``include_masks_verified=True`` if that decision
changes.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..exceptions import ConfigError

#: Loose files directly under sam2/ that are pipeline-internal, not shipped.
EXCLUDED_META_FILES = frozenset({"ref_frame.txt", "mask_to_object_multi.json"})

#: Directories directly under sam2/ that are pipeline-internal refinement
#: provenance (iteration-by-iteration snapshots from manual mask review),
#: not final shipped data -- same category as EXCLUDED_META_FILES. Only
#: present on scenes that went through >=1 refinement pass; absent on
#: clean single-pass scenes. Not recognised by any packer.py modality
#: mapping either way, so excluding it is a pure size/time win with no
#: behavior change for what actually gets packed.
EXCLUDED_META_DIRS = frozenset({"mask_refinement"})

#: Top-level dirs under <scene>/ego/ that are never part of the arranged tree
#: (only rgb/ and sam2/ are recognised modalities).
_ARRANGED_TOP_LEVEL = ("rgb", "sam2")


@dataclass(frozen=True)
class EgoArrangeResult:
    """One scene's outcome from :func:`prepare_ego_layout`."""

    ego_scene_dir: str  # e.g. "scene020"
    scene_id: Optional[str] = None  # resolved mos/ sibling name, if found
    dst: Optional[Path] = None
    status: str = "ok"  # "ok" | "no_mos_sibling" | "no_rgb" | "no_sam2"
    message: str = ""


@dataclass(frozen=True)
class EgoLayoutReport:
    results: List[EgoArrangeResult] = field(default_factory=list)

    @property
    def ok(self) -> List[EgoArrangeResult]:
        return [r for r in self.results if r.status == "ok"]

    @property
    def skipped(self) -> List[EgoArrangeResult]:
        return [r for r in self.results if r.status != "ok"]


_SCENE_NUM_RE = re.compile(r"^scene0*(\d+)$")
_MOS_SCENE_NUM_RE = re.compile(r"^scene0*(\d+)(\.|$)")


def _ego_scene_number(ego_scene_dir_name: str) -> Optional[int]:
    m = _SCENE_NUM_RE.match(ego_scene_dir_name)
    return int(m.group(1)) if m else None


def resolve_mos_scene_id(mos_root: Path, ego_scene_dir_name: str) -> Optional[str]:
    """Validate an ego scene dir has a ``mos/`` sibling, by scene NUMBER, and
    return the CANONICAL scene_id to use for both.

    IMPORTANT: the canonical scene_id is ego's own bare, zero-padded form
    (``scene020``) — verified against the live IRVLUTD/RPX repo, where
    scenes are published as bare ``scene001``..``scene100`` (NOT the
    location-suffixed form some local working copies use, e.g.
    ``scene20.su.checkerboard`` — that's a local/staging naming quirk that
    predates upload; the actual frame/mask bytes underneath match the live
    repo exactly, only the on-disk directory name differs). Do not use the
    matched mos/ directory's own (possibly suffixed) name as the scene_id —
    that was the bug in an earlier version of this function.

    ``mos_root`` is used purely as a validation gate — confirms a scene
    with this number actually exists as a mos/ scene, regardless of what
    its local directory happens to be named. Returns ``None`` if there's no
    exactly-one match (missing, or ambiguous — caller should treat either
    as "skip, don't guess").
    """
    n = _ego_scene_number(ego_scene_dir_name)
    if n is None or not mos_root.is_dir():
        return None
    matches = [
        d.name for d in mos_root.iterdir()
        if d.is_dir() and _MOS_SCENE_NUM_RE.match(d.name)
        and int(_MOS_SCENE_NUM_RE.match(d.name).group(1)) == n
    ]
    if len(matches) != 1:
        return None
    return ego_scene_dir_name  # canonical bare form, e.g. "scene020"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        if src.is_dir():
            # Mirror with a REAL directory containing per-file symlinks,
            # not one directory-level symlink. A directory symlink is
            # invisible to any standard recursive tree walk that doesn't
            # explicitly opt in to following symlinks -- notably
            # ``Path.rglob`` (Python's default: does not descend into
            # symlinked directories), which lossless_convert.py's
            # _plan_tree relies on. A single dir-symlink here would make
            # every rgb/mask frame invisible to lossless-convert while
            # still resolving fine for direct-path reads (scan/pack) --
            # confirmed via a real lossless-convert dry-run before this
            # fix: 0 of ~1500 real ego frames were discovered, only the
            # two individually-symlinked meta files per scene were. Still
            # 100% symlinks (nothing written into the source), just at
            # file granularity so any tree walk works uniformly.
            dst.mkdir(parents=True, exist_ok=True)
            for entry in sorted(src.rglob("*")):
                if not entry.is_file():
                    continue
                target = dst / entry.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(entry.resolve())
        else:
            dst.symlink_to(src.resolve())
    elif mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ConfigError(f"unknown mode {mode!r}", hint="Use 'symlink' or 'copy'.")


def arrange_ego_scene(
    ego_scene_src: Path,
    dst_ego_root: Path,
    scene_id: str,
    include_masks_verified: bool = False,
    mode: str = "symlink",
) -> Path:
    """Arrange one ``<ego_root>/scene<NNN>/ego/`` capture into
    ``<dst_ego_root>/<scene_id>/0/``. Returns the phase-0 dst dir.

    Idempotent: existing links/files at the destination are left alone (so
    re-running after adding a new scene doesn't touch already-arranged
    ones). Remove the destination scene dir first if you want a clean
    re-arrange.
    """
    rgb_src = ego_scene_src / "rgb"
    sam2_src = ego_scene_src / "sam2"
    if not rgb_src.is_dir():
        raise ConfigError(f"no rgb/ under {ego_scene_src}")
    if not sam2_src.is_dir():
        raise ConfigError(f"no sam2/ under {ego_scene_src}")

    dst_phase = dst_ego_root / scene_id / "0"

    _link_or_copy(rgb_src, dst_phase / "rgb", mode)

    # Allow-list, not exclude-list: ship exactly rgb/ + sam2/masks/ +
    # mask_to_object.json (+ verified_masks.txt, tiny provenance) --
    # nothing else. masks_aux (bbox_overlay/dino_output/palette/
    # contour_gt_masks/rgb_and_mask/masks_contour_with_hidden) is QC/
    # visualization data, not read by any ego task recipe, and dropped
    # entirely per the "just rgb + masks (+ the id->name mapping)"
    # decision -- ego's per-frame composite images are large (high-res
    # GoPro source), so this is also the single biggest size lever.
    dst_sam2 = dst_phase / "sam2"
    masks_src = sam2_src / "masks"
    if not masks_src.is_dir():
        raise ConfigError(f"no sam2/masks/ under {ego_scene_src}")
    _link_or_copy(masks_src, dst_sam2 / "masks", mode)
    if include_masks_verified and (sam2_src / "masks_verified").is_dir():
        _link_or_copy(sam2_src / "masks_verified", dst_sam2 / "masks_verified", mode)
    for fname in ("mask_to_object.json", "verified_masks.txt"):
        src_file = sam2_src / fname
        if src_file.is_file():
            _link_or_copy(src_file, dst_sam2 / fname, mode)

    return dst_phase


def prepare_ego_layout(
    ego_captures_root: Path,
    mos_root: Path,
    dst_root: Path,
    scenes: Optional[List[str]] = None,
    include_masks_verified: bool = False,
    mode: str = "symlink",
) -> EgoLayoutReport:
    """Arrange every (or a selected subset of) ego scene into
    ``<dst_root>/ego/<scene_id>/0/`` — ready for ``scan_capture_root``.

    ``scenes``: optional list of ego scene dir names (``["scene020",
    "scene032"]``) to restrict to. Default: every ``scene*`` dir under
    ``ego_captures_root``.
    """
    ego_captures_root = Path(ego_captures_root)
    mos_root = Path(mos_root)
    dst_ego_root = Path(dst_root) / "ego"

    if scenes is None:
        candidates = sorted(
            d.name for d in ego_captures_root.iterdir()
            if d.is_dir() and _ego_scene_number(d.name) is not None
        )
    else:
        candidates = list(scenes)

    results: List[EgoArrangeResult] = []
    for name in candidates:
        ego_scene_dir = ego_captures_root / name / "ego"
        if not (ego_scene_dir / "rgb").is_dir():
            results.append(EgoArrangeResult(name, status="no_rgb",
                                             message=f"no rgb/ under {ego_scene_dir}"))
            continue
        if not (ego_scene_dir / "sam2" / "masks").is_dir():
            results.append(EgoArrangeResult(name, status="no_sam2",
                                             message=f"no sam2/masks/ under {ego_scene_dir}"))
            continue

        scene_id = resolve_mos_scene_id(mos_root, name)
        if scene_id is None:
            results.append(EgoArrangeResult(
                name, status="no_mos_sibling",
                message=f"no unique mos/ scene matching {name}'s number",
            ))
            continue

        dst = arrange_ego_scene(ego_scene_dir, dst_ego_root, scene_id,
                                include_masks_verified=include_masks_verified, mode=mode)
        results.append(EgoArrangeResult(name, scene_id=scene_id, dst=dst))

    return EgoLayoutReport(results=results)
