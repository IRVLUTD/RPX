"""Walk an on-disk RPX capture tree and report structured inventory.

Source layout (as captured by the team)::

    <root>/
    ├── mos/                                  # multi-object scenes
    │   └── scene<N>.<building>.<area>/<phase>/<modality>/...
    ├── sos/                                  # single-object scenes
    │   └── object<N>.<name>/0/<modality>/...
    └── ego/                                  # egocentric scenes (one phase, "0")
        └── scene<N>.<building>.<area>/0/<modality>/...
            # scene_id MUST match its sibling mos/ scene exactly — that's
            # how manifest.py joins an ego scene to its MOS split/difficulty
            # tier. Use `prepare-ego` (see ego_layout.py) to arrange a raw
            # `<scene>/ego/` capture into this shape before scanning.

Scene type is determined by which top-level subdirectory the scene lives
under (``mos/``, ``sos/``, or ``ego/``) — not by directory name pattern.
This makes naming flexible and reduces classification errors.

The scanner walks every modality subdir, counts files, sums bytes, and
returns a structured ``ScanResult`` that downstream tooling (packer,
manifest builder, CLI ``--dry-run``) can format however it likes. The
scanner is *purely informational* — it never mutates the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .recipes import SceneType

# Top-level subdirectory → scene type. Anything else under root (READMEs,
# stray captures, half-finished work) is reported via ``ScanResult.skipped``.
_TYPE_DIRS: Dict[str, SceneType] = {
    "mos": SceneType.MULTI_OBJECT,
    "sos": SceneType.SINGLE_OBJECT,
    "ego": SceneType.EGO,
}

#: The inverse of ``_TYPE_DIRS`` — the single source of truth for "where does
#: a scene of this type live on disk". Every other module (packer, manifest,
#: ...) that needs this mapping should import it from here rather than
#: keeping its own copy — a second, un-synced copy is exactly how the
#: pre-ego manifest.py bug happened (its own `"mos" if ... else "sos"`
#: ternary never learned about a third scene type).
SRC_SUBDIR_BY_TYPE: Dict[SceneType, str] = {v: k for k, v in _TYPE_DIRS.items()}


@dataclass(frozen=True)
class ModalityInventory:
    """Inventory of a single modality directory under one phase."""

    name: str  # e.g. "rgb", "depth", "fisheye", "cam_pose", "sam2/masks"
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class PhaseInventory:
    """Inventory of one phase directory (set of modalities)."""

    phase_index: int
    modalities: Dict[str, ModalityInventory]

    @property
    def total_bytes(self) -> int:
        return sum(m.total_bytes for m in self.modalities.values())

    @property
    def file_count(self) -> int:
        return sum(m.file_count for m in self.modalities.values())


@dataclass(frozen=True)
class SceneInventory:
    """Inventory of one scene directory (set of phases)."""

    scene_id: str
    scene_type: SceneType
    phases: List[PhaseInventory]

    @property
    def total_bytes(self) -> int:
        return sum(p.total_bytes for p in self.phases)

    @property
    def file_count(self) -> int:
        return sum(p.file_count for p in self.phases)


@dataclass(frozen=True)
class ScanResult:
    """Top-level scan result over a capture root."""

    root: Path
    scenes: List[SceneInventory]
    skipped: List[Path] = field(default_factory=list)

    # ── aggregate helpers ────────────────────────────────────────────
    def by_type(self, scene_type: SceneType) -> List[SceneInventory]:
        return [s for s in self.scenes if s.scene_type is scene_type]

    @property
    def total_bytes(self) -> int:
        return sum(s.total_bytes for s in self.scenes)

    @property
    def file_count(self) -> int:
        return sum(s.file_count for s in self.scenes)

    def modality_totals(self) -> Dict[str, ModalityInventory]:
        """Aggregate per-modality file counts and bytes across all phases."""
        agg: Dict[str, list[int]] = {}
        for scene in self.scenes:
            for phase in scene.phases:
                for name, inv in phase.modalities.items():
                    if name not in agg:
                        agg[name] = [0, 0]
                    agg[name][0] += inv.file_count
                    agg[name][1] += inv.total_bytes
        return {
            name: ModalityInventory(name=name, file_count=fc, total_bytes=tb)
            for name, (fc, tb) in sorted(agg.items())
        }


# --------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------- #


def _measure_dir(d: Path) -> tuple[int, int]:
    """``(file_count, total_bytes)`` for everything under ``d`` (recursive)."""
    files, total = 0, 0
    for p in d.rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return files, total


def _scan_phase(phase_dir: Path) -> PhaseInventory:
    """Walk a phase dir and inventory every immediate modality subdir.

    ``sam2/`` is special: each of its subdirs (``masks/``, ``bbox_overlay/``,
    ...) is reported separately under ``sam2/<sub>`` so the packer can
    decide which subset to ship per modality. Loose files at the ``sam2/``
    root (``mask_to_object.json``, ``verified_masks.txt``,
    ``iter*_faulty.txt``) are bundled into a synthetic ``sam2/_meta``
    pseudo-modality to keep the inventory complete.
    """
    try:
        idx = int(phase_dir.name)
    except ValueError:
        idx = -1

    modalities: Dict[str, ModalityInventory] = {}
    for child in sorted(phase_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "sam2":
            for sub in sorted(child.iterdir()):
                if sub.is_dir():
                    fc, tb = _measure_dir(sub)
                    modalities[f"sam2/{sub.name}"] = ModalityInventory(
                        name=f"sam2/{sub.name}",
                        file_count=fc,
                        total_bytes=tb,
                    )
            meta_files = [p for p in child.iterdir() if p.is_file()]
            if meta_files:
                fc = len(meta_files)
                tb = sum(p.stat().st_size for p in meta_files)
                modalities["sam2/_meta"] = ModalityInventory(
                    name="sam2/_meta",
                    file_count=fc,
                    total_bytes=tb,
                )
            continue
        fc, tb = _measure_dir(child)
        modalities[child.name] = ModalityInventory(
            name=child.name,
            file_count=fc,
            total_bytes=tb,
        )

    return PhaseInventory(phase_index=idx, modalities=modalities)


def scan_capture_root(root: Path) -> ScanResult:
    """Scan a capture-tree root and return a :class:`ScanResult`.

    Walks ``root/mos/*`` (multi-object scenes) and ``root/sos/*`` (single-
    object scenes). Anything else under root, and anything inside ``mos/``
    or ``sos/`` that isn't a scene directory, is collected into
    :attr:`ScanResult.skipped` so the caller can flag typos / strays.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"capture root does not exist: {root}")

    scenes: List[SceneInventory] = []
    skipped: List[Path] = []
    for entry in sorted(root.iterdir()):
        # Top-level: only ``mos/`` and ``sos/`` are recognised.
        if not entry.is_dir() or entry.name not in _TYPE_DIRS:
            skipped.append(entry)
            continue

        scene_type = _TYPE_DIRS[entry.name]
        for scene_dir in sorted(entry.iterdir()):
            if not scene_dir.is_dir():
                skipped.append(scene_dir)
                continue
            phase_dirs = sorted(
                [p for p in scene_dir.iterdir() if p.is_dir()],
                key=lambda p: p.name,
            )
            phases = [_scan_phase(p) for p in phase_dirs]
            scenes.append(
                SceneInventory(
                    scene_id=scene_dir.name,
                    scene_type=scene_type,
                    phases=phases,
                )
            )

    return ScanResult(root=root, scenes=scenes, skipped=skipped)
