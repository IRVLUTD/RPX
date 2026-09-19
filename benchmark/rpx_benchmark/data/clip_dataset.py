"""Per-(scene, phase) clip iteration for Video Depth and other
sequence-level tasks (e.g. D3 tracking, which is also clip-based).

The flat :class:`~rpx_benchmark.loader.RPXDataset` yields one independent
:class:`~rpx_benchmark.api.Sample` per frame — correct for per-frame
tasks (Image Depth, D2, D4/D5).  Video and tracking tasks instead need the
whole phase clip in one shot.  :class:`PhaseClipDataset` groups a depth
manifest's frames by ``(scene, phase)``, orders them by frame index, and
yields a :class:`PhaseClip` per group, decoding every modality through the
*existing* ``RPXDataset`` loaders — no duplicate IO code, one source of
truth for how RGB / depth / pose are read.

Design notes (``benchmark/README.md`` §3):

* Iteration order is ``outer = scene (sorted), inner = phase (0, 1, 2)``,
  which matches the cell-log grouping key ``(model, task, scene, phase)``.
* Clips are loaded lazily, one at a time, so host-RAM stays at ~1 clip
  (~0.6 GB) regardless of dataset size.
* Clips shorter than :data:`~rpx_benchmark.metrics.depth_temporal.MIN_CLIP_FRAMES`
  are **skipped** (logged) — temporal metrics are meaningless on a few
  frames, and we never pad (padding biases OPW/TAE downward).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..api import Difficulty, Phase
from ..loader import RPXDataset
from ..logging_utils import get_logger
from ..metrics.depth_alignment import DEPTH_MAX_M, DEPTH_MIN_M
from ..metrics.depth_temporal import MIN_CLIP_FRAMES

log = get_logger(__name__)

__all__ = ["PhaseClip", "PhaseClipDataset", "D1VClip", "D1VClipDataset"]

#: Canonical phase ordering for clip iteration and the temporal axis.
_PHASE_TO_IDX: Dict[Phase, int] = {Phase.CLUTTER: 0, Phase.INTERACTION: 1, Phase.CLEAN: 2}

#: Default D435 intrinsics (shared with the metric calculators).
_D435_K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

_TRAILING_INT = re.compile(r"(\d+)(?!.*\d)")


@dataclass
class PhaseClip:
    """One phase clip for Video Depth evaluation (decisions doc §3.2).

    Attributes
    ----------
    rgb_seq : (T, H, W, 3) uint8
    depth_gt_seq : (T, H, W) float32, metres (0 = invalid sentinel).
    valid_mask_seq : (T, H, W) bool — GT in (0.3, 5.0) m and finite.
    scene_id : str
    phase_idx : int — 0=clutter, 1=interaction, 2=clean.
    frame_indices : (T,) int — original frame numbers, ascending.
    intrinsics : (3, 3) float64 — pinhole K (default D435).
    poses : (T, 4, 4) float64 | None — world-from-camera SE(3); ``None``
        if any frame in the clip lacks a T265 pose.
    difficulty : Difficulty | None — the scene's ESD tier (easy/medium/
        hard), needed so the (scene, phase) cell can carry its split tag
        for cross-split (phase×difficulty) Φ. ``None`` if untagged.
    """

    rgb_seq: np.ndarray
    depth_gt_seq: np.ndarray
    valid_mask_seq: np.ndarray
    scene_id: str
    phase_idx: int
    frame_indices: np.ndarray
    intrinsics: np.ndarray
    poses: Optional[np.ndarray]
    difficulty: Optional[Difficulty] = None

    @property
    def num_frames(self) -> int:
        return int(self.rgb_seq.shape[0])

    @property
    def phase(self) -> Phase:
        return {0: Phase.CLUTTER, 1: Phase.INTERACTION, 2: Phase.CLEAN}[self.phase_idx]


def _phase_idx_of(entry: Dict[str, Any]) -> Optional[int]:
    """Resolve an entry's phase to a canonical 0/1/2 index, or ``None``."""
    v = entry.get("phase")
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v) if int(v) in (0, 1, 2) else None
    try:
        return _PHASE_TO_IDX[Phase(v)]
    except (ValueError, KeyError):
        return None


def _frame_key(entry: Dict[str, Any]) -> int:
    """Best-effort ascending frame index for ordering within a clip.

    Prefers an explicit ``frame`` field (top-level or in ``metadata``),
    then the trailing integer of the sample ``id``, then the RGB filename
    stem.  Falls back to 0 so ordering is at worst stable-by-insertion.
    """
    for src in (entry.get("frame"), (entry.get("metadata") or {}).get("frame")):
        if isinstance(src, (int, np.integer)):
            return int(src)
    for text in (str(entry.get("id", "")), Path(str(entry.get("rgb", ""))).stem):
        m = _TRAILING_INT.search(text)
        if m:
            return int(m.group(1))
    return 0


@dataclass
class PhaseClipDataset:
    """Groups a depth manifest into per-(scene, phase) clips.

    Construct via :meth:`from_manifest`.  Iterating yields
    :class:`PhaseClip` objects in ``(scene, phase)`` order, skipping clips
    with fewer than ``min_frames`` frames.
    """

    base: RPXDataset
    min_frames: int = MIN_CLIP_FRAMES
    intrinsics: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.intrinsics is None:
            self.intrinsics = _D435_K

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        min_frames: int = MIN_CLIP_FRAMES,
        intrinsics: Optional[np.ndarray] = None,
    ) -> "PhaseClipDataset":
        base = RPXDataset.from_manifest(manifest_path, batch_size=1)
        return cls(base=base, min_frames=min_frames,
                   intrinsics=intrinsics if intrinsics is not None else _D435_K)

    # -- grouping ---------------------------------------------------------- #

    def _groups(self) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
        """Return ``[(scene, phase_idx, [entries...]), ...]`` in iteration order.

        Entries are sorted by frame index within each group; groups are
        sorted by ``(scene, phase_idx)``.  Entries with no resolvable
        scene or phase are dropped (warned once per dropped count).
        """
        buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        dropped = 0
        for entry in self.base.samples:
            scene = entry.get("scene") or (entry.get("metadata") or {}).get("scene_id")
            phase_idx = _phase_idx_of(entry)
            if scene is None or phase_idx is None:
                dropped += 1
                continue
            buckets.setdefault((str(scene), phase_idx), []).append(entry)
        if dropped:
            log.warning("PhaseClipDataset: dropped %d entries with no scene/phase", dropped)

        groups = []
        for (scene, phase_idx), entries in sorted(buckets.items(), key=lambda kv: kv[0]):
            entries_sorted = sorted(entries, key=_frame_key)
            groups.append((scene, phase_idx, entries_sorted))
        return groups

    def clip_plan(self) -> List[Tuple[str, int, int]]:
        """``[(scene, phase_idx, n_frames), ...]`` without loading pixels.

        Cheap introspection for run planning / progress bars; reflects the
        ``min_frames`` skip.
        """
        return [
            (scene, phase_idx, len(entries))
            for scene, phase_idx, entries in self._groups()
            if len(entries) >= self.min_frames
        ]

    # -- loading ----------------------------------------------------------- #

    def _load_clip(self, scene: str, phase_idx: int, entries: List[Dict[str, Any]]) -> PhaseClip:
        rgbs, depths, valids, frame_ids, poses = [], [], [], [], []
        pose_complete = True
        difficulty: Optional[Difficulty] = None
        for entry in entries:
            sample = self.base._load_sample(entry)
            depth = np.asarray(sample.ground_truth.depth_map, dtype=np.float32)
            rgbs.append(np.asarray(sample.rgb, dtype=np.uint8))
            depths.append(depth)
            valids.append(np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M))
            frame_ids.append(_frame_key(entry))
            if sample.camera_pose is None:
                pose_complete = False
            else:
                poses.append(np.asarray(sample.camera_pose, dtype=np.float64))
            # Difficulty is a per-scene ESD tier — constant within a
            # (scene, phase) clip; take the first non-null and warn on drift.
            if sample.difficulty is not None:
                if difficulty is None:
                    difficulty = sample.difficulty
                elif difficulty != sample.difficulty:
                    log.warning(
                        "PhaseClipDataset: %s phase %d has mixed difficulty (%s vs %s); keeping %s",
                        scene, phase_idx, difficulty, sample.difficulty, difficulty,
                    )

        return PhaseClip(
            rgb_seq=np.stack(rgbs),
            depth_gt_seq=np.stack(depths),
            valid_mask_seq=np.stack(valids),
            scene_id=scene,
            phase_idx=phase_idx,
            frame_indices=np.asarray(frame_ids, dtype=np.int64),
            intrinsics=self.intrinsics,
            poses=np.stack(poses) if (pose_complete and poses) else None,
            difficulty=difficulty,
        )

    def __iter__(self) -> Iterator[PhaseClip]:
        for scene, phase_idx, entries in self._groups():
            if len(entries) < self.min_frames:
                log.warning(
                    "PhaseClipDataset: skipping %s phase %d — only %d frames (< %d)",
                    scene, phase_idx, len(entries), self.min_frames,
                )
                continue
            yield self._load_clip(scene, phase_idx, entries)

    def __len__(self) -> int:
        """Number of clips that will be yielded (after the min_frames skip)."""
        return len(self.clip_plan())


# --------------------------------------------------------------------------- #
# Paper-facing aliases — RPX depth task labels onto the generic clip API.
# The generic names above are canonical; these are the paper's Video Depth labels.
# --------------------------------------------------------------------------- #

#: Paper alias for :class:`PhaseClip` (RPX task Video Depth).
D1VClip = PhaseClip
#: Paper alias for :class:`PhaseClipDataset` (RPX task Video Depth).
D1VClipDataset = PhaseClipDataset
