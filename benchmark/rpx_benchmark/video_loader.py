"""Sequence-shaped loader for video tasks (D1-V today; future video tasks).

Iterates per ``(scene, phase)`` clip, yielding
:class:`~rpx_benchmark.api.VideoSample`. The loader supports an
optional ``frame_budget`` + ``sampling`` mode for the temporal-
resolution ablation study (run at full 250 frames + downsampled
budgets such as 150/75/25 to characterise how each video depth model
degrades with sparser input).

Cell-log granularity
--------------------

One ``(scene, phase)`` clip → one cell-log row keyed
``(model, task, scene, phase, frame_budget)``. ``frame_budget=None``
(or the literal 0) is the canonical "full clip" row that feeds Φ;
non-None budgets feed the ablation plot. Φ is computed only over
``frame_budget=None`` rows to keep the comparison across models
honest.

Sampling modes
--------------

* ``"all"`` — every frame on disk (the default). ``frame_budget`` must
  be ``None`` in this mode.
* ``"stride"`` — uniform stride keeping the first and last frame
  anchored. Reduces to FPS-in-1D-timestamp when frames are evenly
  spaced (our 30 fps capture is).
* ``"fps_se3"`` — farthest-point sampling in 3D camera-translation
  space (T265 positions). First and last frame anchored. Selects
  ``frame_budget`` frames that maximise spatial coverage of the
  camera path. This is the most interesting mode for a robotics
  paper: it answers "does losing frames hurt because of less data
  or because we miss key viewpoints?" — only the second is
  diagnostic. Deterministic given ``(scene, phase, frame_budget)``.

Per-clip scale-and-shift alignment for affine-invariant models is
applied at the runner level, not here — this loader is decode-only.

Memory budget per full-budget clip at D435 resolution: ~540 MB
(250 × 640 × 480 × 3 B RGB + 250 × 640 × 480 × 4 B depth). Workable
on a 64 GB host with one clip in flight per worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Literal, Optional

import numpy as np

from .api import TaskType, VideoDepthGroundTruth, VideoSample
from .decode_contracts import safe_load_depth, safe_load_pose, safe_load_rgb
from .exceptions import ConfigError, ManifestError
from .logging_utils import get_logger

log = get_logger(__name__)


SamplingMode = Literal["all", "stride", "fps_se3"]


@dataclass
class D1VDataset:
    """Iterates per ``(scene, phase)`` clip for the video-depth task.

    Yielded value is :class:`~rpx_benchmark.api.VideoSample` whose
    ``rgb_seq`` is a ``(T, H, W, 3) uint8`` stack and whose
    ``ground_truth`` is :class:`~rpx_benchmark.api.VideoDepthGroundTruth`
    with the same ``T``. When ``frame_budget`` is ``None`` (the
    canonical run), ``T`` is whatever the phase has on disk; otherwise
    ``T == frame_budget`` and ``metadata['frame_indices']`` records
    which original-phase indices were kept.

    Parameters
    ----------
    samples : list[dict]
        Per-clip manifest entries (one dict per (scene, phase)).
        Produced by
        :func:`rpx_benchmark.dataset_hub.split_manifests.write_split_manifests`
        in its ``"video_depth"`` mode. Each dict carries ``scene_id``,
        ``phase``, ``frame_filenames`` (sorted list), and optionally
        ``pose_filenames``, ``intrinsics`` (3×3 list), ``difficulty``.
    task : TaskType
        Must be :data:`~rpx_benchmark.api.TaskType.VIDEO_DEPTH`.
    root : Path
        Snapshot root that ``frame_filenames`` and ``pose_filenames``
        are resolved against.
    batch_size : int
        Always 1 for D1-V — clips are not stackable. Kept for API
        uniformity with :class:`~rpx_benchmark.loader.RPXDataset`.
    frame_budget : int or None
        If set, subsample each clip to exactly this many frames via
        ``sampling``. ``None`` means use every frame on disk.
    sampling : Literal["all", "stride", "fps_se3"]
        Subsampling strategy. ``"all"`` requires
        ``frame_budget is None``. ``"stride"`` is uniform-with-anchored-
        endpoints. ``"fps_se3"`` is farthest-point sampling in T265
        position space (requires ``pose_filenames`` in the manifest
        entry).
    """

    samples: List[dict]
    task: TaskType
    root: Path
    batch_size: int = 1
    frame_budget: Optional[int] = None
    sampling: SamplingMode = "all"

    def __post_init__(self) -> None:
        if self.sampling == "all" and self.frame_budget is not None:
            raise ConfigError(
                "sampling='all' is incompatible with a frame_budget",
                hint="Drop the budget (None means take every frame) or "
                "pick sampling='stride' or 'fps_se3'.",
            )
        if self.sampling != "all" and self.frame_budget is None:
            raise ConfigError(
                f"sampling={self.sampling!r} requires a frame_budget",
                hint="Pass frame_budget=<int> or set sampling='all'.",
            )
        if self.frame_budget is not None and self.frame_budget < 2:
            raise ConfigError(
                f"frame_budget must be >= 2 (got {self.frame_budget})",
                hint="Temporal metrics need at least two frames.",
            )

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[List[VideoSample]]:
        for group in self.samples:
            yield [self._load_one_clip(group)]

    # ------------------------------------------------------------------ #
    # Per-clip loading
    # ------------------------------------------------------------------ #

    def _load_one_clip(self, group: dict) -> VideoSample:
        scene = group["scene_id"]
        phase = group["phase"]
        frame_files = list(group["frame_filenames"])
        pose_files = list(group.get("pose_filenames") or [])

        if not frame_files:
            raise ManifestError(
                f"clip {scene}/{phase}: frame_filenames is empty",
            )
        if pose_files and len(pose_files) != len(frame_files):
            raise ManifestError(
                f"clip {scene}/{phase}: pose_filenames length "
                f"{len(pose_files)} != frame_filenames length "
                f"{len(frame_files)}",
            )

        # 1. Decide which frame indices to keep.
        T_full = len(frame_files)
        if self.frame_budget is None:
            keep = np.arange(T_full, dtype=np.int32)
        else:
            if self.frame_budget > T_full:
                raise ManifestError(
                    f"clip {scene}/{phase}: frame_budget="
                    f"{self.frame_budget} exceeds available frame count "
                    f"{T_full}",
                )
            if self.sampling == "stride":
                keep = _stride_sample(T_full, self.frame_budget)
            elif self.sampling == "fps_se3":
                if not pose_files:
                    raise ManifestError(
                        f"clip {scene}/{phase}: sampling='fps_se3' "
                        f"requires pose_filenames in the manifest entry",
                    )
                positions = _read_translation_track(self.root, pose_files)
                keep = _fps_se3(positions, self.frame_budget)
            else:  # pragma: no cover — guarded by __post_init__
                raise ConfigError(
                    f"unknown sampling mode {self.sampling!r}",
                    hint="Pick one of 'all', 'stride', 'fps_se3'.",
                )

        # 2. Decode RGB + depth + poses ONLY for the kept indices.
        #    Reading 250 PNGs then throwing 200 away would waste 80% of
        #    the I/O budget; kept-only decode is the only sane strategy
        #    when the budget is small.
        rgb_seq = np.stack([safe_load_rgb(self.root / frame_files[i]) for i in keep])
        # Frame files come from the rgb manifest; depth files mirror
        # the layout (rgb/<stem>.png ↔ depth/<stem>.png). Group dicts
        # carry an explicit ``depth_filenames`` list to avoid relying on
        # that convention; fall back if it's absent.
        depth_files = group.get("depth_filenames") or [
            _swap_modality(p, "rgb", "depth") for p in frame_files
        ]
        depth_u16 = np.stack([safe_load_depth(self.root / depth_files[i]) for i in keep])
        depth_seq = depth_u16.astype(np.float32) / 1000.0
        valid_seq = depth_u16 > 0  # D435 zero sentinel

        camera_pose_seq: Optional[np.ndarray] = None
        if pose_files:
            poses = []
            for i in keep:
                pos, quat = safe_load_pose(self.root / pose_files[i])
                T_mat = _se3_from_pos_quat(pos, quat)
                poses.append(T_mat)
            camera_pose_seq = np.stack(poses)

        gt = VideoDepthGroundTruth(
            depth_map_seq=depth_seq,
            valid_mask_seq=valid_seq,
            frame_indices=keep.astype(np.int32),
        )
        # Stash optional extras the temporal calculator may want on the
        # GT object (the calculator reads them via getattr so absence is
        # graceful).
        if camera_pose_seq is not None:
            object.__setattr__(gt, "poses", camera_pose_seq)
        object.__setattr__(gt, "rgb_seq", rgb_seq)

        metadata: dict = {
            "scene_id": scene,
            "phase_idx": phase,
            "frame_indices": keep.tolist(),
            "frame_budget": int(self.frame_budget) if self.frame_budget else 0,
            "sampling": self.sampling,
        }
        if "intrinsics" in group:
            metadata["intrinsics"] = np.asarray(group["intrinsics"], dtype=np.float32)

        from .api import Difficulty, Phase

        _PHASE_BY_INT = {0: Phase.CLUTTER, 1: Phase.INTERACTION, 2: Phase.CLEAN}
        try:
            phase_enum: Phase | None = (
                Phase(phase) if isinstance(phase, str) else _PHASE_BY_INT.get(int(phase))
            )
        except (ValueError, TypeError):
            phase_enum = None
        difficulty_str = group.get("difficulty")
        try:
            difficulty_enum: Difficulty | None = (
                Difficulty(difficulty_str) if difficulty_str else None
            )
        except ValueError:
            difficulty_enum = None

        return VideoSample(
            id=f"{scene}_{phase}_budget{metadata['frame_budget']}",
            rgb_seq=rgb_seq,
            ground_truth=gt,
            metadata=metadata,
            phase=phase_enum,
            difficulty=difficulty_enum,
            camera_pose_seq=camera_pose_seq,
        )

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        batch_size: int = 1,
        frame_budget: Optional[int] = None,
        sampling: SamplingMode = "all",
    ) -> "D1VDataset":
        """Build a D1VDataset from a per-clip JSON manifest.

        Expected JSON shape::

            {
              "task": "video_depth",
              "root": "/abs/path/or/snapshot/root",
              "samples": [
                {
                  "scene_id": "scene_001",
                  "phase": 0,
                  "frame_filenames": ["scenes/scene_001/0/rgb/00000.png", ...],
                  "depth_filenames": ["scenes/scene_001/0/depth/00000.png", ...],
                  "pose_filenames":  ["scenes/scene_001/0/cam_pose/00000.npz", ...],
                  "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],
                  "difficulty": "easy"
                },
                ...
              ]
            }

        The writer for this shape lives in
        :func:`rpx_benchmark.dataset_hub.split_manifests.write_split_manifests`
        under the ``video_depth`` recipe.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise ManifestError(
                f"Manifest file not found: {manifest_path}",
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ManifestError(
                f"Manifest at {manifest_path} is not valid JSON: {e}",
            ) from e
        task_str = payload.get("task")
        if task_str != "video_depth":
            raise ManifestError(
                f"D1VDataset.from_manifest expected task='video_depth'; got {task_str!r}",
            )
        root = Path(payload.get("root") or manifest_path.parent)
        samples = payload.get("samples") or []
        if not isinstance(samples, list):
            raise ManifestError(
                f"Manifest 'samples' must be a list, got {type(samples).__name__}",
            )
        return cls(
            samples=samples,
            task=TaskType.VIDEO_DEPTH,
            root=root,
            batch_size=batch_size,
            frame_budget=frame_budget,
            sampling=sampling,
        )


# --------------------------------------------------------------------------- #
# Frame-selection helpers
# --------------------------------------------------------------------------- #


def _stride_sample(T_full: int, budget: int) -> np.ndarray:
    """Uniform-stride frame selection with first/last anchored.

    Equivalent to FPS in 1D timestamp space at constant frame rate.
    Always includes indices 0 and ``T_full - 1`` so the clip endpoints
    are preserved (matters for temporal-edge metrics).
    """
    if budget == T_full:
        return np.arange(T_full, dtype=np.int32)
    # np.linspace(0, T_full-1, budget) gives evenly-spaced floats with
    # both endpoints; round and dedupe.
    raw = np.linspace(0, T_full - 1, budget)
    idx = np.unique(np.round(raw).astype(np.int32))
    # In rare cases rounding collapses adjacent picks; pad with the
    # nearest missing indices to hit the exact budget.
    if len(idx) < budget:
        all_idx = set(idx.tolist())
        for cand in range(T_full):
            if cand not in all_idx:
                all_idx.add(cand)
                if len(all_idx) == budget:
                    break
        idx = np.sort(np.fromiter(all_idx, dtype=np.int32))
    return idx[:budget]


def _fps_se3(positions: np.ndarray, budget: int) -> np.ndarray:
    """Farthest-point sampling in 3D camera-translation space.

    Anchors the first frame (index 0) and iteratively adds the index
    whose minimum Euclidean distance to the selected set is largest.
    Deterministic given ``(positions, budget)``.

    Parameters
    ----------
    positions : (T_full, 3) float64
        T265 camera positions in metres.
    budget : int
        Number of frames to select. Must satisfy ``2 <= budget <= T_full``.

    Returns
    -------
    np.ndarray
        Sorted indices, shape ``(budget,)``, dtype int32.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ConfigError(
            f"_fps_se3: expected (T, 3) positions, got {positions.shape}",
            hint="Pass T265 translations as an (N, 3) float64 array.",
        )
    T_full = positions.shape[0]
    if not (2 <= budget <= T_full):
        raise ConfigError(
            f"_fps_se3: budget {budget} out of range [2, {T_full}]",
            hint="Budget must be in [2, len(positions)].",
        )

    selected = [0]
    # Squared distance from each frame to the closest selected frame.
    sq_dist_to_set = np.sum((positions - positions[0]) ** 2, axis=1)
    sq_dist_to_set[0] = -np.inf  # never re-pick

    while len(selected) < budget:
        next_idx = int(np.argmax(sq_dist_to_set))
        selected.append(next_idx)
        # Update min-distance bookkeeping.
        new_sq = np.sum((positions - positions[next_idx]) ** 2, axis=1)
        sq_dist_to_set = np.minimum(sq_dist_to_set, new_sq)
        sq_dist_to_set[next_idx] = -np.inf

    return np.sort(np.array(selected, dtype=np.int32))


def _se3_from_pos_quat(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Pack a (3,) position + (4,) xyzw quaternion into a 4×4 SE(3)."""
    qx, qy, qz, qw = quat_xyzw
    # Standard xyzw → rotation matrix
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    T_mat = np.eye(4, dtype=np.float64)
    T_mat[:3, :3] = R
    T_mat[:3, 3] = position
    return T_mat


def _read_translation_track(root: Path, pose_files: List[str]) -> np.ndarray:
    """Read pose files and return ``(T, 3) float64`` translation track.

    Uses :func:`~rpx_benchmark.decode_contracts.safe_load_pose` to honour
    both ``.npz`` (legacy) and ``.npy`` (v2) on-disk formats.
    """
    positions = []
    for p in pose_files:
        pos, _quat = safe_load_pose(root / p)
        positions.append(pos)
    return np.stack(positions).astype(np.float64)


def _swap_modality(rel_path: str, src: str, dst: str) -> str:
    """``scenes/x/0/rgb/00000.png`` → ``scenes/x/0/depth/00000.png``.

    Only swaps the *first* occurrence of ``/src/`` to avoid corrupting
    paths that happen to mention ``rgb`` or ``depth`` elsewhere.
    """
    needle = f"/{src}/"
    repl = f"/{dst}/"
    i = rel_path.find(needle)
    if i < 0:
        return rel_path  # caller will surface a missing-file error downstream
    return rel_path[:i] + repl + rel_path[i + len(needle) :]


# --------------------------------------------------------------------------- #
# Per-clip scale-and-shift alignment for affine-invariant video models
# --------------------------------------------------------------------------- #


def align_scale_and_shift_per_clip(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    valid_mask_seq: np.ndarray,
) -> np.ndarray:
    """Per-clip scale + shift alignment — thin wrapper around the shared
    pooled solver.

    Delegates to :func:`~rpx_benchmark.metrics.depth_alignment.align_pred_to_gt_pooled`
    in ``ls_affine`` mode so the math lives in exactly one place
    (shared with the D1-F per-scene-phase aligner). Per-clip (not
    per-frame) alignment is the standard convention for video depth —
    per-frame alignment would zero out the temporal-consistency
    signal the video metrics measure.

    The returned sequence is ``s * pred_seq + t`` so callers can plug
    it straight into the metric calculators.
    """
    from .metrics.depth_alignment import align_pred_to_gt_pooled

    if pred_seq.shape != gt_seq.shape or pred_seq.shape != valid_mask_seq.shape:
        raise ConfigError(
            f"align_scale_and_shift_per_clip: shape mismatch "
            f"pred={pred_seq.shape} gt={gt_seq.shape} valid={valid_mask_seq.shape}",
            hint="All three sequences must be (T, H, W) with matching T/H/W.",
        )
    return align_pred_to_gt_pooled(
        pred_seq=pred_seq,
        gt_seq=gt_seq,
        mode="ls_affine",
        valid_seq=valid_mask_seq.astype(bool),
    )


__all__ = [
    "D1VDataset",
    "_fps_se3",
    "_stride_sample",
    "align_scale_and_shift_per_clip",
]
