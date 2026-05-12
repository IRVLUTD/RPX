"""On-the-fly deterministic NVS evaluation sample generation for RPX.

Generates (context_views, target_query) tuples for feed-forward NVS
evaluation. Uses the same scenes and phases as the RCPE pipeline
(Clutter + Clean only, same exclusions).

For each (scene, phase) sequence of ~250 frames, the generator
produces multiple evaluation configurations:

- **Varying context density**: N_context ∈ {2, 4, 8, 16} — how does
  rendering quality scale with number of input views?
- **Interpolation**: context views bracket the target — tests
  between-view rendering (easy).
- **Extrapolation**: target view is outside the context span — tests
  beyond-view prediction (hard).
- **Cross-phase**: context from Clutter, target from Clean — tests
  NVS under scene rearrangement (novel, unique to RPX).

The generator is deterministic (seeded), produces no disk writes, and
plugs into the existing RPXDataset interface.

Usage
-----
    from rpx_benchmark.nvs_pairs import NVSPairGenerator, NVSConfig

    gen = NVSPairGenerator(
        extracted_root=snap / "extracted",
        parquet_path=snap / "manifest/frames_v1.parquet",
        split="easy",
    )
    print(gen.summary())

    # Iterate: each sample has context views + target query + GT
    for sample in gen.iter_samples():
        context_rgbs = sample["context_rgbs"]       # list of HxWx3
        context_depths = sample["context_depths"]    # list of HxW
        context_poses = sample["context_poses"]      # list of 4x4
        target_pose = sample["target_pose"]          # 4x4
        gt_rgb = sample["target_rgb"]                # HxW3 (for eval)
        gt_depth = sample["target_depth"]            # HxW (for eval)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger

# Exclusions shared with RCPE pipeline
EXCLUDED_SCENE_IDS: Set[str] = {"scene58"}
EXCLUDED_SCENE_PHASES: Set[Tuple[str, int]] = {
    ("scene83.jsom.garden.pot", 0),
    ("scene98.jsom.atrium", 0),
    ("scene100.jsom.atrium", 0),
    ("scene50.ecss.out.stairs", 2),
    ("scene72.ecsw.atriumStairs", 2),
}
VALID_PHASES: Tuple[int, ...] = (0, 2)

log = get_logger(__name__)

RPX_SEED = 5_062_026

# Context view counts to evaluate at
DEFAULT_CONTEXT_COUNTS: Tuple[int, ...] = (2, 4, 8, 16)

# Number of target views per context configuration per (scene, phase)
DEFAULT_TARGETS_PER_CONFIG = 25


@dataclass
class NVSConfig:
    """Configuration for NVS evaluation sample generation."""

    context_counts: Tuple[int, ...] = DEFAULT_CONTEXT_COUNTS
    targets_per_config: int = DEFAULT_TARGETS_PER_CONFIG
    include_cross_phase: bool = True
    cross_phase_targets: int = 15  # per scene per context count
    seed: int = RPX_SEED


@dataclass
class NVSSample:
    """One NVS evaluation sample — context views + target query."""

    id: str
    scene_id: str
    phase: int                      # phase of context views
    phase_target: int               # phase of target view (same or different)
    n_context: int

    # Paths (loaded lazily by the consumer)
    context_rgb_paths: List[str]
    context_depth_paths: List[str]
    context_pose_paths: List[str]
    target_rgb_path: str            # GT for evaluation
    target_depth_path: str          # GT for depth evaluation (novel)
    target_pose_path: str           # query camera pose

    # Metadata
    sample_type: str                # "interpolation" | "extrapolation" | "cross_phase"
    context_frame_idxs: List[int]
    target_frame_idx: int
    difficulty: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scene_id": self.scene_id,
            "phase": self.phase,
            "phase_target": self.phase_target,
            "n_context": self.n_context,
            "sample_type": self.sample_type,
            "context_rgb_paths": self.context_rgb_paths,
            "context_depth_paths": self.context_depth_paths,
            "context_pose_paths": self.context_pose_paths,
            "target_rgb_path": self.target_rgb_path,
            "target_depth_path": self.target_depth_path,
            "target_pose_path": self.target_pose_path,
            "context_frame_idxs": self.context_frame_idxs,
            "target_frame_idx": self.target_frame_idx,
            "difficulty": self.difficulty,
            "metadata": {
                "sample_type": self.sample_type,
                "n_context": self.n_context,
                "scene_id": self.scene_id,
                "phase": self.phase,
                "phase_target": self.phase_target,
            },
        }


def _modality_path(scene: str, phase: int, modality: str, frame_idx: int) -> str:
    return f"scenes/{scene}/{phase}/{modality}/{frame_idx:05d}.{'npz' if modality == 'cam_pose' else 'png'}"


def _is_valid(scene: str, phase: int) -> bool:
    if scene in EXCLUDED_SCENE_IDS:
        return False
    if (scene, phase) in EXCLUDED_SCENE_PHASES:
        return False
    if phase not in VALID_PHASES:
        return False
    return True


class NVSPairGenerator:
    """Deterministic on-the-fly NVS evaluation sample generator.

    Loads only poses at construction time. RGB/depth loaded by the
    consumer at evaluation time.
    """

    def __init__(
        self,
        extracted_root: Path | str,
        parquet_path: Path | str,
        split: str,
        config: NVSConfig | None = None,
        snapshot_root: Path | str | None = None,
        repo_id: str | None = None,
    ) -> None:
        self.root = Path(extracted_root)
        self.split = split
        self.cfg = config or NVSConfig()
        self._snapshot_root = Path(snapshot_root) if snapshot_root else None
        self._repo_id = repo_id

        self._sequences: Dict[Tuple[str, int], List[int]] = {}
        self._cross_scenes: List[str] = []
        self._load_metadata(Path(parquet_path))

    def _load_metadata(self, parquet_path: Path) -> None:
        """Load frame indices per (scene, phase). Lightweight — no pixel data."""
        df = pd.read_parquet(parquet_path)

        scene_split = df.groupby("scene_id")["split"].agg(
            lambda s: s.value_counts().idxmax()
        )
        df = df.drop(columns=["split"]).merge(
            scene_split.rename("split"), left_on="scene_id", right_index=True
        )
        df = df[
            (df["split"].astype(str) == self.split)
            & df["has_cam_pose"].fillna(False).astype(bool)
            & df["has_rgb"].fillna(False).astype(bool)
            & df["has_depth"].fillna(False).astype(bool)
        ]

        for (scene, phase), grp in df.groupby(["scene_id", "phase"], sort=True):
            phase = int(phase)
            if not _is_valid(scene, phase):
                continue
            frame_idxs = sorted(int(r["frame_idx"]) for _, r in grp.iterrows())
            self._sequences[(scene, phase)] = frame_idxs

        scenes_0 = {s for (s, p) in self._sequences if p == 0}
        scenes_2 = {s for (s, p) in self._sequences if p == 2}
        self._cross_scenes = sorted(scenes_0 & scenes_2)

        log.info(
            "NVS generator: %d sequences, %d cross-phase scenes, split=%s",
            len(self._sequences), len(self._cross_scenes), self.split,
        )

    def _select_context_and_targets(
        self,
        frame_idxs: List[int],
        n_context: int,
        n_targets: int,
        rng: np.random.Generator,
    ) -> List[Tuple[List[int], int, str]]:
        """Select context views (evenly spaced) and target views.

        Returns list of (context_idxs, target_idx, sample_type).
        """
        n = len(frame_idxs)
        if n_context >= n:
            return []

        # Evenly space context views across the sequence
        context_positions = np.linspace(0, n - 1, n_context, dtype=int)
        context_idxs = [frame_idxs[p] for p in context_positions]
        context_set = set(context_positions)

        # Available targets: all frames NOT in context
        available = [i for i in range(n) if i not in context_set]
        if not available:
            return []

        # Split into interpolation (between first and last context) and
        # extrapolation (outside context span)
        interp = [i for i in available if context_positions[0] < i < context_positions[-1]]
        extrap = [i for i in available if i <= context_positions[0] or i >= context_positions[-1]]

        results = []

        # Interpolation targets
        n_interp = min(len(interp), n_targets // 2) if interp else 0
        if n_interp > 0:
            sel = rng.choice(len(interp), size=n_interp, replace=False)
            for idx in sel:
                results.append((context_idxs, frame_idxs[interp[idx]], "interpolation"))

        # Extrapolation targets
        n_extrap = min(len(extrap), n_targets - n_interp) if extrap else 0
        if n_extrap > 0:
            sel = rng.choice(len(extrap), size=n_extrap, replace=False)
            for idx in sel:
                results.append((context_idxs, frame_idxs[extrap[idx]], "extrapolation"))

        return results

    def iter_samples(self) -> Iterator[NVSSample]:
        """Yield NVS evaluation samples deterministically."""
        rng = np.random.default_rng(self.cfg.seed)
        cfg = self.cfg

        # ── Intra-phase samples ──────────────────────────────────────
        for (scene, phase), frame_idxs in sorted(self._sequences.items()):
            for n_ctx in cfg.context_counts:
                pairs = self._select_context_and_targets(
                    frame_idxs, n_ctx, cfg.targets_per_config, rng,
                )
                for ctx_idxs, tgt_idx, stype in pairs:
                    sample_id = (
                        f"{scene}__{phase}__ctx{n_ctx}__{stype}"
                        f"__{'_'.join(f'{c:05d}' for c in ctx_idxs[:2])}"
                        f"__tgt{tgt_idx:05d}"
                    )
                    yield NVSSample(
                        id=sample_id,
                        scene_id=scene,
                        phase=phase,
                        phase_target=phase,
                        n_context=n_ctx,
                        context_rgb_paths=[
                            _modality_path(scene, phase, "rgb", c) for c in ctx_idxs
                        ],
                        context_depth_paths=[
                            _modality_path(scene, phase, "depth", c) for c in ctx_idxs
                        ],
                        context_pose_paths=[
                            _modality_path(scene, phase, "cam_pose", c) for c in ctx_idxs
                        ],
                        target_rgb_path=_modality_path(scene, phase, "rgb", tgt_idx),
                        target_depth_path=_modality_path(scene, phase, "depth", tgt_idx),
                        target_pose_path=_modality_path(scene, phase, "cam_pose", tgt_idx),
                        sample_type=stype,
                        context_frame_idxs=ctx_idxs,
                        target_frame_idx=tgt_idx,
                        difficulty=self.split,
                    )

        # ── Cross-phase samples ──────────────────────────────────────
        if cfg.include_cross_phase:
            for scene in self._cross_scenes:
                frames_0 = self._sequences[(scene, 0)]
                frames_2 = self._sequences[(scene, 2)]
                for n_ctx in cfg.context_counts:
                    if n_ctx >= len(frames_0):
                        continue
                    # Context from Clutter (phase 0), target from Clean (phase 2)
                    ctx_positions = np.linspace(0, len(frames_0) - 1, n_ctx, dtype=int)
                    ctx_idxs = [frames_0[p] for p in ctx_positions]

                    n_tgt = min(len(frames_2), cfg.cross_phase_targets)
                    tgt_sel = rng.choice(len(frames_2), size=n_tgt, replace=False)
                    for ti in tgt_sel:
                        tgt_idx = frames_2[ti]
                        sample_id = (
                            f"{scene}__cross__ctx{n_ctx}"
                            f"__{'_'.join(f'{c:05d}' for c in ctx_idxs[:2])}"
                            f"__tgt{tgt_idx:05d}"
                        )
                        yield NVSSample(
                            id=sample_id,
                            scene_id=scene,
                            phase=0,
                            phase_target=2,
                            n_context=n_ctx,
                            context_rgb_paths=[
                                _modality_path(scene, 0, "rgb", c) for c in ctx_idxs
                            ],
                            context_depth_paths=[
                                _modality_path(scene, 0, "depth", c) for c in ctx_idxs
                            ],
                            context_pose_paths=[
                                _modality_path(scene, 0, "cam_pose", c) for c in ctx_idxs
                            ],
                            target_rgb_path=_modality_path(scene, 2, "rgb", tgt_idx),
                            target_depth_path=_modality_path(scene, 2, "depth", tgt_idx),
                            target_pose_path=_modality_path(scene, 2, "cam_pose", tgt_idx),
                            sample_type="cross_phase",
                            context_frame_idxs=ctx_idxs,
                            target_frame_idx=tgt_idx,
                            difficulty=self.split,
                        )

    def samples(self) -> List[NVSSample]:
        """Materialise all samples as a list."""
        return list(self.iter_samples())

    def summary(self) -> str:
        """Human-readable summary."""
        samples = self.samples()
        by_type = {}
        by_ctx = {}
        for s in samples:
            by_type[s.sample_type] = by_type.get(s.sample_type, 0) + 1
            by_ctx[s.n_context] = by_ctx.get(s.n_context, 0) + 1

        scenes = len({s.scene_id for s in samples})
        lines = [
            f"RPX-NVS Pair Generator — split={self.split}",
            f"  Scenes:           {scenes}",
            f"  Total samples:    {len(samples):,}",
            f"  By type:          {by_type}",
            f"  By n_context:     {by_ctx}",
            f"  Context counts:   {self.cfg.context_counts}",
            f"  Cross-phase:      {self.cfg.include_cross_phase}",
            f"  Seed:             {self.cfg.seed}",
        ]
        return "\n".join(lines)
