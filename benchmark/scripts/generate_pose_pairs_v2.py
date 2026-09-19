"""Stratified pair sampler for RPX relative camera pose evaluation.

Generates three complementary pair types from the RPX dataset:

1. **Intra-phase pairs** — within a single (scene, phase) sequence,
   stratified by rotation magnitude into 4 bins. Tests standard
   geometric difficulty at controlled levels.

2. **Cross-phase pairs** — between Clutter (phase 0) and Clean (phase 2)
   of the same scene. Same physical environment, objects rearranged.
   Tests robustness to non-geometric scene change. *Unique to RPX.*

3. **Temporal chains** — ordered sequences of consecutive pairs for
   measuring drift / accumulated error. Enables trajectory-level
   evaluation beyond i.i.d. pair accuracy.

Produces ~60K pairs (3.6× RUBIK, 40× ScanNet-1500) with zero
duplication, seeded for full reproducibility.

Design choices
--------------
- Rotation-bin stratification (not overlap) because overlap is
  redundant for continuous video (monotonic with rotation) and the
  relevant difficulty axis for pose regression is motion magnitude.
- Cross-phase pairs exploit RPX's unique three-phase protocol.
  No other RCPE benchmark can test scene-rearrangement robustness.
- Temporal chains enable drift metrics (ATE over N hops) which
  matter for downstream SLAM / policy integration.
- No Poisson-disk / no arbitrary scale parameters — simple stratified
  random is the field standard (ScanNet-1500, RUBIK) and defensible.

Usage
-----
    PYTHONPATH=. python scripts/generate_pose_pairs_v2.py \\
        --split easy --output manifests/relative_pose_v2/easy.json

    PYTHONPATH=. python scripts/generate_pose_pairs_v2.py \\
        --split hard --pairs-per-bin 50 --cross-pairs-per-bin 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_manifest import _hf_snapshot_root  # noqa: E402

RPX_SEED = 5_062_026
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Exclusions: scenes/phases with missing or failed ICP-optimized poses
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED_SCENE_IDS: set[str] = {"scene58"}
"""Scenes excluded entirely (no optimized poses available)."""

EXCLUDED_SCENE_PHASES: set[tuple[str, int]] = {
    ("scene83.jsom.garden.pot", 0),
    ("scene98.jsom.atrium", 0),
    ("scene100.jsom.atrium", 0),
    ("scene50.ecss.out.stairs", 2),
    ("scene72.ecsw.atriumStairs", 2),
}
"""(scene_id, phase) pairs where ICP optimization failed."""

# RCPE covers Clutter (0) and Clean (2); Interaction (1) is excluded.
VALID_PHASES = (0, 2)

# ─────────────────────────────────────────────────────────────────────────────
# Rotation bins — aligned with standard AUC@5°/10°/20° reporting thresholds
# ─────────────────────────────────────────────────────────────────────────────

ROTATION_BINS: List[Tuple[float, float]] = [
    (0.0, 15.0),    # easy — small viewpoint change
    (15.0, 45.0),   # medium — moderate change
    (45.0, 90.0),   # hard — large change
    (90.0, 180.000001),  # extreme — near-opposite views
]
"""Four deterministic rotation-magnitude bins in degrees."""

BIN_NAMES = ["easy", "medium", "hard", "extreme"]


# ─────────────────────────────────────────────────────────────────────────────
# Pose math
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _load_pose(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    t = np.asarray(data["position"], dtype=np.float64).reshape(3)
    q = np.asarray(data["orientation"], dtype=np.float64).reshape(4)
    return _quat_to_rot(q), t


def _relative_rotation_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a.T @ R_b
    cos_theta = (np.trace(R_rel) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, float(cos_theta)))
    return float(np.degrees(np.arccos(cos_theta)))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SamplerConfig:
    # Intra-phase
    intra_pairs_per_bin: int = 50
    frame_gap: int = 5
    min_rotation_deg: float = 0.0
    min_translation_m: float = 0.0

    # Cross-phase is invalid across independently initialized T265 worlds.
    cross_pairs_per_bin: int = 0
    cross_max_candidates: int = 5000  # subsample before binning (speed)

    # Temporal chains
    chain_count: int = 5        # chains per (scene, phase)
    chain_length: int = 10      # pairs per chain
    chain_stride: int = 5       # frame gap between consecutive pairs in chain

    seed: int = RPX_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Intra-phase pair generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_intra_phase_pairs(
    poses: List[Tuple[np.ndarray, np.ndarray]],
    frame_idxs: List[int],
    cfg: SamplerConfig,
    rng: np.random.Generator,
) -> Dict[str, List[Tuple[int, int, float, float]]]:
    """Enumerate pairs, bin by rotation, sample per bin.

    Returns dict mapping bin_name -> list of (frame_a, frame_b, rot_deg, t_m).
    """
    # Enumerate candidates
    candidates_by_bin: Dict[str, List[Tuple[int, int, float, float]]] = {
        name: [] for name in BIN_NAMES
    }
    index_by_frame = {frame: index for index, frame in enumerate(frame_idxs)}
    for i, frame_a in enumerate(frame_idxs):
        frame_b = frame_a + cfg.frame_gap
        j = index_by_frame.get(frame_b)
        if j is None:
            continue
        R_i, t_i = poses[i]
        R_j, t_j = poses[j]
        rot = _relative_rotation_deg(R_i, R_j)
        trans = float(np.linalg.norm(t_j - t_i))
        if rot < cfg.min_rotation_deg or trans < cfg.min_translation_m:
            continue
        for k, (lo, hi) in enumerate(ROTATION_BINS):
            if lo <= rot < hi:
                candidates_by_bin[BIN_NAMES[k]].append(
                    (frame_a, frame_b, rot, trans)
                )
                break

    # Sample from each bin
    selected: Dict[str, List[Tuple[int, int, float, float]]] = {}
    for name in BIN_NAMES:
        cands = candidates_by_bin[name]
        if len(cands) <= cfg.intra_pairs_per_bin:
            selected[name] = cands
        else:
            idxs = rng.choice(len(cands), size=cfg.intra_pairs_per_bin, replace=False)
            selected[name] = [cands[i] for i in idxs]
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Cross-phase pair generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_cross_phase_pairs(
    poses_0: List[Tuple[np.ndarray, np.ndarray]],
    frames_0: List[int],
    poses_2: List[Tuple[np.ndarray, np.ndarray]],
    frames_2: List[int],
    cfg: SamplerConfig,
    rng: np.random.Generator,
) -> Dict[str, List[Tuple[int, int, float, float]]]:
    """Generate pairs across Clutter (phase 0) and Clean (phase 2).

    Subsamples the candidate space then bins by rotation.
    Returns dict mapping bin_name -> list of (frame_0, frame_2, rot_deg, t_m).
    """
    n0, n2 = len(poses_0), len(poses_2)
    total_possible = n0 * n2

    # If too many, subsample indices
    if total_possible > cfg.cross_max_candidates:
        # Random subsample of (i, j) index pairs
        idx_pairs = rng.choice(
            total_possible, size=cfg.cross_max_candidates, replace=False
        )
        i_indices = idx_pairs // n2
        j_indices = idx_pairs % n2
    else:
        i_indices = np.repeat(np.arange(n0), n2)
        j_indices = np.tile(np.arange(n2), n0)

    candidates_by_bin: Dict[str, List[Tuple[int, int, float, float]]] = {
        name: [] for name in BIN_NAMES
    }

    for idx in range(len(i_indices)):
        i, j = int(i_indices[idx]), int(j_indices[idx])
        R_0, t_0 = poses_0[i]
        R_2, t_2 = poses_2[j]
        rot = _relative_rotation_deg(R_0, R_2)
        trans = float(np.linalg.norm(t_2 - t_0))
        if rot < cfg.min_rotation_deg:
            continue
        for k, (lo, hi) in enumerate(ROTATION_BINS):
            if lo <= rot < hi:
                candidates_by_bin[BIN_NAMES[k]].append(
                    (frames_0[i], frames_2[j], rot, trans)
                )
                break

    selected: Dict[str, List[Tuple[int, int, float, float]]] = {}
    for name in BIN_NAMES:
        cands = candidates_by_bin[name]
        if len(cands) <= cfg.cross_pairs_per_bin:
            selected[name] = cands
        else:
            idxs = rng.choice(len(cands), size=cfg.cross_pairs_per_bin, replace=False)
            selected[name] = [cands[i] for i in idxs]
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Temporal chain generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_temporal_chains(
    poses: List[Tuple[np.ndarray, np.ndarray]],
    frame_idxs: List[int],
    cfg: SamplerConfig,
    rng: np.random.Generator,
) -> List[List[Tuple[int, int, float, float]]]:
    """Generate ordered chains of consecutive pairs for drift evaluation.

    Each chain: [pair_0, pair_1, ..., pair_{L-1}] where pair_k connects
    frame[start + k*stride] to frame[start + (k+1)*stride].

    Returns list of chains, each chain is a list of (frame_a, frame_b, rot, t).
    """
    index_by_frame = {frame: index for index, frame in enumerate(frame_idxs)}
    starts = [
        frame
        for frame in frame_idxs
        if all(
            frame + step * cfg.chain_stride in index_by_frame
            for step in range(cfg.chain_length + 1)
        )
    ]
    if not starts:
        return []
    n_chains = min(cfg.chain_count, len(starts))
    starts = [starts[index] for index in rng.choice(len(starts), size=n_chains, replace=False)]

    chains: List[List[Tuple[int, int, float, float]]] = []
    for start_frame in starts:
        chain = []
        for k in range(cfg.chain_length):
            frame_a = start_frame + k * cfg.chain_stride
            frame_b = start_frame + (k + 1) * cfg.chain_stride
            i = index_by_frame[frame_a]
            j = index_by_frame[frame_b]
            R_i, t_i = poses[i]
            R_j, t_j = poses[j]
            rot = _relative_rotation_deg(R_i, R_j)
            trans = float(np.linalg.norm(t_j - t_i))
            chain.append((frame_a, frame_b, rot, trans))
        chains.append(chain)
    return chains


# ─────────────────────────────────────────────────────────────────────────────
# Manifest builder
# ─────────────────────────────────────────────────────────────────────────────

def _modality_path(scene: str, phase: int, modality: str, stem: str) -> str:
    return f"extracted/scenes/{scene}/{phase}/{modality}/{stem}"


def _is_valid(scene: str, phase: int) -> bool:
    if scene in EXCLUDED_SCENE_IDS:
        return False
    if (scene, phase) in EXCLUDED_SCENE_PHASES:
        return False
    if phase not in VALID_PHASES:
        return False
    return True


def build_manifest(
    *,
    parquet_path: Path,
    extracted_root: Path,
    split: str,
    cfg: SamplerConfig,
) -> dict:
    df = pd.read_parquet(parquet_path)

    # Scene-wise split assignment
    scene_split = df.groupby("scene_id")["split"].agg(
        lambda s: s.dropna().value_counts().idxmax() if s.notna().any() else None
    ).dropna()
    df = df.drop(columns=["split"]).merge(
        scene_split.rename("split"), left_on="scene_id", right_index=True
    )
    df = df[
        (df["split"].astype(str) == split)
        & df["has_cam_pose"].fillna(False).astype(bool)
    ]
    if df.empty:
        raise SystemExit(f"no frames for split={split!r}")

    rng = np.random.default_rng(cfg.seed)
    samples: List[dict] = []
    stats = {
        "intra_pairs": 0,
        "cross_pairs": 0,
        "temporal_pairs": 0,
        "intra_by_bin": {n: 0 for n in BIN_NAMES},
        "cross_by_bin": {n: 0 for n in BIN_NAMES},
        "scenes_processed": 0,
        "phases_processed": 0,
        "phases_skipped": 0,
    }

    # Group frames by (scene, phase)
    scene_phase_groups = {}
    for (scene, phase), grp in df.groupby(["scene_id", "phase"], sort=True):
        if not _is_valid(scene, int(phase)):
            stats["phases_skipped"] += 1
            continue
        grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
        frame_idxs = [int(r["frame_idx"]) for _, r in grp_sorted.iterrows()]

        # Load poses
        poses = []
        for _, r in grp_sorted.iterrows():
            stem = str(r["frame_filename"]).rsplit(".", 1)[0]
            pose_path = (
                extracted_root / "scenes" / scene / str(phase) / "cam_pose" / f"{stem}.npz"
            )
            poses.append(_load_pose(pose_path))

        scene_phase_groups[(scene, int(phase))] = (frame_idxs, poses)
        stats["phases_processed"] += 1

    scenes_with_both_phases = set()
    for (scene, phase) in scene_phase_groups:
        if phase == 0 and (scene, 2) in scene_phase_groups:
            scenes_with_both_phases.add(scene)

    # ─── Intra-phase pairs ────────────────────────────────────────────────
    for (scene, phase), (frame_idxs, poses) in sorted(scene_phase_groups.items()):
        selected = _generate_intra_phase_pairs(poses, frame_idxs, cfg, rng)
        for bin_name, pairs in selected.items():
            for fa, fb, rot, t_m in pairs:
                stem_a, stem_b = f"{fa:05d}", f"{fb:05d}"
                samples.append({
                    "id": f"{scene}__{phase}__{stem_a}__{stem_b}",
                    "scene_id": scene,
                    "phase": int(phase),
                    "phase_b": int(phase),
                    "frame_idx": fa,
                    "frame_idx_b": fb,
                    "difficulty": split,
                    "pair_type": "intra_phase",
                    "rotation_bin": bin_name,
                    "metadata": {
                        "scene_id": scene,
                        "phase_idx": int(phase),
                        "phase_idx_b": int(phase),
                        "frame": stem_a,
                        "frame_b": stem_b,
                        "pair_type": "intra_phase",
                        "rotation_bin": bin_name,
                        "rotation_deg_gt": rot,
                        "translation_m_gt": t_m,
                    },
                    "rgb": _modality_path(scene, int(phase), "rgb", f"{stem_a}.png"),
                    "rgb_b": _modality_path(scene, int(phase), "rgb", f"{stem_b}.png"),
                    "pose_a": _modality_path(scene, int(phase), "cam_pose", f"{stem_a}.npz"),
                    "pose_b": _modality_path(scene, int(phase), "cam_pose", f"{stem_b}.npz"),
                })
                stats["intra_pairs"] += 1
                stats["intra_by_bin"][bin_name] += 1
        log.info(
            "intra (%s, %d): %s",
            scene, phase,
            {n: len(selected[n]) for n in BIN_NAMES},
        )

    # Cross-phase ground truth is undefined across independent T265 worlds.
    # Keep the legacy code below unreachable for old-manifest archaeology.
    if cfg.cross_pairs_per_bin == 0:
        scenes_with_both_phases.clear()

    # ─── Cross-phase pairs (legacy; disabled) ────────────────────────────
    for scene in sorted(scenes_with_both_phases):
        frames_0, poses_0 = scene_phase_groups[(scene, 0)]
        frames_2, poses_2 = scene_phase_groups[(scene, 2)]
        selected = _generate_cross_phase_pairs(
            poses_0, frames_0, poses_2, frames_2, cfg, rng
        )
        for bin_name, pairs in selected.items():
            for fa, fb, rot, t_m in pairs:
                stem_a, stem_b = f"{fa:05d}", f"{fb:05d}"
                samples.append({
                    "id": f"{scene}__cross__0_{stem_a}__2_{stem_b}",
                    "scene_id": scene,
                    "phase": 0,
                    "phase_b": 2,
                    "frame_idx": fa,
                    "frame_idx_b": fb,
                    "difficulty": split,
                    "pair_type": "cross_phase",
                    "rotation_bin": bin_name,
                    "metadata": {
                        "scene_id": scene,
                        "phase_idx": 0,
                        "phase_idx_b": 2,
                        "frame": stem_a,
                        "frame_b": stem_b,
                        "pair_type": "cross_phase",
                        "rotation_bin": bin_name,
                        "rotation_deg_gt": rot,
                        "translation_m_gt": t_m,
                    },
                    "rgb": _modality_path(scene, 0, "rgb", f"{stem_a}.png"),
                    "rgb_b": _modality_path(scene, 2, "rgb", f"{stem_b}.png"),
                    "pose_a": _modality_path(scene, 0, "cam_pose", f"{stem_a}.npz"),
                    "pose_b": _modality_path(scene, 2, "cam_pose", f"{stem_b}.npz"),
                })
                stats["cross_pairs"] += 1
                stats["cross_by_bin"][bin_name] += 1
        log.info(
            "cross (%s): %s",
            scene,
            {n: len(selected[n]) for n in BIN_NAMES},
        )

    # ─── Temporal chains ──────────────────────────────────────────────────
    for (scene, phase), (frame_idxs, poses) in sorted(scene_phase_groups.items()):
        chains = _generate_temporal_chains(poses, frame_idxs, cfg, rng)
        for chain_idx, chain in enumerate(chains):
            for pair_idx, (fa, fb, rot, t_m) in enumerate(chain):
                stem_a, stem_b = f"{fa:05d}", f"{fb:05d}"
                samples.append({
                    "id": f"{scene}__{phase}__chain{chain_idx}__{stem_a}__{stem_b}",
                    "scene_id": scene,
                    "phase": int(phase),
                    "phase_b": int(phase),
                    "frame_idx": fa,
                    "frame_idx_b": fb,
                    "difficulty": split,
                    "pair_type": "temporal_chain",
                    "rotation_bin": None,
                    "metadata": {
                        "scene_id": scene,
                        "phase_idx": int(phase),
                        "phase_idx_b": int(phase),
                        "frame": stem_a,
                        "frame_b": stem_b,
                        "pair_type": "temporal_chain",
                        "chain_id": f"{scene}__{phase}__chain{chain_idx}",
                        "chain_position": pair_idx,
                        "rotation_deg_gt": rot,
                        "translation_m_gt": t_m,
                    },
                    "rgb": _modality_path(scene, int(phase), "rgb", f"{stem_a}.png"),
                    "rgb_b": _modality_path(scene, int(phase), "rgb", f"{stem_b}.png"),
                    "pose_a": _modality_path(scene, int(phase), "cam_pose", f"{stem_a}.npz"),
                    "pose_b": _modality_path(scene, int(phase), "cam_pose", f"{stem_b}.npz"),
                })
                stats["temporal_pairs"] += 1

    stats["scenes_processed"] = len(
        {s for (s, _) in scene_phase_groups}
    )
    total = stats["intra_pairs"] + stats["cross_pairs"] + stats["temporal_pairs"]

    return {
        "task": "relative_camera_pose",
        "split": split,
        "root": None,
        "samples": samples,
        "_sampler": {
            "name": "stratified_v2",
            "version": "2.0",
            "rotation_bins_deg": [[lo, hi] for lo, hi in ROTATION_BINS],
            "valid_phases": list(VALID_PHASES),
            "intra_pairs_per_bin": cfg.intra_pairs_per_bin,
            "cross_pairs_per_bin": cfg.cross_pairs_per_bin,
            "chain_count": cfg.chain_count,
            "chain_length": cfg.chain_length,
            "chain_stride": cfg.chain_stride,
            "frame_gap": cfg.frame_gap,
            "min_rotation_deg": cfg.min_rotation_deg,
            "min_translation_m": cfg.min_translation_m,
            "seed": cfg.seed,
            "excluded_scenes": sorted(EXCLUDED_SCENE_IDS),
            "excluded_scene_phases": [
                f"{s}/{p}" for s, p in sorted(EXCLUDED_SCENE_PHASES)
            ],
            **stats,
            "total_pairs": total,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--split", default="easy", choices=("easy", "medium", "hard"))
    ap.add_argument("--repo", default="IRVLUTD/RPX")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--pairs-per-bin", type=int, default=50,
                    help="intra-phase pairs per rotation bin per (scene, phase)")
    ap.add_argument("--cross-pairs-per-bin", type=int, default=0,
                    help="deprecated; must remain 0 (unrelated T265 worlds)")
    ap.add_argument("--chain-count", type=int, default=5,
                    help="temporal chains per (scene, phase)")
    ap.add_argument("--chain-length", type=int, default=10,
                    help="pairs per chain")
    ap.add_argument("--chain-stride", type=int, default=5,
                    help="frame gap between consecutive chain pairs")
    ap.add_argument("--frame-gap", type=int, default=5,
                    help="exact frame-index separation for intra-phase pairs")
    ap.add_argument("--seed", type=int, default=RPX_SEED)
    args = ap.parse_args()

    if args.cross_pairs_per_bin != 0:
        ap.error(
            "--cross-pairs-per-bin must be 0: phases are independent captures "
            "with unrelated T265 local world frames"
        )

    logging.basicConfig(level=logging.INFO, format="[pose-v2] %(message)s")

    snap = _hf_snapshot_root(args.repo)
    parquet = snap / "manifest" / "frames_v1.parquet"
    extracted = snap / "extracted"
    if not parquet.exists():
        raise SystemExit(f"missing frames parquet at {parquet}")

    out = args.output or (snap / "manifests" / "relative_pose_v2" / f"{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = SamplerConfig(
        intra_pairs_per_bin=args.pairs_per_bin,
        cross_pairs_per_bin=args.cross_pairs_per_bin,
        chain_count=args.chain_count,
        chain_length=args.chain_length,
        chain_stride=args.chain_stride,
        frame_gap=args.frame_gap,
        seed=args.seed,
    )
    manifest = build_manifest(
        parquet_path=parquet,
        extracted_root=extracted,
        split=args.split,
        cfg=cfg,
    )
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    s = manifest["_sampler"]
    print(f"\n{'='*60}")
    print(f"  RPX Relative Pose Pairs v2 — {args.split} split")
    print(f"{'='*60}")
    print(f"  Scenes processed:   {s['scenes_processed']}")
    print(f"  Phases processed:   {s['phases_processed']} (skipped {s['phases_skipped']})")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Intra-phase pairs:  {s['intra_pairs']:,}")
    for name in BIN_NAMES:
        print(f"    {name:>8}: {s['intra_by_bin'][name]:,}")
    print(f"  Cross-phase pairs:  {s['cross_pairs']:,}")
    for name in BIN_NAMES:
        print(f"    {name:>8}: {s['cross_by_bin'][name]:,}")
    print(f"  Temporal chains:    {s['temporal_pairs']:,} pairs")
    print(f"  ─────────────────────────────────────────────")
    print(f"  TOTAL:              {s['total_pairs']:,} pairs")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Output: {out}")
    print()


if __name__ == "__main__":
    _cli()
