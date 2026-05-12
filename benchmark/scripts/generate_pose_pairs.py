"""Poisson-disk pair sampler for the RPX relative-pose split.

The default split-manifest writer (``_RelativePoseSpec`` in
``rpx_benchmark/dataset_hub/split_manifests.py``) pairs consecutive
frames at a fixed stride (5). That gives trivially short baselines:
empirically, > 95 % of pairs have < 5 ° camera motion and centimeter-
scale translation, so the GT translation direction is noise-dominated
and pose-AUC collapses to zero across every adapter.

This script replaces stride-5 with a **discrete Poisson-disk** selection
over the (rotation, translation) plane. Within each (scene, phase):

1. enumerate every candidate pair ``(i, j)``, ``j > i``;
2. embed it as ``(rot_deg, t_m × SCALE_DEG_PER_M)`` — a 2-D point;
3. seeded-shuffle the candidates;
4. greedily accept a candidate iff no previously-accepted pair in the
   same (scene, phase) lies within radius ``r`` in that 2-D embedding;
5. stop at the per-(scene, phase) cap, or when no candidates remain.

The result is a manifest matching the canonical schema at
``<snap>/manifests/relative_pose/<split>.json`` — drop-in compatible
with ``run_relative_pose.py --pairs-manifest <path>``.

Usage
-----
    PYTHONPATH=. python scripts/generate_pose_pairs.py \\
        --split easy \\
        --output <snap>/manifests/relative_pose_poisson/easy.json \\
        --radius-deg 2.0 \\
        --scale-deg-per-m 100.0 \\
        --pairs-per-phase 253 \\
        --max-frame-gap 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_manifest import _hf_snapshot_root  # noqa: E402

RPX_SEED = 5_062_026
log = logging.getLogger(__name__)

# ─────────────────────────  exclusions  ──────────────────────────────────
# Poses were optimized via ICP using sequences in object-corr-out-GT-masks.
# scene58 is entirely missing from that source, so no optimized poses exist.
# The following (scene, phase) pairs failed ICP optimization and must be
# excluded from pair generation.

EXCLUDED_SCENE_IDS: set[str] = {"scene58"}
"""Scenes excluded entirely (no optimized poses available)."""

EXCLUDED_SCENE_PHASES: set[tuple[str, int]] = {
    # Task 0 failures
    ("scene83.jsom.garden.pot", 0),
    ("scene98.jsom.atrium", 0),
    ("scene100.jsom.atrium", 0),
    # Task 2 failures
    ("scene50.ecss.out.stairs", 2),
    ("scene72.ecsw.atriumStairs", 2),
}
"""(scene_id, phase) pairs where ICP optimization failed."""


# ─────────────────────────  pose math  ───────────────────────────────────


def _quat_to_rot(quat_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3×3 rotation matrix. Matches the convention in
    rpx_benchmark.loader._load_pose."""
    x, y, z, w = quat_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_pose(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_world_cam, t_world_cam) from a cam_pose .npz."""
    data = np.load(npz_path)
    t = np.asarray(data["position"], dtype=np.float64).reshape(3)
    q = np.asarray(data["orientation"], dtype=np.float64).reshape(4)
    return _quat_to_rot(q), t


def _relative_rotation_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotations."""
    R_rel = R_a.T @ R_b
    cos_theta = (np.trace(R_rel) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, float(cos_theta)))
    return float(np.degrees(np.arccos(cos_theta)))


# ─────────────────────────  Poisson disk  ─────────────────────────────────


def poisson_disk_select(
    points: np.ndarray,
    radius: float,
    *,
    max_keep: int | None = None,
    seed: int = RPX_SEED,
) -> np.ndarray:
    """Return indices selected by discrete Poisson-disk sampling.

    Greedy O(N · K) where K is the kept count. We grid-bin the 2-D
    embedding so neighbour lookup is O(1) amortised.

    Parameters
    ----------
    points : (N, 2) array
        2-D embedding of each candidate.
    radius : float
        Minimum distance between any two kept points.
    max_keep : int, optional
        Stop once this many points have been kept. ``None`` → no cap.
    seed : int
        Determines the shuffle order. Pin to ``RPX_SEED`` for reproducibility.

    Returns
    -------
    np.ndarray
        Indices into ``points`` of the kept items, in selection order.
    """
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    n = points.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    # Cell size = radius. With this choice, any two points whose grid
    # cells differ by ≥ 2 in either axis are guaranteed ≥ radius apart,
    # so a 3×3 neighbourhood lookup catches every possible conflict.
    cell = radius
    grid: dict[tuple[int, int], list[int]] = {}

    kept: list[int] = []
    r2 = radius * radius
    for idx in order:
        if max_keep is not None and len(kept) >= max_keep:
            break
        p = points[idx]
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        too_close = False
        for dx in (-1, 0, 1):
            if too_close:
                break
            for dy in (-1, 0, 1):
                neighbours = grid.get((gx + dx, gy + dy))
                if not neighbours:
                    continue
                for k in neighbours:
                    diff = points[k] - p
                    if diff[0] * diff[0] + diff[1] * diff[1] < r2:
                        too_close = True
                        break
                if too_close:
                    break
        if too_close:
            continue
        kept.append(int(idx))
        grid.setdefault((gx, gy), []).append(int(idx))
    return np.asarray(kept, dtype=np.int64)


# ─────────────────────────  pair generation  ─────────────────────────────


@dataclass
class SamplerConfig:
    radius_deg: float = 2.0
    scale_deg_per_m: float = 100.0
    pairs_per_phase: int | None = 253
    max_frame_gap: int = 200
    min_rotation_deg: float = 0.1
    min_translation_m: float = 0.005
    seed: int = RPX_SEED


def _candidate_pairs_2d(
    poses: list[tuple[np.ndarray, np.ndarray]],
    cfg: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate every pair (i, j) with j > i within ``max_frame_gap``.

    Returns ``(pair_indices, points_2d)``:
    - ``pair_indices`` : (M, 2) int — (i, j) per row
    - ``points_2d``    : (M, 2) float — (rot_deg, t_m × scale)
    Pairs with sub-threshold motion are filtered out (they make poor
    eval pairs — no signal to evaluate against centimeter-scale GT).
    """
    n = len(poses)
    pairs: list[tuple[int, int]] = []
    pts: list[tuple[float, float]] = []
    for i in range(n):
        R_i, t_i = poses[i]
        upper = min(n, i + cfg.max_frame_gap + 1)
        for j in range(i + 1, upper):
            R_j, t_j = poses[j]
            theta = _relative_rotation_deg(R_i, R_j)
            t_m = float(np.linalg.norm(t_j - t_i))
            if theta < cfg.min_rotation_deg and t_m < cfg.min_translation_m:
                continue
            pairs.append((i, j))
            pts.append((theta, t_m * cfg.scale_deg_per_m))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(pairs, dtype=np.int64), np.asarray(pts, dtype=np.float64)


def select_pairs_for_phase(
    frame_idxs: list[int],
    poses: list[tuple[np.ndarray, np.ndarray]],
    cfg: SamplerConfig,
    seed_offset: int = 0,
) -> list[tuple[int, int, float, float]]:
    """Sample pairs for one (scene, phase). Returns a list of
    ``(frame_idx_a, frame_idx_b, rot_deg, t_m)`` tuples."""
    pair_idx, pts = _candidate_pairs_2d(poses, cfg)
    if pair_idx.shape[0] == 0:
        return []
    kept = poisson_disk_select(
        pts,
        cfg.radius_deg,
        max_keep=cfg.pairs_per_phase,
        seed=cfg.seed + seed_offset,
    )
    out: list[tuple[int, int, float, float]] = []
    for k in kept:
        i, j = int(pair_idx[k, 0]), int(pair_idx[k, 1])
        rot_deg = float(pts[k, 0])
        t_scaled = float(pts[k, 1])
        t_m = t_scaled / cfg.scale_deg_per_m
        out.append((frame_idxs[i], frame_idxs[j], rot_deg, t_m))
    return out


# ─────────────────────────  manifest writer  ─────────────────────────────


def _modality_path(scene: str, phase: int, modality: str, stem: str) -> str:
    """Match the path layout used in the canonical manifest."""
    return f"extracted/scenes/{scene}/{phase}/{modality}/{stem}"


def build_manifest(
    *,
    parquet_path: Path,
    extracted_root: Path,
    split: str,
    cfg: SamplerConfig,
) -> dict:
    df = pd.read_parquet(parquet_path)
    # Mirror the loader's scene-wise split assignment: every phase of a
    # scene shares one tier (avoids cross-phase tier mixing).
    scene_split = df.groupby("scene_id")["split"].agg(lambda s: s.value_counts().idxmax())
    df = df.drop(columns=["split"]).merge(
        scene_split.rename("split"), left_on="scene_id", right_index=True
    )
    df = df[(df["split"].astype(str) == split) & df["has_cam_pose"].fillna(False).astype(bool)]
    if df.empty:
        raise SystemExit(f"no frames with cam_pose for split={split!r} in {parquet_path}")

    samples: list[dict] = []
    n_candidates_total = 0
    n_excluded = 0
    for seed_offset, ((scene, phase), grp) in enumerate(
        df.groupby(["scene_id", "phase"], sort=True)
    ):
        # Skip scenes/phases with missing or failed ICP-optimized poses.
        if scene in EXCLUDED_SCENE_IDS:
            n_excluded += len(grp)
            log.info("SKIP (%s, %d): scene entirely excluded (no optimized poses)", scene, phase)
            continue
        if (scene, int(phase)) in EXCLUDED_SCENE_PHASES:
            n_excluded += len(grp)
            log.info("SKIP (%s, %d): ICP optimization failed for this phase", scene, phase)
            continue

        grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
        frame_idxs = [int(r["frame_idx"]) for _, r in grp_sorted.iterrows()]
        # Load every pose for this (scene, phase) once.
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        for _, r in grp_sorted.iterrows():
            stem = str(r["frame_filename"]).rsplit(".", 1)[0]
            poses.append(
                _load_pose(
                    extracted_root / "scenes" / scene / str(phase) / "cam_pose" / f"{stem}.npz"
                )
            )

        n_candidates_total += len(poses) * (len(poses) - 1) // 2
        selected = select_pairs_for_phase(frame_idxs, poses, cfg, seed_offset=seed_offset)

        rot_buf: list[float] = []
        t_buf: list[float] = []
        for fa, fb, rot_deg, t_m in selected:
            stem_a, stem_b = f"{fa:05d}", f"{fb:05d}"
            samples.append(
                {
                    "id": f"{scene}__{phase}__{stem_a}__{stem_b}",
                    "scene_id": scene,
                    "phase": int(phase),
                    "frame_idx": fa,
                    "frame_idx_b": fb,
                    "difficulty": split,
                    "metadata": {
                        "scene_id": scene,
                        "phase_idx": int(phase),
                        "frame": stem_a,
                        "frame_b": stem_b,
                        "pair_stride": fb - fa,  # variable now, kept for back-compat
                        "rotation_deg_gt": rot_deg,
                        "translation_m_gt": t_m,
                        "sampler": "poisson_disk",
                        "sampler_radius_deg": cfg.radius_deg,
                        "sampler_scale_deg_per_m": cfg.scale_deg_per_m,
                    },
                    "rgb": _modality_path(scene, int(phase), "rgb", f"{stem_a}.png"),
                    "rgb_b": _modality_path(scene, int(phase), "rgb", f"{stem_b}.png"),
                    "pose_a": _modality_path(scene, int(phase), "cam_pose", f"{stem_a}.npz"),
                    "pose_b": _modality_path(scene, int(phase), "cam_pose", f"{stem_b}.npz"),
                }
            )
            rot_buf.append(rot_deg)
            t_buf.append(t_m)
        log.info(
            "(%s, %d): %d frames → %d candidates → %d kept "
            "(rot µ=%.1f° σ=%.1f° | t µ=%.2f m σ=%.2f m)",
            scene,
            phase,
            len(poses),
            len(poses) * (len(poses) - 1) // 2,
            len(selected),
            float(np.mean(rot_buf)) if rot_buf else 0.0,
            float(np.std(rot_buf)) if rot_buf else 0.0,
            float(np.mean(t_buf)) if t_buf else 0.0,
            float(np.std(t_buf)) if t_buf else 0.0,
        )

    return {
        "task": "relative_camera_pose",
        "split": split,
        "root": None,
        "samples": samples,
        "_sampler": {
            "name": "poisson_disk",
            "radius_deg": cfg.radius_deg,
            "scale_deg_per_m": cfg.scale_deg_per_m,
            "pairs_per_phase": cfg.pairs_per_phase,
            "max_frame_gap": cfg.max_frame_gap,
            "min_rotation_deg": cfg.min_rotation_deg,
            "min_translation_m": cfg.min_translation_m,
            "seed": cfg.seed,
            "n_candidates_total": n_candidates_total,
            "n_kept": len(samples),
            "n_frames_excluded": n_excluded,
            "excluded_scenes": sorted(EXCLUDED_SCENE_IDS),
            "excluded_scene_phases": [
                f"{s}/{p}" for s, p in sorted(EXCLUDED_SCENE_PHASES)
            ],
        },
    }


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--split", default="easy", choices=("easy", "medium", "hard"))
    ap.add_argument("--repo", default="itaykadosh/rpx-test")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="manifest path (default: <snap>/manifests/relative_pose_poisson/<split>.json)",
    )
    ap.add_argument(
        "--radius-deg",
        type=float,
        default=2.0,
        help="minimum 2-D distance between kept pairs (default: 2.0)",
    )
    ap.add_argument(
        "--scale-deg-per-m",
        type=float,
        default=100.0,
        help="meters → equivalent degrees scaling for the 2-D embedding "
        "(default: 100, i.e. 10 cm baseline ≈ 10°)",
    )
    ap.add_argument(
        "--pairs-per-phase",
        type=int,
        default=253,
        help="hard cap per (scene, phase) — set 0 to disable",
    )
    ap.add_argument(
        "--max-frame-gap",
        type=int,
        default=200,
        help="don't consider pairs more than this many frames apart",
    )
    ap.add_argument("--min-rotation-deg", type=float, default=0.1)
    ap.add_argument("--min-translation-m", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=RPX_SEED)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[poisson] %(message)s")

    snap = _hf_snapshot_root(args.repo)
    parquet = snap / "manifest" / "frames_v1.parquet"
    extracted = snap / "extracted"
    if not parquet.exists():
        raise SystemExit(
            f"missing frames parquet at {parquet} — populate the HF cache first "
            "(e.g. `python -m rpx_benchmark.dataset_hub.cli manifest --tasks relative_pose`)."
        )
    out = args.output or (snap / "manifests" / "relative_pose_poisson" / f"{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = SamplerConfig(
        radius_deg=args.radius_deg,
        scale_deg_per_m=args.scale_deg_per_m,
        pairs_per_phase=args.pairs_per_phase if args.pairs_per_phase > 0 else None,
        max_frame_gap=args.max_frame_gap,
        min_rotation_deg=args.min_rotation_deg,
        min_translation_m=args.min_translation_m,
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
    print(f"[poisson] wrote {out}")
    print(
        f"[poisson] kept {manifest['_sampler']['n_kept']} pairs "
        f"out of {manifest['_sampler']['n_candidates_total']} candidates "
        f"(r={cfg.radius_deg}°, scale={cfg.scale_deg_per_m} °/m, "
        f"cap/phase={cfg.pairs_per_phase})"
    )


if __name__ == "__main__":
    _cli()
