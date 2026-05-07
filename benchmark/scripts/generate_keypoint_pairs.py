#!/usr/bin/env python
"""Precompute ground-truth keypoint correspondences.

For each (frame A, frame B) pair picked by the shared relative-pose
sampler:

  1. Sample N valid depth pixels in frame A.
  2. Back-project to 3D using camera intrinsics + depth_A.
  3. Transform to frame B via the ground-truth relative pose.
  4. Re-project into frame B's image plane.
  5. Flag visibility for in-bounds and occlusion-consistent points.

All ``--num-points`` points are kept in the output arrays (padded if
fewer survive), so the per-pair tensors have a fixed shape. The
``visibility`` mask is the GT for matcher evaluation.

Intrinsics default to *approximate* D435 RGB 640x480 values. Provide
real factory calibration via ``--intrinsics intrinsics.json`` for
publication-quality correspondences::

    {"fx": 605.12, "fy": 605.00, "cx": 320.1, "cy": 240.3}

Usage::

    python scripts/generate_keypoint_pairs.py \
        --local-root /data/rpx \
        --intrinsics calibration/d435.json \
        --num-points 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

# Re-use pair sampler + pose helpers from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_pair_task_manifests as _pairs  # noqa: E402

DEFAULT_INTRINSICS = {"fx": 605.0, "fy": 605.0, "cx": 320.0, "cy": 240.0}


def load_intrinsics(path: Path | None) -> Dict[str, float]:
    if path is None:
        print("[warn] --intrinsics not provided; falling back to approximate "
              "D435 640x480 defaults. Not suitable for publication-quality "
              "evaluation.", file=sys.stderr)
        return dict(DEFAULT_INTRINSICS)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(data[k]) for k in ("fx", "fy", "cx", "cy")}


def load_depth_metres(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        mm = np.array(im, dtype=np.float32)
    metres = mm / 1000.0
    metres[mm == 0] = 0.0
    return metres


def backproject(
    coords_xy: np.ndarray,
    depths: np.ndarray,
    K: Dict[str, float],
) -> np.ndarray:
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    X = (x - K["cx"]) * depths / K["fx"]
    Y = (y - K["cy"]) * depths / K["fy"]
    return np.stack([X, Y, depths], axis=1)


def project(
    points_3d: np.ndarray,
    K: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    safe_Z = np.maximum(Z, 1e-6)
    u = K["fx"] * X / safe_Z + K["cx"]
    v = K["fy"] * Y / safe_Z + K["cy"]
    return np.stack([u, v], axis=1), Z


def transform_points(points_3d: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points_3d @ R.T + t


def _frame_seed(base: int, scene: str, phase: str, fa: str, fb: str) -> int:
    h = hashlib.blake2s(f"{scene}/{phase}/{fa}/{fb}".encode(), digest_size=4).digest()
    return (base ^ int.from_bytes(h, "big")) & 0x7FFFFFFF


def sample_pair_keypoints(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    K: Dict[str, float],
    num_points: int,
    rng: np.random.Generator,
    occlusion_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = depth_a.shape
    ys, xs = np.where(depth_a > 0)
    if len(ys) == 0:
        return (np.zeros((num_points, 2), np.float32),
                np.zeros((num_points, 2), np.float32),
                np.zeros((num_points,), bool))

    k = min(num_points, len(ys))
    idx = rng.choice(len(ys), size=k, replace=False)
    coords_a = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    depths_a = depth_a[ys[idx], xs[idx]].astype(np.float32)

    pts3d_a = backproject(coords_a, depths_a, K)
    # Transform from camera A (via world) into camera B.
    T_b_a = np.linalg.inv(pose_b) @ pose_a
    pts3d_b = transform_points(pts3d_a, T_b_a)
    coords_b, depths_reproj = project(pts3d_b, K)

    in_bounds = (
        (coords_b[:, 0] >= 0) & (coords_b[:, 0] < W) &
        (coords_b[:, 1] >= 0) & (coords_b[:, 1] < H) &
        (depths_reproj > 0)
    )
    visibility = in_bounds.copy()

    # Occlusion test: projected depth must match observed depth in B.
    u = np.clip(coords_b[:, 0].astype(np.int32), 0, W - 1)
    v = np.clip(coords_b[:, 1].astype(np.int32), 0, H - 1)
    observed = depth_b[v, u]
    depth_diff = np.abs(observed - depths_reproj)
    occluded = (observed > 0) & (depth_diff > occlusion_threshold)
    visibility &= ~occluded
    visibility &= (observed > 0)

    def _pad(a: np.ndarray, shape: Tuple[int, ...], dtype) -> np.ndarray:
        out = np.zeros(shape, dtype=dtype)
        out[: len(a)] = a
        return out

    return (_pad(coords_a, (num_points, 2), np.float32),
            _pad(coords_b.astype(np.float32), (num_points, 2), np.float32),
            _pad(visibility, (num_points,), bool))


def main() -> int:
    try:
        from rpx_benchmark.banner import show_banner
        show_banner(subtitle="scripts/generate_keypoint_pairs.py")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--num-points", type=int, default=512)
    # Default to the project-wide canonical seed (MMDDYYYY 05/06/2026).
    from rpx_benchmark.determinism import RPX_SEED
    parser.add_argument("--seed", type=int, default=RPX_SEED)
    parser.add_argument("--occlusion-threshold", type=float, default=0.05,
                        help="Max |d_reproj - d_observed| (m) still counted visible")
    parser.add_argument("--min-t", type=float, default=0.05)
    parser.add_argument("--max-t", type=float, default=0.60)
    parser.add_argument("--min-r", type=float, default=2.0)
    parser.add_argument("--max-r", type=float, default=35.0)
    parser.add_argument("--src-stride", type=int, default=10)
    args = parser.parse_args()

    local_root: Path = args.local_root
    scenes_root = local_root / "scenes"
    if not scenes_root.is_dir():
        print(f"error: {scenes_root} missing", file=sys.stderr)
        return 2
    esd_path = local_root / "esd_scores.json"
    if not esd_path.exists():
        print(f"error: {esd_path} missing", file=sys.stderr)
        return 2

    K = load_intrinsics(args.intrinsics)
    cfg = _pairs.PairConfig(
        min_t=args.min_t, max_t=args.max_t,
        min_r=args.min_r, max_r=args.max_r,
        src_stride=args.src_stride,
    )

    sequences = _pairs.discover_pose_sequences(scenes_root)
    esd = _pairs.load_esd_scores(esd_path)
    print(f"discovered {len(sequences)} pose sequences")

    manifests: Dict[str, Dict[str, Any]] = {
        d: {"task": "keypoint_matching", "split": d, "scenes": [], "samples": []}
        for d in ("easy", "medium", "hard")
    }
    pair_count = 0

    for seq in sequences:
        scene, phase = seq["scene"], seq["phase"]
        esd_entry = esd.get((scene, phase))
        if esd_entry is None:
            continue
        difficulty = esd_entry["difficulty"]
        if difficulty not in manifests:
            continue

        phase_dir = scenes_root / scene / phase
        depth_dir = phase_dir / "depth"
        if not depth_dir.is_dir():
            continue

        frame_ids = seq["frame_ids"]
        poses = [_pairs.load_pose_npz(phase_dir / "pose" / f"{fid}.npz")
                 for fid in frame_ids]
        pairs = _pairs.sample_pairs(poses, cfg)
        if not pairs:
            continue

        out_dir = phase_dir / "keypoints"
        out_dir.mkdir(exist_ok=True)
        phase_name = seq["phase_name"]
        manifests[difficulty]["scenes"].append({"scene": scene, "phase": phase})

        for i, j in pairs:
            fa, fb = frame_ids[i], frame_ids[j]
            depth_a_path = depth_dir / f"{fa}.png"
            depth_b_path = depth_dir / f"{fb}.png"
            if not (depth_a_path.exists() and depth_b_path.exists()):
                continue
            depth_a = load_depth_metres(depth_a_path)
            depth_b = load_depth_metres(depth_b_path)

            rng = np.random.default_rng(_frame_seed(args.seed, scene, phase, fa, fb))
            p0, p1, vis = sample_pair_keypoints(
                depth_a, depth_b, poses[i], poses[j], K,
                args.num_points, rng, args.occlusion_threshold,
            )

            base = f"{fa}_{fb}"
            np.save(out_dir / f"{base}_points0.npy", p0)
            np.save(out_dir / f"{base}_points1.npy", p1)
            np.save(out_dir / f"{base}_visibility.npy", vis)

            manifests[difficulty]["samples"].append({
                "id": f"{scene}_{phase_name}_{base}",
                "scene": scene,
                "phase": phase_name,
                "difficulty": difficulty,
                "rgb":        f"scenes/{scene}/{phase}/rgb/{fa}.png",
                "rgb_b":      f"scenes/{scene}/{phase}/rgb/{fb}.png",
                "points0":    f"scenes/{scene}/{phase}/keypoints/{base}_points0.npy",
                "points1":    f"scenes/{scene}/{phase}/keypoints/{base}_points1.npy",
                "visibility": f"scenes/{scene}/{phase}/keypoints/{base}_visibility.npy",
            })
            pair_count += 1

    out_root = local_root / "manifests" / "keypoint_matching"
    out_root.mkdir(parents=True, exist_ok=True)
    for diff, m in manifests.items():
        if not m["samples"]:
            continue
        with (out_root / f"{diff}.json").open("w", encoding="utf-8") as f:
            json.dump(m, f)
        print(f"  keypoint_matching  {diff:<6} {len(m['samples'])} pairs "
              f"({len(m['scenes'])} phases)")
    print(f"total: {pair_count} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
