#!/usr/bin/env python3
"""
Minimal per-subdir, order-preserving Poisson-disk sampling over *pose distance* with Numba.

Pose distance:
    d = sqrt( ||Δt||^2 + (lambda_rot * theta)^2 )
where theta is the geodesic angle (radians) between unit quaternions [x,y,z,w].

Input layout:
    <scene_dir>/<subdir>/<pose_subdir>/*.npz
Each .npz must contain:
    position    -> [x, y, z] (meters)
    orientation -> [qx, qy, qz, qw] (xyzw)

CLI examples:
    # Fixed radius (meters in pose distance)
    python pds_pose_per_subdir_min_numba.py --scene-dir ./scene --r 0.25 --lambda-rot 0.3

    # Exactly K per subdir (solve radius via bisection)
    python pds_pose_per_subdir_min_numba.py --scene-dir ./scene --k 100 --lambda-rot 0.3
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
import math

# ---------------- Numba kernel: batched pose distances ----------------

@njit(cache=True)
def _pose_dists_to_neighbors(x_i, q_i, neigh_xyz, neigh_q, lambda_rot):
    """
    Compute pose distances from (x_i, q_i) to a batch of neighbors.
    All arrays are float64 and quaternions are (approximately) unit-norm.
    Returns 1D array of distances (meters).
    """
    m = neigh_xyz.shape[0]
    out = np.empty(m, dtype=np.float64)
    for k in range(m):
        # translation
        dx0 = x_i[0] - neigh_xyz[k, 0]
        dx1 = x_i[1] - neigh_xyz[k, 1]
        dx2 = x_i[2] - neigh_xyz[k, 2]
        dt = math.sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2)
        # rotation (geodesic)
        dot = q_i[0]*neigh_q[k,0] + q_i[1]*neigh_q[k,1] + q_i[2]*neigh_q[k,2] + q_i[3]*neigh_q[k,3]
        if dot < 0.0: dot = -dot
        if dot > 1.0: dot = 1.0
        theta = 2.0 * math.acos(dot)
        lr = lambda_rot * theta
        out[k] = math.sqrt(dt*dt + lr*lr)
    return out

# ---------------- Simple IO ----------------

def _load_xyz_q(files):
    xyz, quat = [], []
    for p in files:
        d = np.load(p)
        pos = np.asarray(d["position"], np.float64).reshape(3)
        q = np.asarray(d["orientation"], np.float64).reshape(4)
        n = np.linalg.norm(q)
        q = q / n if n > 0 else np.array([0,0,0,1], dtype=np.float64)
        xyz.append(pos); quat.append(q)
    return np.vstack(xyz), np.vstack(quat)

# ---------------- Order-preserving PDS (KD-tree + Numba) ----------------

def _pds_order_preserving_numba(xyz: np.ndarray, quat: np.ndarray, r: float, lambda_rot: float) -> list[int]:
    """
    Forward scan; keep first & last; accept i if pose distance >= r to ALL kept.
    Uses a KD-tree on kept XYZ to prune, and Numba kernel for the neighbor checks.
    """
    N = len(xyz)
    if N == 0: return []
    if N == 1: return [0]

    kept = [0]
    kept_xyz = [xyz[0]]
    tree = cKDTree(np.vstack(kept_xyz))

    for i in range(1, N-1):
        # quick NN in translation
        d_nn, _ = tree.query(xyz[i], k=1)
        if d_nn >= r:
            kept.append(i)
            kept_xyz.append(xyz[i])
            tree = cKDTree(np.vstack(kept_xyz))
            continue

        # check only neighbors within r in translation using full pose distance
        cand_idx = tree.query_ball_point(xyz[i], r)
        ok = True
        if cand_idx:
            neigh_xyz = np.vstack([kept_xyz[j] for j in cand_idx])
            neigh_q   = np.vstack([quat[kept[j]] for j in cand_idx])
            d_pose = _pose_dists_to_neighbors(xyz[i], quat[i], neigh_xyz, neigh_q, lambda_rot)
            if np.any(d_pose < r):
                ok = False

        if ok:
            kept.append(i)
            kept_xyz.append(xyz[i])
            tree = cKDTree(np.vstack(kept_xyz))

    if kept[-1] != N-1:
        kept.append(N-1)
    return kept

def _bisection_for_k_numba(xyz: np.ndarray, quat: np.ndarray, k: int, lambda_rot: float,
                           r_min: float = 0.01, r_max: float = 3.0, iters: int = 25) -> list[int]:
    """Find r so PDS returns ~k samples. Keeps endpoints; trims interior if overshoot."""
    N = len(xyz); k = int(np.clip(k, 2, max(2, N)))
    best = None; lo, hi = r_min, r_max
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        sel = _pds_order_preserving_numba(xyz, quat, mid, lambda_rot)
        if len(sel) >= k:
            best = sel; lo = mid
        else:
            hi = mid
    if best is None:
        best = _pds_order_preserving_numba(xyz, quat, r_min, lambda_rot)
    if len(best) > k:
        core = best[1:-1]
        best = [best[0]] + core[:k-2] + [best[-1]]
    return best

# ---------------- Public API ----------------

def run_per_subdir(scene_dir: str,
                   pose_subdir: str = "cam_pose",
                   splits=("0","1","2"),
                   r: float | None = None,
                   k: int | None = None,
                   lambda_rot: float = 0.3) -> dict[str, list[str]]:
    """
    Return dict{subdir: [selected pose filenames]} with order preserved & endpoints kept.
    Choose either fixed radius r (meters) or target count k (per subdir).
    """
    assert (r is None) ^ (k is None), "Specify exactly one of r or k."
    out = {}
    for s in splits:
        pose_dir = Path(scene_dir) / s / pose_subdir
        files = sorted(pose_dir.glob("*.npz"))
        if not files:
            out[str(s)] = []
            continue
        xyz, quat = _load_xyz_q(files)
        sel = _bisection_for_k_numba(xyz, quat, k, lambda_rot) if k is not None \
              else _pds_order_preserving_numba(xyz, quat, r, lambda_rot)
        out[str(s)] = [str(files[i]) for i in sel]
    return out

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Subdir-wise PDS filenames via pose distance (Numba, order-preserving).")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--pose-subdir", default="cam_pose")
    ap.add_argument("--splits", default="0,1,2")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--r", type=float, help="Poisson radius in meters (pose distance).")
    mode.add_argument("--k", type=int, help="Target samples per subdir (solve radius via bisection).")
    ap.add_argument("--lambda-rot", type=float, default=0.3, help="Meters per radian; rotation weight.")
    args = ap.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    results = run_per_subdir(args.scene_dir, args.pose_subdir, splits, r=args.r, k=args.k, lambda_rot=args.lambda_rot)

    # Print neat subdir-wise lists
    for s in splits:
        print(f"\n# Subdir {s}")
        for f in results.get(str(s), []):
            print(f)

if __name__ == "__main__":
    main()
