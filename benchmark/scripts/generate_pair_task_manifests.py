#!/usr/bin/env python
"""Generate manifests for pair-based RPX tasks.

Covers:
  * ``relative_camera_pose`` — frame pairs with a meaningful baseline
  * ``novel_view_synthesis`` — (source, target) frame pairs

Both walk per-(scene, phase) pose streams and sample pairs whose relative
transform satisfies configurable translation/rotation bounds, so models
are not evaluated on nearly-identical frames. Determinism is preserved:
same inputs + flags always produce the same pair list (no randomness in
the default sampler).

Sparse-depth and keypoint-matching pair precomputation are intentionally
not handled here; they require per-frame point sampling and projection
passes and will live in a separate script.

Usage::

    python scripts/generate_pair_task_manifests.py \\
        --local-root /data/rpx \\
        --tasks relative_camera_pose novel_view_synthesis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PHASE_NAMES = {"0": "clutter", "1": "interaction", "2": "clean"}
PAIR_TASKS = ("relative_camera_pose", "novel_view_synthesis")
CANDIDATE_OFFSETS = (5, 10, 15, 20, 30, 45)


# ------------------------------------------------------------------ #
# Pose helpers (duplicated from loader to keep this script standalone)
# ------------------------------------------------------------------ #

def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def load_pose_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    position = data["position"].astype(np.float64)
    quat = data["orientation"].astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(quat)
    T[:3, 3] = position
    return T


def rot_angle_deg(R: np.ndarray) -> float:
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def relative_transform(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_a) @ T_b


# ------------------------------------------------------------------ #
# Pair sampling
# ------------------------------------------------------------------ #

@dataclass
class PairConfig:
    min_t: float = 0.05     # metres
    max_t: float = 0.60
    min_r: float = 2.0      # degrees
    max_r: float = 35.0
    src_stride: int = 10    # every Nth frame is a candidate source


def sample_pairs(
    poses: List[np.ndarray],
    cfg: PairConfig,
) -> List[Tuple[int, int]]:
    """Deterministic pair sampling over an ordered pose sequence.

    For each source frame at index ``i`` (stepped by ``src_stride``), walks
    ``CANDIDATE_OFFSETS`` until it finds a target whose relative transform
    is within the translation/rotation window. First match wins.
    """
    pairs: List[Tuple[int, int]] = []
    n = len(poses)
    for i in range(0, n, cfg.src_stride):
        for dj in CANDIDATE_OFFSETS:
            j = i + dj
            if j >= n:
                break
            T_rel = relative_transform(poses[i], poses[j])
            t_mag = float(np.linalg.norm(T_rel[:3, 3]))
            r_deg = rot_angle_deg(T_rel[:3, :3])
            if cfg.min_t <= t_mag <= cfg.max_t and cfg.min_r <= r_deg <= cfg.max_r:
                pairs.append((i, j))
                break
    return pairs


# ------------------------------------------------------------------ #
# Disk scanning
# ------------------------------------------------------------------ #

def discover_pose_sequences(scenes_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scene_dir in sorted(scenes_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for phase in ("0", "1", "2"):
            pose_dir = scene_dir / phase / "pose"
            rgb_dir = scene_dir / phase / "rgb"
            if not pose_dir.is_dir() or not rgb_dir.is_dir():
                continue
            pose_files = sorted(p for p in pose_dir.iterdir()
                                if p.suffix == ".npz")
            frame_ids = [p.stem for p in pose_files]
            if len(frame_ids) < max(CANDIDATE_OFFSETS) + 1:
                continue
            out.append({
                "scene": scene_dir.name,
                "phase": phase,
                "phase_name": PHASE_NAMES.get(phase, phase),
                "frame_ids": frame_ids,
            })
    return out


def load_esd_scores(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {(str(e["scene"]), str(e["phase"])): e for e in data}


# ------------------------------------------------------------------ #
# Manifest assembly
# ------------------------------------------------------------------ #

def _rel(scene: str, phase: str, sub: str, frame: str, ext: str) -> str:
    return f"scenes/{scene}/{phase}/{sub}/{frame}.{ext}"


def build_rel_pose_entry(
    scene: str, phase: str, phase_name: str,
    frame_a: str, frame_b: str, difficulty: str,
) -> Dict[str, Any]:
    return {
        "id": f"{scene}_{phase_name}_{frame_a}_{frame_b}",
        "scene": scene,
        "phase": phase_name,
        "difficulty": difficulty,
        "rgb": _rel(scene, phase, "rgb", frame_a, "png"),
        "rgb_b": _rel(scene, phase, "rgb", frame_b, "png"),
        "pose_a": _rel(scene, phase, "pose", frame_a, "npz"),
        "pose_b": _rel(scene, phase, "pose", frame_b, "npz"),
    }


def build_nvs_entry(
    scene: str, phase: str, phase_name: str,
    frame_src: str, frame_tgt: str, difficulty: str,
) -> Dict[str, Any]:
    return {
        "id": f"{scene}_{phase_name}_{frame_src}_to_{frame_tgt}",
        "scene": scene,
        "phase": phase_name,
        "difficulty": difficulty,
        "rgb": _rel(scene, phase, "rgb", frame_src, "png"),
        "depth": _rel(scene, phase, "depth", frame_src, "png"),
        "pose": _rel(scene, phase, "pose", frame_src, "npz"),
        "target_rgb": _rel(scene, phase, "rgb", frame_tgt, "png"),
        "target_pose": _rel(scene, phase, "pose", frame_tgt, "npz"),
    }


def build_manifests(
    sequences: List[Dict[str, Any]],
    esd: Dict[Tuple[str, str], Dict[str, Any]],
    scenes_root: Path,
    cfg: PairConfig,
    tasks: List[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    manifests: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task in tasks:
        for d in ("easy", "medium", "hard"):
            manifests[(task, d)] = {
                "task": task, "split": d, "scenes": [], "samples": [],
            }

    for seq in sequences:
        scene, phase = seq["scene"], seq["phase"]
        esd_entry = esd.get((scene, phase))
        if esd_entry is None:
            continue
        difficulty = esd_entry["difficulty"]
        if difficulty not in ("easy", "medium", "hard"):
            continue

        # Load pose stream once per (scene, phase).
        phase_dir = scenes_root / scene / phase
        poses = [load_pose_npz(phase_dir / "pose" / f"{fid}.npz")
                 for fid in seq["frame_ids"]]
        pairs = sample_pairs(poses, cfg)
        if not pairs:
            continue

        frame_ids = seq["frame_ids"]
        phase_name = seq["phase_name"]
        scene_entry = {"scene": scene, "phase": phase}

        if "relative_camera_pose" in tasks:
            m = manifests[("relative_camera_pose", difficulty)]
            m["scenes"].append(scene_entry)
            for i, j in pairs:
                m["samples"].append(build_rel_pose_entry(
                    scene, phase, phase_name,
                    frame_ids[i], frame_ids[j], difficulty,
                ))

        if "novel_view_synthesis" in tasks:
            # NVS also needs the source depth, so skip phases missing depth.
            if not (phase_dir / "depth").is_dir():
                continue
            m = manifests[("novel_view_synthesis", difficulty)]
            m["scenes"].append(scene_entry)
            for i, j in pairs:
                m["samples"].append(build_nvs_entry(
                    scene, phase, phase_name,
                    frame_ids[i], frame_ids[j], difficulty,
                ))

    return {k: v for k, v in manifests.items() if v["samples"]}


def write_manifests(
    manifests: Dict[Tuple[str, str], Dict[str, Any]],
    out_root: Path,
) -> List[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for (task, difficulty), data in manifests.items():
        out_path = out_root / task / f"{difficulty}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        written.append(out_path)
    return written


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main() -> int:
    try:
        from rpx_benchmark.banner import show_banner
        show_banner(subtitle="scripts/generate_pair_task_manifests.py")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=PAIR_TASKS,
                        default=list(PAIR_TASKS))
    parser.add_argument("--min-t", type=float, default=0.05,
                        help="Min translation magnitude (metres)")
    parser.add_argument("--max-t", type=float, default=0.60)
    parser.add_argument("--min-r", type=float, default=2.0,
                        help="Min rotation (degrees)")
    parser.add_argument("--max-r", type=float, default=35.0)
    parser.add_argument("--src-stride", type=int, default=10,
                        help="Sample a source frame every N frames")
    args = parser.parse_args()

    local_root: Path = args.local_root
    scenes_root = local_root / "scenes"
    if not scenes_root.is_dir():
        print(f"error: {scenes_root} does not exist", file=sys.stderr)
        return 2

    esd_path = local_root / "esd_scores.json"
    if not esd_path.exists():
        print(f"error: {esd_path} not found — required for difficulty labels",
              file=sys.stderr)
        return 2

    cfg = PairConfig(
        min_t=args.min_t, max_t=args.max_t,
        min_r=args.min_r, max_r=args.max_r,
        src_stride=args.src_stride,
    )

    sequences = discover_pose_sequences(scenes_root)
    esd = load_esd_scores(esd_path)
    print(f"discovered {len(sequences)} pose sequences")

    manifests = build_manifests(sequences, esd, scenes_root, cfg, args.tasks)
    written = write_manifests(manifests, local_root / "manifests")
    print(f"wrote {len(written)} manifests")
    for (task, diff), m in sorted(manifests.items()):
        print(f"  {task:<22} {diff:<6} {len(m['samples'])} pairs "
              f"across {len(m['scenes'])} (scene, phase) groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
