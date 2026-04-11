#!/usr/bin/env python
"""Precompute sparse depth samples and emit manifests.

For each valid frame, samples ``--num-samples`` pixels uniformly from
the set of pixels with non-zero depth (the D435 ``0`` sentinel is the
"no return" marker). Coordinates (pixel) and depths (metres) are
written as per-frame ``.npy`` files under
``scenes/<scene>/<phase>/sparse_depth/`` so a single frame used in
multiple tasks keeps a stable cache footprint.

Usage::

    python scripts/generate_sparse_depth.py \
        --local-root /data/rpx \
        --num-samples 256
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

PHASE_NAMES = {"0": "clutter", "1": "interaction", "2": "clean"}


def load_esd(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {(str(e["scene"]), str(e["phase"])): e for e in data}


def load_depth_mm(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im, dtype=np.uint16)


def sample_valid(
    depth_mm: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(depth_mm > 0)
    if len(ys) == 0:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    k = min(n, len(ys))
    idx = rng.choice(len(ys), size=k, replace=False)
    coords = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    depths = depth_mm[ys[idx], xs[idx]].astype(np.float32) / 1000.0
    return coords, depths


def _frame_seed(base: int, scene: str, phase: str, frame: str) -> int:
    h = hashlib.blake2s(f"{scene}/{phase}/{frame}".encode(), digest_size=4).digest()
    return (base ^ int.from_bytes(h, "big")) & 0x7FFFFFFF


def main() -> int:
    try:
        from rpx_benchmark.banner import show_banner
        show_banner(subtitle="scripts/generate_sparse_depth.py")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=5,
                        help="Emit a manifest entry every Nth frame")
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

    esd = load_esd(esd_path)

    manifests: Dict[str, Dict[str, Any]] = {
        d: {"task": "sparse_depth", "split": d, "scenes": [], "samples": []}
        for d in ("easy", "medium", "hard")
    }
    processed = 0

    for scene_dir in sorted(scenes_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        scene = scene_dir.name
        for phase in ("0", "1", "2"):
            phase_dir = scene_dir / phase
            rgb_dir = phase_dir / "rgb"
            depth_dir = phase_dir / "depth"
            if not (rgb_dir.is_dir() and depth_dir.is_dir()):
                continue
            esd_entry = esd.get((scene, phase))
            if esd_entry is None:
                continue
            difficulty = esd_entry["difficulty"]
            if difficulty not in manifests:
                continue

            phase_name = PHASE_NAMES.get(phase, phase)
            out_dir = phase_dir / "sparse_depth"
            out_dir.mkdir(exist_ok=True)

            depth_files = sorted(depth_dir.glob("*.png"))
            if not depth_files:
                continue
            manifests[difficulty]["scenes"].append({"scene": scene, "phase": phase})

            for idx, depth_file in enumerate(depth_files):
                if idx % args.stride != 0:
                    continue
                frame = depth_file.stem
                rgb_file = rgb_dir / f"{frame}.png"
                if not rgb_file.exists():
                    continue
                rng = np.random.default_rng(_frame_seed(args.seed, scene, phase, frame))
                depth_mm = load_depth_mm(depth_file)
                coords, depths = sample_valid(depth_mm, args.num_samples, rng)
                if len(coords) == 0:
                    continue

                np.save(out_dir / f"{frame}_coords.npy", coords)
                np.save(out_dir / f"{frame}_depths.npy", depths)

                manifests[difficulty]["samples"].append({
                    "id": f"{scene}_{phase_name}_{frame}",
                    "scene": scene,
                    "phase": phase_name,
                    "difficulty": difficulty,
                    "rgb":   f"scenes/{scene}/{phase}/rgb/{frame}.png",
                    "depth": f"scenes/{scene}/{phase}/depth/{frame}.png",
                    "coordinates":
                        f"scenes/{scene}/{phase}/sparse_depth/{frame}_coords.npy",
                    "depths":
                        f"scenes/{scene}/{phase}/sparse_depth/{frame}_depths.npy",
                })
                processed += 1

    out_root = local_root / "manifests" / "sparse_depth"
    out_root.mkdir(parents=True, exist_ok=True)
    for diff, m in manifests.items():
        if not m["samples"]:
            continue
        with (out_root / f"{diff}.json").open("w", encoding="utf-8") as f:
            json.dump(m, f)
        print(f"  sparse_depth  {diff:<6} {len(m['samples'])} frames "
              f"({len(m['scenes'])} phases)")
    print(f"total: {processed} frames processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
