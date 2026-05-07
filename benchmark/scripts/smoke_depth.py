"""Smoke test: run ZoeDepth on a few RPX frames from the HF cache, compute metrics.

Bypasses the toolkit's download_split (no published manifest for the test mirror
yet) — pulls RGB+depth from the cached tar shards directly. Used to validate the
model adapter works on RPX imagery before wiring into the full pipeline.

Usage
-----
    PYTHONPATH=. python scripts/smoke_depth.py [--frames N]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _open_tar(path: Path):
    return tarfile.open(path, "r")


def _read_frame(tar_path: Path, member_name: str) -> np.ndarray:
    with _open_tar(tar_path) as tf:
        f = tf.extractfile(member_name)
        return np.array(Image.open(BytesIO(f.read())))


def _find_pairs(local_root: Path, max_frames: int) -> list[dict]:
    """Yield (rgb, gt_depth_mm) pairs across (scene, phase) tuples in the cache."""
    pairs = []
    for rgb_tar in sorted(local_root.glob("scenes/*/*/rgb.tar")):
        scene = rgb_tar.parent.parent.name
        phase = rgb_tar.parent.name
        depth_tar = rgb_tar.parent / "depth.tar"
        if not depth_tar.exists():
            continue
        with _open_tar(rgb_tar) as tf:
            rgb_members = sorted(m.name for m in tf if m.isfile())
        if not rgb_members:
            continue
        # one frame per (scene, phase) is enough for a smoke test
        for fname in [rgb_members[len(rgb_members) // 2]]:
            stem = Path(fname).stem
            depth_member = f"depth/{stem}.png"
            pairs.append({
                "scene": scene, "phase": phase, "frame": stem,
                "rgb_tar": rgb_tar, "rgb_member": fname,
                "depth_tar": depth_tar, "depth_member": depth_member,
            })
            if len(pairs) >= max_frames:
                return pairs
    return pairs


def _depth_metrics(pred_m: np.ndarray, gt_mm: np.ndarray) -> dict:
    """Standard monocular-depth metrics on the valid-pixel set, pred + gt in metres."""
    gt_m = gt_mm.astype(np.float32) / 1000.0
    valid = (gt_m > 0.3) & (gt_m < 5.0) & (pred_m > 0.0) & np.isfinite(pred_m)
    if valid.sum() < 100:
        return {"valid_px": int(valid.sum())}
    g = gt_m[valid]
    p = pred_m[valid]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))
    return {
        "valid_px": int(valid.sum()),
        "abs_rel": abs_rel,
        "rmse_m": rmse,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "pred_med_m": float(np.median(p)),
        "gt_med_m": float(np.median(g)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4,
                    help="number of (scene, phase) frames to smoke (default: 4)")
    args = ap.parse_args()

    from rpx_benchmark.dataset_hub import download_for_task
    print("== ensuring rgb + depth shards are in the HF cache ==")
    res = download_for_task(
        task="segmentation", split="easy",
        repo_id="itaykadosh/rpx-test", extra_modalities=["depth"],
    )
    local = Path(res.local_dir)

    print("== finding rgb / depth pairs ==")
    pairs = _find_pairs(local, args.frames)
    if not pairs:
        sys.exit("no (rgb, depth) pairs found in cache")
    for p in pairs:
        print(f"  {p['scene']}/{p['phase']}/{p['frame']}")

    print("\n== loading ZoeDepth ==")
    t0 = time.time()
    from depth_models.zoedepth import ZoeDepth
    zoe = ZoeDepth(device="cuda")
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("\n== inference + metrics ==")
    print(f"{'scene':>22} {'phase':>5} {'frame':>5}   {'AbsRel':>7} {'RMSE_m':>7} "
          f"{'δ1':>5} {'δ2':>5} {'δ3':>5}  {'pred_med':>8} {'gt_med':>7}  {'t_s':>5}")
    for p in pairs:
        rgb = _read_frame(p["rgb_tar"], p["rgb_member"])
        gt_mm = _read_frame(p["depth_tar"], p["depth_member"])
        t0 = time.time()
        pred_m = zoe(rgb)
        dt = time.time() - t0
        m = _depth_metrics(pred_m, gt_mm)
        if "abs_rel" not in m:
            print(f"  {p['scene']:>22} {p['phase']:>5} {p['frame']:>5}   "
                  f"(too few valid pixels: {m['valid_px']})")
            continue
        print(f"  {p['scene']:>22} {p['phase']:>5} {p['frame']:>5}   "
              f"{m['abs_rel']:7.4f} {m['rmse_m']:7.3f} "
              f"{m['delta1']:5.3f} {m['delta2']:5.3f} {m['delta3']:5.3f}  "
              f"{m['pred_med_m']:8.3f} {m['gt_med_m']:7.3f}  {dt:5.2f}")

    print("\nDone. ZoeDepth ran end-to-end on RPX imagery — adapter is sane.")


if __name__ == "__main__":
    main()
