"""Render the same RPX depth frame through several matplotlib colormaps for comparison.

Usage
-----
    PYTHONPATH=. python scripts/compare_depth_colormaps.py [REPO_ID]

Pulls scene23.jo.4f / phase 0 / frame 00100 from the HuggingFace cache, applies
each colormap with the same per-frame 1st–99th percentile stretch, and writes
a side-by-side PNG to benchmark/site/depth_colormaps.png.
"""

import sys
import tarfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rpx_benchmark.dataset_hub import download_for_task

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "itaykadosh/rpx-test"
OUT_PNG = Path(__file__).resolve().parent.parent / "site" / "depth_colormaps.png"
OUT_PNG.parent.mkdir(exist_ok=True)

# Order: most likely candidates first, then perceptually-uniform alternatives,
# then jet as the historical baseline reviewers will compare against.
COLORMAPS = [
    ("turbo", "high contrast, robotics-default (RealSense, OpenCV)"),
    ("plasma", "perceptually uniform, vivid"),
    ("viridis", "perceptually uniform, paper-grade"),
    ("cividis", "colorblind-safe, calm aesthetic"),
    ("magma", "perceptually uniform, dark→warm"),
    ("inferno", "perceptually uniform, dark→hot"),
    ("gray", "no hue, just luminance"),
    ("jet", "historical baseline (avoid for new work)"),
]


def stretch(d: np.ndarray) -> np.ndarray:
    valid = d > 0
    if valid.sum() < 100:
        return np.zeros_like(d, dtype=np.float32)
    lo, hi = np.percentile(d[valid], [1.0, 99.0])
    out = np.zeros_like(d, dtype=np.float32)
    out[valid] = np.clip((d[valid] - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return out


def main():
    print(f"== fetching segmentation/easy + depth from {REPO_ID} ==")
    res = download_for_task(
        task="segmentation",
        split="easy",
        repo_id=REPO_ID,
        extra_modalities=["depth"],
    )
    local = Path(res.local_dir)

    # First scene/phase that has a depth tar.
    for tar in sorted(local.glob("scenes/*/*/depth.tar")):
        scene = tar.parent.parent.name
        phase = tar.parent.name
        depth_tar = tar
        break
    else:
        sys.exit("no depth tar in cache")

    # Pick a mid-walkaround frame for variety.
    target = "rgb/00100.png"
    with tarfile.open(depth_tar, "r") as tf:
        members = sorted(m.name for m in tf if m.isfile())
        member = next((m for m in members if "00100" in m), members[len(members) // 2])
        f = tf.extractfile(member)
        depth = np.array(Image.open(BytesIO(f.read())))

    print(f"  scene/phase/frame: {scene}/{phase}/{member}")
    print(f"  depth: {depth.shape} {depth.dtype}  range=[{depth.min()}, {depth.max()}]")
    valid = depth > 0
    if valid.sum() > 100:
        lo, hi = np.percentile(depth[valid], [1, 99])
        print(f"  per-frame 1–99 pct stretch band: [{lo:.0f}, {hi:.0f}] mm")

    norm = stretch(depth.astype(np.float32))

    n = len(COLORMAPS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows))
    axes = axes.flatten()
    for ax, (cmap_name, blurb) in zip(axes, COLORMAPS, strict=False):
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 0  # invalid pixels black
        ax.imshow(rgb)
        ax.set_title(f"{cmap_name}\n{blurb}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"RPX depth colormap comparison · {scene}/{phase}/{Path(member).stem}\n"
        f"per-frame 1–99 pct stretch · invalid pixels rendered black",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"  saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
