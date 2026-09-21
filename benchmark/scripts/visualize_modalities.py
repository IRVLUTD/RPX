"""Visualize all four core RPX modalities for a single (scene, phase, frame).

Renders RGB, depth, fisheye (stereo), and instance masks side-by-side.

Usage
-----
    PYTHONPATH=. python scripts/visualize_modalities.py [REPO_ID]

REPO_ID defaults to anonymous/RPX.
"""

import sys
import tarfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rpx_benchmark.dataset_hub import download_for_task

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "anonymous/RPX"
OUT_PNG = Path(__file__).resolve().parent.parent / "site" / "rpx_modalities.png"
OUT_PNG.parent.mkdir(exist_ok=True)


def list_members(tar_path: Path):
    with tarfile.open(tar_path, "r") as tf:
        return [m.name for m in tf if m.isfile()]


def read_member(tar_path: Path, member_name: str) -> bytes:
    with tarfile.open(tar_path, "r") as tf:
        f = tf.extractfile(member_name)
        if f is None:
            raise FileNotFoundError(f"{member_name} not in {tar_path}")
        return f.read()


def find_tar(local_dir: Path, scene: str, phase: str, modality: str) -> Path | None:
    """Locate the tar for (scene, phase, modality), checking raw + label paths."""
    raw = local_dir / "scenes" / scene / phase / f"{modality}.tar"
    if raw.exists():
        return raw
    for label_tar in (local_dir / "scenes" / scene / phase / "labels" / modality).glob("v*.tar"):
        return label_tar
    return None


def main():
    print(f"== fetching segmentation/easy + depth + fisheye from {REPO_ID} ==")
    res = download_for_task(
        task="segmentation",
        split="easy",
        repo_id=REPO_ID,
        extra_modalities=["depth", "fisheye"],
    )
    local = Path(res.local_dir)
    print(f"  local_dir    : {local}")
    print(f"  bytes_fetched: {res.bytes_fetched / 1e6:.1f} MB")

    # Pick the first scene/phase that has all four modalities.
    for rgb_tar in sorted(local.glob("scenes/*/*/rgb.tar")):
        scene = rgb_tar.parent.parent.name
        phase = rgb_tar.parent.name
        depth_tar = find_tar(local, scene, phase, "depth")
        fisheye_tar = find_tar(local, scene, phase, "fisheye")
        mask_tar = find_tar(local, scene, phase, "masks")
        if all([depth_tar, fisheye_tar, mask_tar]):
            break
    else:
        sys.exit("no (scene, phase) has all four modalities downloaded")

    print(f"  sample       : scene={scene} phase={phase}")
    print(f"  rgb_tar      : {rgb_tar.name}")
    print(f"  depth_tar    : {depth_tar.name}")
    print(f"  fisheye_tar  : {fisheye_tar.name}")
    print(f"  mask_tar     : {mask_tar.name}")

    # Use the first frame from rgb to fix the frame index across modalities.
    rgb_members = sorted(list_members(rgb_tar))
    if not rgb_members:
        sys.exit("rgb tar empty")
    rgb_member = rgb_members[0]  # e.g. "rgb/00000.png"
    frame_id = Path(rgb_member).stem  # "00000"
    print(f"  frame_id     : {frame_id}")

    # Read RGB.
    rgb = np.array(Image.open(BytesIO(read_member(rgb_tar, rgb_member))))

    # Read depth — keep raw dtype, fall back if naming differs.
    depth_members = list_members(depth_tar)
    depth_member = next((m for m in depth_members if frame_id in m), depth_members[0])
    depth = np.array(Image.open(BytesIO(read_member(depth_tar, depth_member))))

    # Read fisheye — typically stereo, one frame_id appears in both left + right paths.
    fisheye_members = sorted(list_members(fisheye_tar))
    matched = [m for m in fisheye_members if frame_id in m]
    if len(matched) >= 2:
        fisheye_left = np.array(Image.open(BytesIO(read_member(fisheye_tar, matched[0]))))
        fisheye_right = np.array(Image.open(BytesIO(read_member(fisheye_tar, matched[1]))))
    else:
        fisheye_left = np.array(
            Image.open(
                BytesIO(read_member(fisheye_tar, matched[0] if matched else fisheye_members[0]))
            )
        )
        fisheye_right = None

    # Read masks.
    mask_members = list_members(mask_tar)
    mask_member = next((m for m in mask_members if frame_id in m), mask_members[0])
    mask = np.array(Image.open(BytesIO(read_member(mask_tar, mask_member))))

    print(f"  rgb          : {rgb.shape} {rgb.dtype}")
    print(f"  depth        : {depth.shape} {depth.dtype}  range=[{depth.min()}, {depth.max()}]")
    print(f"  fisheye_l    : {fisheye_left.shape} {fisheye_left.dtype}")
    if fisheye_right is not None:
        print(f"  fisheye_r    : {fisheye_right.shape} {fisheye_right.dtype}")
    print(f"  mask         : {mask.shape} {mask.dtype}  unique={len(np.unique(mask))}")

    # Lay out: 2 rows × 3 cols.
    #  row 0: RGB | depth | mask overlay
    #  row 1: fisheye L | fisheye R (or blank) | (blank)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB · {scene}/{phase}/{frame_id}")
    axes[0, 0].axis("off")

    # Depth: clip to robotically meaningful range (0.3–3 m → mm)
    depth_m = depth.astype(np.float32) / 1000.0  # mm → m, assuming D435 convention
    depth_disp = np.where((depth_m > 0.0) & (depth_m < 5.0), depth_m, np.nan)
    im = axes[0, 1].imshow(depth_disp, cmap="turbo", vmin=0.3, vmax=3.0)
    axes[0, 1].set_title(f"Depth (m) · {depth_member}")
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(rgb)
    if mask.ndim == 2:
        masked = np.ma.masked_where(mask == 0, mask)
        axes[0, 2].imshow(masked, cmap="tab20", alpha=0.55)
    else:
        axes[0, 2].imshow(mask, alpha=0.55)
    axes[0, 2].set_title(f"RGB + masks · {mask_member}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(fisheye_left, cmap="gray" if fisheye_left.ndim == 2 else None)
    axes[1, 0].set_title(f"Fisheye L · {matched[0] if matched else fisheye_members[0]}")
    axes[1, 0].axis("off")

    if fisheye_right is not None:
        axes[1, 1].imshow(fisheye_right, cmap="gray" if fisheye_right.ndim == 2 else None)
        axes[1, 1].set_title(f"Fisheye R · {matched[1]}")
    else:
        axes[1, 1].text(0.5, 0.5, "(no stereo pair)", ha="center", va="center")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"  saved        : {OUT_PNG}")


if __name__ == "__main__":
    main()
