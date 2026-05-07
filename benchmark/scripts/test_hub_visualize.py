"""Smoke-test the HF dataset and visualize one (scene, phase) sample.

Usage
-----
    python scripts/test_hub_visualize.py [REPO_ID]

REPO_ID defaults to itaykadosh/rpx-test for the staging mirror.
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
OUT_PNG = Path(__file__).resolve().parent.parent / "site" / "rpx_hub_smoke.png"
OUT_PNG.parent.mkdir(exist_ok=True)


def first_member(tar_path: Path):
    """Return (name, bytes) of the first file member in a tar archive."""
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if m.isfile():
                f = tf.extractfile(m)
                return m.name, f.read() if f else b""
    raise FileNotFoundError(f"no files in {tar_path}")


def find_tars(local_dir: Path, modality: str):
    """Yield (scene, phase, tar_path) triples for shards of `modality`."""
    for tar in sorted(local_dir.glob(f"scenes/*/*/{modality}.tar")):
        scene = tar.parent.parent.name
        phase = tar.parent.name
        yield scene, phase, tar
    for tar in sorted(local_dir.glob(f"scenes/*/*/labels/{modality}/v*.tar")):
        scene = tar.parent.parent.parent.parent.name
        phase = tar.parent.parent.parent.name
        yield scene, phase, tar


def main():
    print(f"== downloading segmentation/easy from {REPO_ID} ==")
    res = download_for_task(task="segmentation", split="easy", repo_id=REPO_ID)
    local = Path(res.local_dir)
    print(f"  local_dir   : {local}")
    print(f"  bytes_fetched: {res.bytes_fetched / 1e6:.1f} MB")

    rgb_shards = list(find_tars(local, "rgb"))
    mask_shards = list(find_tars(local, "masks"))
    print(f"  rgb shards  : {len(rgb_shards)}")
    print(f"  mask shards : {len(mask_shards)}")

    if not rgb_shards:
        sys.exit("no rgb shards downloaded — recipe or repo wrong")

    # First rgb tar; pull the matching mask tar from the same (scene, phase).
    scene, phase, rgb_tar = rgb_shards[0]
    mask_tar = next((t for s, p, t in mask_shards if s == scene and p == phase), None)

    rgb_name, rgb_bytes = first_member(rgb_tar)
    rgb = np.array(Image.open(BytesIO(rgb_bytes)))
    print(f"  sample      : scene={scene} phase={phase} frame={rgb_name}")
    print(f"  rgb shape   : {rgb.shape} dtype={rgb.dtype}")

    fig_cols = 2 if mask_tar else 1
    fig, axes = plt.subplots(1, fig_cols, figsize=(6 * fig_cols, 5))
    if fig_cols == 1:
        axes = [axes]

    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB · {scene}/{phase}\n{rgb_name}")
    axes[0].axis("off")

    if mask_tar:
        mask_name, mask_bytes = first_member(mask_tar)
        mask = np.array(Image.open(BytesIO(mask_bytes)))
        print(f"  mask shape  : {mask.shape} dtype={mask.dtype}")
        if mask.ndim == 2:
            disp = mask
            cmap = "tab20"
        else:
            disp = mask
            cmap = None
        axes[1].imshow(rgb)
        axes[1].imshow(disp, cmap=cmap, alpha=0.55)
        axes[1].set_title(f"RGB + mask overlay\n{mask_name}")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"  saved       : {OUT_PNG}")


if __name__ == "__main__":
    main()
