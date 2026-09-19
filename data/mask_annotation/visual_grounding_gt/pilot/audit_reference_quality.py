"""One-off audit: mask-area-fraction distribution across all 70 published SOS
objects' frames, used to set sos_reference.py's MIN_REF_MASK_AREA_FRAC from
real data instead of a guess. Downloads only labels/masks/v1.tar per object
(~1MB each, ~70MB total) -- never rgb.tar. Not part of the generation
pipeline; run once, record the result in sos_reference.py's comment."""
import io
import os
import sys
import tarfile

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "vqa_gt"))
sys.path.insert(0, "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt")

import sos_catalog as sc  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


def main():
    catalog = sc.load_catalog()
    fracs = []
    n_objects = 0
    for object_id, obj in catalog.items():
        shard = f"{obj.sos_data_path}0/labels/masks/v1.tar"
        p = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=shard, revision="main")
        with tarfile.open(p) as tf:
            names = sorted(n for n in tf.getnames() if n.lower().endswith(".png"))
            for n in names[::5]:  # every 5th of 500 -> 100 frames/object
                mask = np.asarray(Image.open(io.BytesIO(tf.extractfile(n).read())))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                area = int((mask > 0).sum())
                H, W = mask.shape[:2]
                fracs.append(area / (W * H))
        real = os.path.realpath(p)
        os.remove(p)
        if os.path.exists(real):
            os.remove(real)
        n_objects += 1
        print(f"[{n_objects}/{len(catalog)}] {object_id}: {len(names[::5])} frames scored", flush=True)

    fracs = np.array(fracs)
    print(f"\n{len(fracs)} (object,frame) samples across {n_objects} objects")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{p}: {np.percentile(fracs, p):.5f}")
    print("min:", fracs.min(), "max:", fracs.max())


if __name__ == "__main__":
    main()
