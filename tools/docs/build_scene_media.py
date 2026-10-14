#!/usr/bin/env python3
"""Rebuild the scene photos used across the toolkit docs from released RPX data.

The docs show one scene throughout: scene049, a Hard scene from capture site 10
(outdoor steps under tree shade). Frames were picked by hand for sharpness; each
is the sharpest frame (variance of the Laplacian) in its window.

Inputs are the released shards at the pinned dataset revision. Point --data at a
folder holding them (named as below) or let the script download them:

    python tools/docs/build_scene_media.py --data /path/to/shards

Masks are drawn from the released SAM 2 labels with one colour per object, so an
object keeps its colour across views. Depth is shown with viridis over the
0.3-5.0 m evaluation range; invalid pixels are black.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageOps

REVISION = "51432e5d48895bbafb490ae7c5e34e0f6b7da7d6"
ASSETS = Path(__file__).resolve().parents[2] / "benchmark/rpx_benchmark/guides/assets"
SCENE = "scene049"
FRAMES = {"clutter": ("0", 67), "interaction": ("1", 185), "clean": ("2", 80), "ego": ("ego", 149)}
NEXT_FRAME = 186                       # adjacent Interaction frame for the two-image walkthroughs
REFERENCE = ("lion.2", 329, 39)        # SOS object, frame, global object ID (the lion in scene049)
# One colour per local mask ID; the same palette as the paper's scene figures.
PALETTE = ["#c875b7", "#eea026", "#68bce4", "#169579", "#e0ca39", "#237bad", "#c77528"]
SHARDS = {
    "rgb_0": f"scenes/{SCENE}/0/rgb.tar", "rgb_1": f"scenes/{SCENE}/1/rgb.tar",
    "rgb_2": f"scenes/{SCENE}/2/rgb.tar", "rgb_ego": f"scenes/{SCENE}/ego/rgb.tar",
    "mask_0": f"scenes/{SCENE}/0/labels/masks/v1.tar", "mask_1": f"scenes/{SCENE}/1/labels/masks/v1.tar",
    "mask_2": f"scenes/{SCENE}/2/labels/masks/v1.tar", "mask_ego": f"scenes/{SCENE}/ego/labels/masks/v1.tar",
    "depth_1": f"scenes/{SCENE}/1/depth.tar", "fisheye_1": f"scenes/{SCENE}/1/fisheye.tar",
    "sos_rgb": f"objects/{REFERENCE[0]}/0/rgb.tar", "sos_mask": f"objects/{REFERENCE[0]}/0/labels/masks/v1.tar",
}


def rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def member(path, name):
    with tarfile.open(path) as tar:
        return tar.extractfile(name).read()


def image(path, name, mode="RGB"):
    return np.asarray(Image.open(io.BytesIO(member(path, name))).convert(mode))


def mask(path, frame):
    m = np.asarray(Image.open(io.BytesIO(member(path, f"sam2/masks/{frame:05d}.png"))))
    return m[..., 0] if m.ndim == 3 else m


def crop_4_3(*arrays):
    h, w = arrays[0].shape[:2]
    if w * 3 == h * 4:
        return arrays
    cw = h * 4 // 3
    x0 = (w - cw) // 2
    return tuple(a[:, x0:x0 + cw] for a in arrays)


def overlay(photo, labels):
    out = photo.astype(np.float32)
    ids = [i for i in np.unique(labels) if i]
    for i in ids:
        sel = labels == i
        out[sel] = out[sel] * 0.70 + rgb(PALETTE[(int(i) - 1) % len(PALETTE)]) * 0.30
    out = out.astype(np.uint8)
    width = max(2, out.shape[1] // 320)
    for i in ids:
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = tuple(int(c) for c in rgb(PALETTE[(int(i) - 1) % len(PALETTE)]))
        cv2.drawContours(out, contours, -1, colour, width, cv2.LINE_AA)
    return out


def save(array, name, size=None, quality=92):
    img = Image.fromarray(array)
    if size and img.size != size:
        img = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    target = ASSETS / name
    if target.suffix == ".png":
        img.save(target, optimize=True)
    else:
        img.save(target, quality=quality, optimize=True)
    return {"file": name, "size": list(img.size)}


def build(shards):
    made = []
    for view, (phase, frame) in FRAMES.items():
        photo = image(shards[f"rgb_{phase}"], f"rgb/{frame:05d}.webp")
        labels = mask(shards[f"mask_{phase}"], frame)
        photo, labels = crop_4_3(photo, labels)
        size = (960, 720) if view == "ego" else (640, 480)
        made.append(save(overlay(photo, labels), f"{view}.jpg", size) | {"frame": frame, "phase": phase})
    phase, frame = FRAMES["interaction"]
    made.append(save(overlay(image(shards["rgb_1"], f"rgb/{frame:05d}.webp"), mask(shards["mask_1"], frame)),
                     "hero-interaction.jpg") | {"frame": frame, "phase": phase})
    raw = image(shards["rgb_1"], f"rgb/{frame:05d}.webp")
    made.append(save(raw, "rgb.jpg") | {"frame": frame})
    made.append(save(raw, f"{SCENE}-rgb.png") | {"frame": frame})
    made.append(save(image(shards["rgb_1"], f"rgb/{NEXT_FRAME:05d}.webp"), f"{SCENE}-next.png") | {"frame": NEXT_FRAME})

    depth = image(shards["depth_1"], f"depth/{frame:05d}.png", mode="I").astype(np.float32) / 1000.0
    valid = (depth >= 0.3) & (depth <= 5.0)
    coloured = (matplotlib.colormaps["viridis"]((np.clip(depth, 0.3, 5.0) - 0.3) / 4.7)[..., :3] * 255).astype(np.uint8)
    coloured[~valid] = 0
    made.append(save(coloured, "depth.png") | {"frame": frame})
    made.append(save(coloured, f"{SCENE}-depth.png") | {"frame": frame})

    labels = mask(shards["mask_1"], frame)
    painted = np.zeros(labels.shape + (3,), np.uint8)
    for i in [i for i in np.unique(labels) if i]:
        painted[labels == i] = rgb(PALETTE[(int(i) - 1) % len(PALETTE)]).astype(np.uint8)
    made.append(save(painted, "mask.png") | {"frame": frame})

    for side in ("left", "right"):
        fisheye = image(shards["fisheye_1"], f"fisheye/{side}/{frame:05d}.webp", mode="L")
        made.append(save(np.stack([fisheye] * 3, -1), f"stereo-{side}.jpg") | {"frame": frame})

    obj, sos_frame, global_id = REFERENCE
    photo = image(shards["sos_rgb"], f"rgb/{sos_frame:05d}.webp")
    ys, xs = np.nonzero(mask(shards["sos_mask"], sos_frame))
    pad = 12
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, photo.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, photo.shape[1])
    made.append(save(photo[y0:y1, x0:x1], f"{SCENE}-sos-reference.png") | {"object": obj, "frame": sos_frame,
                                                                          "global_object_id": global_id})
    return made


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="folder with the shards named <key>.tar (see SHARDS)")
    args = parser.parse_args()
    shards = {}
    for key, remote in SHARDS.items():
        local = args.data / f"{key}.tar" if args.data else None
        if local is None or not local.exists():
            from huggingface_hub import hf_hub_download
            local = Path(hf_hub_download("IRVLUTD/RPX", remote, repo_type="dataset", revision=REVISION))
        shards[key] = local
    made = build(shards)
    note = ASSETS / "scene_media.json"
    note.write_text(json.dumps({"scene": SCENE, "revision": REVISION, "shards": SHARDS, "files": made}, indent=2) + "\n")
    print(f"wrote {len(made)} images from {SCENE}")


if __name__ == "__main__":
    main()
