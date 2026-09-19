"""
Visualize depth maps + mask overlays for frames flagged as having occluded objects.
Shows: RGB | Depth (colorized) | Mask overlay with per-object area ratio labels.

Usage:
    python visual_grounding_gt/viz_depth_occlusion.py \
        --jsonl /tmp/vqa_gt_test/scene41.gh.out_spatial_gt.jsonl \
        --out /tmp/vqa_gt_depth_viz
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def colorize_depth(depth_arr):
    """Map depth to a jet-like colormap. Zero/invalid pixels → black."""
    valid = depth_arr > 0
    out = np.zeros((*depth_arr.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    mn, mx = depth_arr[valid].min(), depth_arr[valid].max()
    norm = np.zeros_like(depth_arr)
    if mx > mn:
        norm[valid] = (depth_arr[valid] - mn) / (mx - mn)

    # jet colormap approximation
    r = np.clip(1.5 - abs(norm * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(norm * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(norm * 4 - 1), 0, 1)
    out[..., 0] = (r * 255).astype(np.uint8)
    out[..., 1] = (g * 255).astype(np.uint8)
    out[..., 2] = (b * 255).astype(np.uint8)
    out[~valid] = 0
    return out


MASK_COLORS = [
    (255, 80,  80),
    (80,  220, 80),
    (80,  80,  255),
    (255, 200, 0),
    (200, 80,  255),
    (0,   200, 220),
    (255, 140, 0),
]


def render_frame(rgb_path, mask_path, depth_path, occ_questions, map_path):
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    H, W = rgb.shape[:2]

    # mask
    mask_im = Image.open(mask_path)
    mask = np.array(mask_im)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.int32)

    # depth
    if depth_path and os.path.exists(depth_path):
        d_im = Image.open(depth_path)
        if d_im.mode not in ("L", "I;16", "I"):
            d_im = d_im.convert("I")
        depth = np.array(d_im).astype(np.float32)
    else:
        depth = np.zeros((H, W), dtype=np.float32)

    depth_color = colorize_depth(depth)

    # load mapping for names
    name_map = {}
    if os.path.exists(map_path):
        with open(map_path) as f:
            data = json.load(f)
        for k, v in data.items():
            try:
                mid = int(k)
            except ValueError:
                continue
            obj = v.get("object", v)
            name_map[mid] = obj.get("name", str(mid))

    # build mask overlay on RGB
    mask_overlay = rgb.copy()
    for mid in np.unique(mask).tolist():
        if mid == 0:
            continue
        color = MASK_COLORS[(mid - 1) % len(MASK_COLORS)]
        region = mask == mid
        mask_overlay[region] = (
            mask_overlay[region] * 0.45 + np.array(color) * 0.55
        ).astype(np.uint8)

    # build mask overlay on depth
    depth_overlay = depth_color.copy()
    for mid in np.unique(mask).tolist():
        if mid == 0:
            continue
        color = MASK_COLORS[(mid - 1) % len(MASK_COLORS)]
        region = mask == mid
        # draw border: dilate mask, XOR to get edge
        from scipy.ndimage import binary_dilation
        border = binary_dilation(region, iterations=2) & ~region
        depth_overlay[border] = color

    # stitch: RGB+mask | Depth+mask border | info panel
    PANEL = 280
    canvas = np.zeros((H, W * 2 + PANEL, 3), dtype=np.uint8)
    canvas[:, :W] = mask_overlay
    canvas[:, W:W*2] = depth_overlay

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(13)
    font_sm = load_font(11)

    # column headers
    draw.text((4, 4), "RGB + masks", fill=(255,255,255), font=font)
    draw.text((W + 4, 4), "Depth (blue=near, red=far)", fill=(255,255,255), font=font)

    # centroid dots + area ratio labels on RGB panel
    for mid in np.unique(mask).tolist():
        if mid == 0:
            continue
        region = mask == mid
        ys, xs = np.nonzero(region)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        color = MASK_COLORS[(mid - 1) % len(MASK_COLORS)]
        r = 6
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(255,255,255), width=2)

    # occlusion info panel
    px, py = W * 2 + 8, 8
    frame_id = occ_questions[0]["frame"] if occ_questions else "?"
    draw.text((px, py), f"Frame {frame_id}", fill=(220,220,220), font=font)
    py += 22

    draw.text((px, py), "Occlusion questions:", fill=(180,180,180), font=font_sm)
    py += 18

    for q in occ_questions:
        ev = q["evidence"]
        ans_color = (80, 255, 120) if q["answer"] == "yes" else (255, 100, 100)
        obj_name = q["question"].split("Is the ")[1].split(" partially")[0]

        # find this object's mid for color
        mid = next((m for m, n in name_map.items() if n == obj_name), None)
        obj_color = MASK_COLORS[(mid - 1) % len(MASK_COLORS)] if mid else (200, 200, 200)

        draw.rectangle([px-2, py-2, px+PANEL-12, py+14], fill=(30,30,30,200))
        draw.text((px, py), obj_name.replace("_", " "), fill=obj_color, font=font)
        py += 16
        occluder = ev.get("occluder") or ev.get("current_area", "—")
        fraction = ev.get("adjacent_fraction", ev.get("ratio", "—"))
        draw.text((px, py),
                  f"  occluder: {occluder}",
                  fill=(200,200,200), font=font_sm)
        py += 14
        draw.text((px, py),
                  f"  adj_fraction: {fraction}",
                  fill=(200,200,200), font=font_sm)
        py += 14
        draw.text((px, py), f"  → {q['answer']}", fill=ans_color, font=font)
        py += 20
        draw.line([(px, py), (px + PANEL - 16, py)], fill=(60,60,60), width=1)
        py += 8

        # also draw ratio label on the mask overlay near centroid
        if mid is not None:
            region = mask == mid
            ys2, xs2 = np.nonzero(region)
            if len(xs2):
                lx, ly = int(xs2.mean()) + 8, int(ys2.mean()) - 20
                label = f"{ev['ratio']:.2f}"
                draw.text((lx, ly), label, fill=ans_color, font=font_sm)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.jsonl) as f:
        records = [json.loads(l) for l in f]

    occ_by_frame = defaultdict(list)
    for r in records:
        if r["type"] == "occlusion_binary":
            occ_by_frame[(r["phase"], r["frame"])].append(r)

    for (phase, fid), qs in sorted(occ_by_frame.items()):
        rgb_path = qs[0]["rgb_path"]
        mask_path = qs[0]["mask_path"]
        depth_path = mask_path.replace("sam2/masks", "depth").replace(
            os.path.join("sam2", "masks"), "depth"
        )
        # depth is at root/depth/fid.png
        root = os.path.dirname(os.path.dirname(os.path.dirname(mask_path)))
        depth_path = os.path.join(root, "depth", f"{fid}.png")
        map_path = os.path.join(os.path.dirname(mask_path), "mask_to_object.json")

        img = render_frame(rgb_path, mask_path, depth_path, qs, map_path)
        out_name = f"phase{phase}_frame{fid}_occlusion.png"
        img.save(os.path.join(args.out, out_name))
        print(f"Saved {out_name}  ({len(qs)} occlusion questions)")

    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
