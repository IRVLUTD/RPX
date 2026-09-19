"""
Visualize generated spatial GT questions for a single scene.
Renders RGB frame with centroid dots, object labels, and Q&A overlay.

Usage:
    python visual_grounding_gt/viz_spatial_gt.py \
        --jsonl /tmp/vqa_gt_test/scene1.library.fountain_spatial_gt.jsonl \
        --out /tmp/vqa_gt_viz
"""

import argparse
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (255, 80,  80),
    (80,  200, 80),
    (80,  80,  255),
    (255, 200, 0),
    (200, 80,  255),
    (0,   200, 220),
    (255, 140, 0),
]

def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def draw_dot(draw, cx, cy, color, r=8):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255), width=2)

def draw_label(draw, text, x, y, font, color):
    # dark background behind text for readability
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=color, font=font)

def render_frame(rgb_path, mask_path, frame_questions):
    rgb = Image.open(rgb_path).convert("RGB")
    mask_arr = np.array(Image.open(mask_path))
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    W, H = rgb.size
    Q_PANEL = 260
    canvas = Image.new("RGB", (W + Q_PANEL, H), (30, 30, 30))
    canvas.paste(rgb, (0, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")
    font_label = load_font(14)
    font_q = load_font(13)
    font_small = load_font(11)

    # collect all unique instances across this frame's questions
    instances = {}
    for q in frame_questions:
        ev = q.get("evidence", {})
        if q["type"] == "spatial_lr_binary":
            instances[q["evidence"]["name_a"]] = q["evidence"]["cx_a"]
            instances[q["evidence"]["name_b"]] = q["evidence"]["cx_b"]
        elif q["type"] == "spatial_lr_neighbor":
            instances[q["evidence"]["query"]] = q["evidence"]["query_cx"]
        elif q["type"] == "depth_closest":
            for name in ev:
                instances[name] = None  # cx not in evidence for depth q
        elif q["type"] == "spatial_ud_binary":
            instances[q["evidence"]["name_a"]] = None
            instances[q["evidence"]["name_b"]] = None

    # compute centroids from mask
    mask_centroids = {}
    for mid in np.unique(mask_arr).tolist():
        if mid == 0:
            continue
        ys, xs = np.nonzero(mask_arr == mid)
        if len(xs) == 0:
            continue
        mask_centroids[mid] = (float(xs.mean()), float(ys.mean()))

    # build name → (cx, cy, color) from mask
    # we need the mapping — load it from the mask_path location
    map_path = os.path.join(os.path.dirname(os.path.dirname(mask_path)), "mask_to_object.json")
    name_to_pos = {}
    color_map = {}
    if os.path.exists(map_path):
        with open(map_path) as f:
            mapping = json.load(f)
        for k, v in mapping.items():
            try:
                mid = int(k)
            except ValueError:
                continue
            obj = v.get("object", v)
            name = obj.get("name", str(mid))
            color_idx = (mid - 1) % len(COLORS)
            color_map[name] = COLORS[color_idx]
            if mid in mask_centroids:
                cx, cy = mask_centroids[mid]
                name_to_pos[name] = (cx, cy)

    # draw centroids + labels on RGB side
    for name, (cx, cy) in name_to_pos.items():
        color = color_map.get(name, (200, 200, 200))
        draw_dot(draw, cx, cy, color)
        draw_label(draw, name.replace("_", " "), cx + 10, cy - 10, font_label, color)

    # draw Q&A panel on right side
    px = W + 10
    py = 10
    draw.text((px, py), f"Frame: {frame_questions[0]['frame']}", fill=(220, 220, 220), font=font_label)
    py += 24

    for i, q in enumerate(frame_questions):
        type_color = {
            "spatial_lr_binary": (120, 200, 255),
            "spatial_lr_neighbor": (120, 255, 160),
            "depth_closest": (255, 200, 80),
            "spatial_ud_binary": (255, 140, 120),
        }.get(q["type"], (200, 200, 200))

        # type tag
        draw.text((px, py), f"[{q['type']}]", fill=type_color, font=font_small)
        py += 16

        # question (word-wrap to ~30 chars)
        words = q["question"].split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > 28:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for ln in lines:
            draw.text((px, py), ln, fill=(255, 255, 255), font=font_q)
            py += 16

        # answer
        ans_color = (80, 255, 120) if q["answer"] == "yes" else (255, 100, 100) if q["answer"] == "no" else (255, 220, 80)
        draw.text((px, py), f"→ {q['answer']}", fill=ans_color, font=font_label)
        py += 22

        # divider
        draw.line([(px, py), (W + Q_PANEL - 10, py)], fill=(80, 80, 80), width=1)
        py += 8

        if py > H - 40:
            break

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to spatial GT JSONL")
    ap.add_argument("--out", required=True, help="output directory for PNG images")
    ap.add_argument("--max_frames", type=int, default=20, help="max frames to render")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.jsonl) as f:
        records = [json.loads(l) for l in f]

    # group by (phase, frame)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r["phase"], r["frame"])].append(r)

    rendered = 0
    for (phase, fid), qs in sorted(groups.items()):
        if rendered >= args.max_frames:
            break
        rgb_path = qs[0]["rgb_path"]
        mask_path = qs[0]["mask_path"]
        if not os.path.exists(rgb_path):
            print(f"Missing RGB: {rgb_path}")
            continue

        img = render_frame(rgb_path, mask_path, qs)
        out_name = f"phase{phase}_frame{fid}.png"
        img.save(os.path.join(args.out, out_name))
        print(f"Saved {out_name}")
        rendered += 1

    print(f"Done. {rendered} frames rendered → {args.out}")


if __name__ == "__main__":
    main()
