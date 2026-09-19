"""Phase-E contact sheet: for each in-context type, render >=10 real
examples as (Image 1 reference crop | Image 2 target frame with the answer
bbox drawn) pairs, plus the fixed question text, into one HTML gallery for
human inspection. Reads only already-generated/validated data (the scene001
dry run) and already-staged scene001 frames -- no new downloads, no writes
to any published output.
"""
import base64
import io
import json
import os
import random
import sys
from collections import defaultdict

from PIL import Image, ImageDraw

STAGE_ROOT = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged"
ITEMS_PATH = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/incontext_dryrun/items.jsonl"
REF_CROPS_DIR = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops/crops"
OUT_HTML = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/incontext_audit_gallery.html"
N_PER_TYPE = 12


def target_frame_path(row):
    phase_dir = "ego" if row["kind"] == "ego" else str(row["phase"])
    base = f"{STAGE_ROOT}/{row['scene_id']}/{phase_dir}/rgb/{row['frame']}"
    # staged rgb frames are extracted straight from the HF tar, which uses
    # .webp (confirmed against the real tar members -- see report) -- try
    # that first, .png as a fallback for any manually-inspected leftovers.
    for ext in (".webp", ".png"):
        if os.path.exists(base + ext):
            return base + ext
    return base + ".webp"


def img_to_data_uri(img, fmt="JPEG", quality=80):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_target_with_bbox(row):
    path = target_frame_path(row)
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = row["answer_bbox"]
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
    return img


def main():
    rows = [json.loads(l) for l in open(ITEMS_PATH)]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    rng = random.Random(7)
    html_parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>In-context VQA audit gallery</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:24px}
h1{font-size:20px} h2{margin-top:40px;border-bottom:1px solid #444;padding-bottom:8px}
.row{display:flex;gap:16px;align-items:flex-start;margin:16px 0;padding:12px;background:#1b1b1b;border-radius:8px}
.row img{max-width:320px;max-height:280px;border:1px solid #333;border-radius:4px}
.col{display:flex;flex-direction:column;gap:4px}
.meta{font-size:12px;color:#aaa;max-width:520px}
.q{font-size:14px;color:#fff;margin:6px 0}
.label{font-size:11px;color:#8ab4f8;text-transform:uppercase;letter-spacing:.05em}
</style></head><body>
<h1>In-context VQA audit gallery -- scene001 dry run</h1>
<p class="meta">Image 1 (left) = deterministic SOS reference crop. Image 2 (right) = the real target frame with the ground-truth answer bbox drawn in red. Sampled deterministically (seed=7) from generated, validated rows.</p>
"""]

    for type_, items in sorted(by_type.items()):
        html_parts.append(f"<h2>{type_} ({len(items)} generated this scene)</h2>")
        sample = rng.sample(items, min(N_PER_TYPE, len(items)))
        for row in sample:
            try:
                ref_path = f"{REF_CROPS_DIR}/{row['reference_object_id']}.png"
                ref_img = Image.open(ref_path)
                target_img = render_target_with_bbox(row)
            except Exception as e:
                html_parts.append(f'<div class="row"><div class="meta">RENDER ERROR for {row["sample_id"]}: {e}</div></div>')
                continue
            ref_uri = img_to_data_uri(ref_img)
            target_uri = img_to_data_uri(target_img)
            evidence = row.get("evidence")
            html_parts.append(f"""
<div class="row">
  <div class="col"><span class="label">Image 1 (reference: {row['reference_object_id']})</span><img src="{ref_uri}"></div>
  <div class="col"><span class="label">Image 2 (target: {row['scene_id']}/{row['kind']}/{row['phase']}/{row['frame']}, answer: {row['answer']})</span><img src="{target_uri}"></div>
  <div class="col meta">
    <div class="q">{row['question']}</div>
    sample_id: {row['sample_id']}<br>
    source_type: {row['source_type']} | source_question: {row['source_question']}<br>
    target: {row['target_object_id']} (scid={row['target_source_catalog_id']}, goid={row['target_global_object_id']})<br>
    reference: {row['reference_object_id']} (scid={row['reference_source_catalog_id']}, goid={row['reference_global_object_id']})<br>
    attribute_kind={row.get('attribute_kind')} value={row.get('attribute_value')} material={row.get('attribute_material')} function={row.get('attribute_function')}<br>
    different_category={row.get('different_category')} | centered={row['answer_is_centered']}<br>
    evidence: {json.dumps(evidence) if evidence else '-'}
  </div>
</div>""")

    html_parts.append("</body></html>")
    with open(OUT_HTML, "w") as f:
        f.write("\n".join(html_parts))
    print(f"wrote {OUT_HTML} ({os.path.getsize(OUT_HTML)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
