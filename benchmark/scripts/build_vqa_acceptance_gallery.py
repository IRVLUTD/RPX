#!/usr/bin/env python3
"""Build a single self-contained HTML visual audit gallery for one model's
acceptance-gate run. For every sample: Image 1 + Image 2 (in-context) or
just the target image (normal), the question, the raw model output, the
parsed prediction, the ground-truth bbox, a prediction/GT overlay drawn on
the target image, and IoU / validity / parse-error / latency. Thumbnails are
embedded as base64 data URIs so the single .html file is portable -- no
external image server or relative asset paths required.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images
from rpx_benchmark.vqa.metrics import bbox_iou
from rpx_benchmark.vqa.outputs import parse_output
from rpx_benchmark.vqa.prompts import display_question

THUMB_MAX_W = 480
GT_COLOR = (46, 204, 113)  # green
PRED_COLOR = (231, 76, 60)  # red


def _to_data_uri(image: Image.Image) -> str:
    if image.width > THUMB_MAX_W:
        scale = THUMB_MAX_W / image.width
        image = image.resize((THUMB_MAX_W, max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _overlay(target: Image.Image, gt_bbox, pred_bbox) -> Image.Image:
    canvas = target.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if gt_bbox is not None:
        draw.rectangle(list(gt_bbox), outline=GT_COLOR, width=max(2, canvas.width // 200))
    if pred_bbox is not None:
        draw.rectangle([round(v) for v in pred_bbox], outline=PRED_COLOR, width=max(2, canvas.width // 200))
    return canvas


def _load_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def _sample_section(sample, pred_row: dict | None, model: str, image_cache: Path) -> str:
    parts: list[str] = []
    raw = pred_row["raw_output"] if pred_row else None
    latency = pred_row.get("latency_ms", pred_row.get("amortized_latency_ms")) if pred_row else None
    parsed = parse_output(sample, raw, model) if raw is not None else None
    iou = 0.0
    if parsed is not None and parsed.valid and parsed.bbox is not None and sample.answer_bbox is not None:
        iou = bbox_iou(parsed.bbox, sample.answer_bbox)

    image_paths = fetch_images(sample, image_cache)
    images_html = []
    if sample.is_in_context:
        ref_image = Image.open(image_paths[0])
        images_html.append(f'<figure><img src="{_to_data_uri(ref_image)}"><figcaption>Image 1 (reference)</figcaption></figure>')
        target_image = Image.open(image_paths[1])
    else:
        target_image = Image.open(image_paths[0])
    overlay = _overlay(target_image, sample.answer_bbox, parsed.bbox if parsed else None)
    caption = "Image 2 (target), GT=green pred=red" if sample.is_in_context else "target, GT=green pred=red"
    images_html.append(f'<figure><img src="{_to_data_uri(overlay)}"><figcaption>{caption}</figcaption></figure>')

    status = "no-prediction" if raw is None else ("valid" if parsed.valid else "parse-error")
    parts.append(f'<section class="sample {status}">')
    parts.append(f'<h3>{html.escape(sample.sample_id)} <span class="type">{html.escape(sample.question_type)}</span></h3>')
    parts.append(f'<div class="images">{"".join(images_html)}</div>')
    parts.append('<dl>')
    parts.append(f'<dt>question</dt><dd>{html.escape(display_question(sample.question))}</dd>')
    parts.append(f'<dt>raw output</dt><dd><code>{html.escape(raw) if raw is not None else "(missing)"}</code></dd>')
    parts.append(f'<dt>parsed prediction</dt><dd>{html.escape(str(parsed.label)) if parsed else "-"} '
                 f'{html.escape(str(parsed.bbox)) if parsed and parsed.bbox else ""}</dd>')
    parts.append(f'<dt>ground truth</dt><dd>{html.escape(sample.answer)} {html.escape(str(sample.answer_bbox))}</dd>')
    parts.append(f'<dt>IoU</dt><dd>{iou:.3f}</dd>')
    parts.append(f'<dt>valid</dt><dd>{parsed.valid if parsed else False}</dd>')
    parts.append(f'<dt>parse error</dt><dd>{html.escape(parsed.error) if parsed and parsed.error else "-"}</dd>')
    parts.append(f'<dt>latency (ms)</dt><dd>{latency if latency is not None else "-"}</dd>')
    parts.append('</dl></section>')
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="render only the first N samples (debugging)")
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]
    predictions = _load_predictions(args.predictions)

    sections = [
        _sample_section(sample, predictions.get(sample.sample_id), args.model, args.image_cache)
        for sample in samples
    ]

    style = """
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 1.5rem; }
    h1 { font-weight: 600; }
    section.sample { border: 1px solid #333; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
    section.sample.parse-error { border-color: #e74c3c; }
    section.sample.no-prediction { border-color: #666; opacity: 0.6; }
    .type { color: #7fb3ff; font-weight: 400; font-size: 0.8em; }
    .images { display: flex; gap: 1rem; flex-wrap: wrap; }
    .images figure { margin: 0; }
    .images img { max-width: 480px; border-radius: 4px; display: block; }
    figcaption { font-size: 0.8em; color: #aaa; margin-top: 0.25rem; }
    dl { display: grid; grid-template-columns: 160px 1fr; gap: 0.25rem 1rem; margin-top: 0.75rem; }
    dt { color: #999; }
    code { color: #f1c40f; word-break: break-word; }
    """
    html_doc = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>VQA acceptance gallery: {html.escape(args.model)}</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>VQA acceptance gallery -- {html.escape(args.model)}</h1>"
        f"<p>{len(samples)} samples, manifest={html.escape(str(args.manifest))}</p>"
        + "\n".join(sections)
        + "</body></html>"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {len(samples)}-sample gallery to {args.out}")


if __name__ == "__main__":
    main()
