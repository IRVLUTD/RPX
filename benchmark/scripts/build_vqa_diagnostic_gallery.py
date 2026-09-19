#!/usr/bin/env python3
"""Build a portable visual gallery for VQA decomposition diagnostics."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from build_vqa_acceptance_gallery import _native_candidates, _overlay, _to_data_uri
from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images
from rpx_benchmark.vqa.prompts import display_question


def _load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise SystemExit(f"duplicate diagnostic row at line {line_number}: {sample_id}")
            rows[sample_id] = row
    return rows


def _localization_figure(target: Image.Image, localization: dict, caption: str) -> str:
    if localization.get("localization_kind") == "native_point":
        rendered = target.copy()
        draw = ImageDraw.Draw(rendered)
        radius = max(5, round(min(rendered.size) / 80))
        for index, point in enumerate(localization.get("native_points_pixel") or [], 1):
            x, y = (float(value) for value in point)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(255, 0, 255),
                width=max(2, radius // 3),
            )
            draw.text((x + radius + 2, y - radius), f"P{index}", fill=(255, 0, 255))
        return (
            f'<figure><img src="{_to_data_uri(rendered)}">'
            f'<figcaption>{html.escape(caption)} — native Molmo points=P1…Pn '
            "(magenta; unscored)</figcaption></figure>"
        )
    strict_bbox = localization.get("strict_bbox") if localization.get("strict_valid") else None
    candidates = _native_candidates(
        localization.get("raw_output"),
        {"adapter_metadata": localization.get("adapter_metadata") or {}},
    )
    rendered = _overlay(target, None, strict_bbox, candidates)
    suffix = "parsed prediction=red"
    if candidates:
        suffix = "all native candidates=C1…Cn (multicolor; unscored)"
    return (
        f'<figure><img src="{_to_data_uri(rendered)}">'
        f'<figcaption>{html.escape(caption)} — {suffix}</figcaption></figure>'
    )


def _section(sample, row: dict, image_cache: Path) -> str:
    paths = fetch_images(sample, image_cache)
    figures: list[str] = []
    if sample.is_in_context:
        with Image.open(paths[0]) as image:
            figures.append(
                f'<figure><img src="{_to_data_uri(image)}">'
                "<figcaption>Image 1 — reference</figcaption></figure>"
            )
    with Image.open(paths[-1]) as image:
        target = image.convert("RGB")
    figures.append(
        f'<figure><img src="{_to_data_uri(_overlay(target, sample.answer_bbox, None))}">'
        "<figcaption>Target — ground truth=green</figcaption></figure>"
    )
    predicted = row["predicted_label_localization"]
    oracle = row["oracle_localization"]
    figures.append(_localization_figure(target, predicted, "Predicted-phrase grounding"))
    figures.append(_localization_figure(target, oracle, "Oracle-phrase grounding"))
    semantic = row["semantic"]
    end = row["end_to_end"]
    if oracle.get("localization_kind") == "native_point":
        localization_metrics = [
            "<dt>predicted phrase point inside GT</dt><dd>"
            f"{bool(predicted.get('any_point_inside_ground_truth_bbox', False))}</dd>",
            "<dt>predicted point normalized miss distance</dt><dd>"
            f"{predicted.get('best_normalized_miss_distance')}</dd>",
            "<dt>oracle point inside GT</dt><dd>"
            f"{bool(oracle.get('any_point_inside_ground_truth_bbox', False))}</dd>",
            "<dt>oracle point normalized miss distance</dt><dd>"
            f"{oracle.get('best_normalized_miss_distance')}</dd>",
        ]
    else:
        localization_metrics = [
            "<dt>predicted phrase IoU upper bound</dt><dd>"
            f'{float(predicted.get("best_iou", 0)):.3f}</dd>',
            "<dt>oracle IoU upper bound</dt><dd>"
            f'{float(oracle.get("best_iou", 0)):.3f}</dd>',
        ]
    return "\n".join(
        [
            '<section class="sample">',
            f'<h2>{html.escape(sample.sample_id)} <small>{html.escape(sample.question_type)}</small></h2>',
            f'<div class="images">{"".join(figures)}</div>',
            "<dl>",
            f'<dt>question</dt><dd>{html.escape(display_question(sample.question))}</dd>',
            f'<dt>ground truth</dt><dd>{html.escape(sample.answer)} {html.escape(str(sample.answer_bbox))}</dd>',
            f'<dt>semantic raw</dt><dd><code>{html.escape(str(semantic.get("raw_output", "")))}</code></dd>',
            f'<dt>semantic phrase</dt><dd>{html.escape(str(semantic.get("normalized_prediction", "")))}</dd>',
            *localization_metrics[:1],
            f'<dt>predicted grounding raw</dt><dd><code>{html.escape(str(predicted.get("raw_output", "")))}</code></dd>',
            *localization_metrics[1:],
            f'<dt>oracle grounding raw</dt><dd><code>{html.escape(str(oracle.get("raw_output", "")))}</code></dd>',
            f'<dt>end-to-end</dt><dd>{html.escape(str(end.get("parse_error") or end.get("source")))}</dd>',
            "</dl></section>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest_samples = load_manifest(args.manifest)
    rows = _load_rows(args.predictions)
    manifest_ids = {sample.sample_id for sample in manifest_samples}
    unknown = sorted(set(rows) - manifest_ids)
    if unknown:
        raise SystemExit(f"diagnostic predictions contain unknown sample IDs: {unknown[:10]}")
    samples = [sample for sample in manifest_samples if sample.sample_id in rows]
    if args.limit is not None:
        samples = samples[: args.limit]
    sections = [_section(sample, rows[sample.sample_id], args.image_cache) for sample in samples]
    style = """
    body { font-family: system-ui,sans-serif; background:#111; color:#eee; margin:0; padding:1.5rem; }
    .warning { background:#422; border:1px solid #e67e22; padding:1rem; border-radius:8px; }
    .sample { border:1px solid #333; border-radius:8px; padding:1rem; margin:1.5rem 0; }
    small { color:#7fb3ff; font-weight:400; } .images { display:flex; gap:1rem; flex-wrap:wrap; }
    figure { margin:0; } img { max-width:480px; border-radius:4px; display:block; }
    figcaption { color:#aaa; font-size:.8rem; margin-top:.25rem; }
    dl { display:grid; grid-template-columns:220px 1fr; gap:.35rem 1rem; }
    dt { color:#999; } code { color:#f1c40f; word-break:break-word; }
    """
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>VQA diagnostic — {html.escape(args.model)}</title><style>{style}</style></head><body>"
        f"<h1>VQA diagnostic — {html.escape(args.model)}</h1>"
        "<p class='warning'><strong>Not benchmark scores.</strong> Predicted-phrase grounding is a "
        "second model call. Oracle grounding discloses the GT label. Any displayed best IoU may "
        "select among native candidates using GT and is only an upper-bound diagnostic. Molmo "
        "native points are displayed and scored only by point-in-GT and normalized miss distance; "
        "they are never converted into boxes.</p>"
        + "\n".join(sections)
        + "</body></html>"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    print(f"wrote {len(samples)}-sample diagnostic gallery to {args.out}")


if __name__ == "__main__":
    main()
