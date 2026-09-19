#!/usr/bin/env python3
"""Build a portable Markdown visual review from VQA smoke-test outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images


GT_COLOR = (46, 204, 113)
PRED_COLOR = (231, 76, 60)
MAX_WIDTH = 960


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _render(source: Path, destination: Path, gt_bbox, pred_bbox=None) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    scale = min(1.0, MAX_WIDTH / image.width)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))

    def scaled(box):
        return [round(float(value) * scale) for value in box]

    draw = ImageDraw.Draw(image)
    width = max(3, image.width // 250)
    if gt_bbox is not None:
        draw.rectangle(scaled(gt_bbox), outline=GT_COLOR, width=width)
    if pred_bbox is not None:
        draw.rectangle(scaled(pred_bbox), outline=PRED_COLOR, width=width)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=90)


def _copy_reference(source: Path, destination: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if image.width > MAX_WIDTH:
        scale = MAX_WIDTH / image.width
        image = image.resize((MAX_WIDTH, max(1, round(image.height * scale))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma4-12b", "internvl2.5-8b", "idefics3-8b", "paligemma2-10b"],
    )
    args = parser.parse_args()

    smoke_dirs = {
        model: args.bundle_root / "outputs" / model / f"sha-{args.sha}" / "smoke"
        for model in args.models
    }
    manifest_path = next(
        (directory / "manifest.jsonl" for directory in smoke_dirs.values() if (directory / "manifest.jsonl").is_file()),
        None,
    )
    if manifest_path is None:
        raise SystemExit("no smoke manifest found")

    samples = load_manifest(manifest_path)
    reports: dict[str, dict | None] = {}
    rows_by_model: dict[str, dict[str, dict]] = {}
    for model, directory in smoke_dirs.items():
        report_path = directory / "report.json"
        if not report_path.is_file():
            reports[model] = None
            rows_by_model[model] = {}
            continue
        report = _read_json(report_path)
        reports[model] = report
        rows_by_model[model] = {str(row["sample_id"]): row for row in report.get("samples", [])}

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    assets = args.out_dir / "assets"
    assets.mkdir(parents=True)
    lines = [
        "# VQA smoke-test visual review",
        "",
        f"Commit: `{args.sha}`  ",
        f"Samples: **{len(samples)}**  ",
        "Boxes: **green = ground truth**, **red = model prediction**.",
        "",
        "## Model summary",
        "",
        "| Model | Status | Parsed | Invalid | Mean IoU | Acc@0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in args.models:
        report = reports[model]
        if report is None:
            lines.append(f"| `{model}` | no report (infrastructure/access failure) | — | — | — | — |")
            continue
        rows = report.get("samples", [])
        valid = sum(row.get("valid") is True for row in rows)
        invalid = len(rows) - valid
        lines.append(
            f"| `{model}` | completed | {valid}/{len(rows)} | {invalid} | "
            f"{report.get('bbox_mean_iou', 0):.4f} | {report.get('bbox_accuracy_at_0_5', 0):.4f} |"
        )

    lines.extend(["", "## Samples", ""])
    for index, sample in enumerate(samples, start=1):
        image_paths = fetch_images(sample, args.image_cache)
        target_source = image_paths[1] if sample.is_in_context else image_paths[0]
        target_name = f"{index:02d}_{_safe_name(sample.sample_id)}_gt.jpg"
        _render(target_source, assets / target_name, sample.answer_bbox)

        lines.extend(
            [
                f"### {index}. `{sample.sample_id}` — `{sample.question_type}`",
                "",
                f"- Question: {sample.question}",
                f"- Answer: `{sample.answer}`",
                f"- GT bbox: `{list(sample.answer_bbox) if sample.answer_bbox is not None else None}`",
                f"- Scene/frame: `{sample.scene_id}` / `{sample.frame}`; kind=`{sample.kind}`; phase=`{sample.phase}`",
                "",
            ]
        )
        if sample.is_in_context:
            reference_name = f"{index:02d}_{_safe_name(sample.sample_id)}_reference.jpg"
            _copy_reference(image_paths[0], assets / reference_name)
            lines.extend(
                [
                    "Image 1 — verified SOS reference crop:",
                    "",
                    f"![reference crop](assets/{reference_name})",
                    "",
                    "Image 2 — target with GT:",
                    "",
                ]
            )
        else:
            lines.extend(["Target image with GT:", ""])
        lines.extend([f"![target with GT](assets/{target_name})", ""])

        for model in args.models:
            row = rows_by_model[model].get(sample.sample_id)
            lines.extend([f"#### `{model}`", ""])
            if row is None:
                lines.extend(["No prediction/report was produced.", ""])
                continue
            pred_bbox = row.get("predicted_bbox")
            overlay_name = f"{index:02d}_{_safe_name(sample.sample_id)}_{_safe_name(model)}.jpg"
            _render(target_source, assets / overlay_name, sample.answer_bbox, pred_bbox)
            lines.extend(
                [
                    f"- Valid: `{row.get('valid')}`; IoU: `{float(row.get('iou') or 0):.4f}`; latency: `{row.get('latency_ms')}` ms",
                    f"- Predicted bbox: `{pred_bbox}`",
                    f"- Parse error: `{row.get('parse_error')}`",
                    f"- Raw output: `{str(row.get('raw_output', '')).replace('`', chr(39))}`",
                    "",
                    f"![{model} overlay](assets/{overlay_name})",
                    "",
                ]
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "VQA_SMOKE_VISUAL_REVIEW.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path} with {len(samples)} samples")


if __name__ == "__main__":
    main()
