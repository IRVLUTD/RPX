#!/usr/bin/env python3
"""Separate VQA semantic selection, oracle localization, and end-to-end errors."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from rpx_benchmark.vqa.contract import BBOX_TYPES, load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images_many
from rpx_benchmark.vqa.metrics import bbox_iou
from rpx_benchmark.vqa.outputs import normalize_label, parse_output
from rpx_benchmark.vqa.prompts import (
    build_oracle_localization_prompt,
    build_prompt,
    build_semantic_diagnostic_prompt,
)
from run_vllm_vqa import RemoteRunner
from vqa_models.vllm_backend import CHECKPOINTS


def comparable_label(value: str) -> str:
    value = normalize_label(value)
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    return value


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    samples = [s for s in load_manifest(args.manifest) if s.question_type in BBOX_TYPES]
    if args.limit is not None:
        samples = samples[: args.limit]
    image_paths = fetch_images_many(samples, args.image_cache)
    runner = RemoteRunner(args.server_url)
    health = runner.health()
    checkpoint = CHECKPOINTS[args.model]
    if (
        health.get("model") != args.model
        or health.get("checkpoint") != checkpoint.repo_id
        or health.get("revision") != checkpoint.revision
    ):
        raise SystemExit(f"resident engine provenance mismatch: {health}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            paths = image_paths[sample.sample_id]
            semantic_spec = build_semantic_diagnostic_prompt(sample, args.model)
            semantic_raw = runner.predict(
                paths,
                semantic_spec.text,
                semantic_spec.max_new_tokens,
                semantic_spec.output_kind,
            )
            semantic_ms = runner.latency_ms
            semantic_prediction = comparable_label(semantic_raw.splitlines()[0] if semantic_raw else "")
            expected_label = comparable_label(sample.answer)
            semantic_correct = semantic_prediction == expected_label

            oracle_spec = build_oracle_localization_prompt(sample, args.model)
            oracle_raw = runner.predict(
                [paths[-1]],
                oracle_spec.text,
                oracle_spec.max_new_tokens,
                oracle_spec.output_kind,
            )
            oracle_ms = runner.latency_ms
            oracle_parsed = parse_output(sample, oracle_raw, args.model)
            oracle_iou = (
                bbox_iou(oracle_parsed.bbox, sample.answer_bbox)
                if oracle_parsed.valid
                and oracle_parsed.bbox is not None
                and sample.answer_bbox is not None
                else 0.0
            )

            end_spec = build_prompt(sample, args.model)
            end_raw = runner.predict(
                paths, end_spec.text, end_spec.max_new_tokens, end_spec.output_kind
            )
            end_ms = runner.latency_ms
            end_parsed = parse_output(sample, end_raw, args.model)
            end_iou = (
                bbox_iou(end_parsed.bbox, sample.answer_bbox)
                if end_parsed.valid
                and end_parsed.bbox is not None
                and sample.answer_bbox is not None
                else 0.0
            )
            row = {
                "sample_id": sample.sample_id,
                "question_type": sample.question_type,
                "question": sample.question,
                "ground_truth_label": sample.answer,
                "ground_truth_bbox": sample.answer_bbox,
                "semantic": {
                    "raw_output": semantic_raw,
                    "normalized_prediction": semantic_prediction,
                    "normalized_ground_truth": expected_label,
                    "exact_match": semantic_correct,
                    "latency_ms": semantic_ms,
                },
                "oracle_localization": {
                    "raw_output": oracle_raw,
                    "valid": oracle_parsed.valid,
                    "parse_error": oracle_parsed.error,
                    "predicted_bbox": oracle_parsed.bbox,
                    "iou": oracle_iou,
                    "hit_at_0_5": oracle_iou >= 0.5,
                    "latency_ms": oracle_ms,
                    "ground_truth_label_disclosed": True,
                },
                "end_to_end": {
                    "raw_output": end_raw,
                    "valid": end_parsed.valid,
                    "parse_error": end_parsed.error,
                    "predicted_label": end_parsed.label,
                    "predicted_bbox": end_parsed.bbox,
                    "iou": end_iou,
                    "hit_at_0_5": end_iou >= 0.5,
                    "latency_ms": end_ms,
                },
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(samples)}] {sample.sample_id} "
                f"semantic={semantic_correct} oracle_iou={oracle_iou:.3f} "
                f"end_iou={end_iou:.3f}",
                flush=True,
            )

    groups = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)

    def aggregate(values: list[dict]) -> dict:
        return {
            "count": len(values),
            "semantic_exact_match": mean(
                [float(row["semantic"]["exact_match"]) for row in values]
            ),
            "oracle_parse_rate": mean(
                [float(row["oracle_localization"]["valid"]) for row in values]
            ),
            "oracle_bbox_mean_iou": mean(
                [row["oracle_localization"]["iou"] for row in values]
            ),
            "oracle_bbox_accuracy_at_0_5": mean(
                [float(row["oracle_localization"]["hit_at_0_5"]) for row in values]
            ),
            "end_to_end_parse_rate": mean(
                [float(row["end_to_end"]["valid"]) for row in values]
            ),
            "end_to_end_bbox_mean_iou": mean(
                [row["end_to_end"]["iou"] for row in values]
            ),
            "end_to_end_bbox_accuracy_at_0_5": mean(
                [float(row["end_to_end"]["hit_at_0_5"]) for row in values]
            ),
            "failure_attribution": {
                "semantic_wrong_oracle_miss": sum(
                    not row["semantic"]["exact_match"]
                    and not row["oracle_localization"]["hit_at_0_5"]
                    for row in values
                ),
                "semantic_wrong_oracle_hit": sum(
                    not row["semantic"]["exact_match"]
                    and row["oracle_localization"]["hit_at_0_5"]
                    for row in values
                ),
                "semantic_right_oracle_miss": sum(
                    row["semantic"]["exact_match"]
                    and not row["oracle_localization"]["hit_at_0_5"]
                    for row in values
                ),
                "semantic_right_oracle_hit": sum(
                    row["semantic"]["exact_match"]
                    and row["oracle_localization"]["hit_at_0_5"]
                    for row in values
                ),
            },
        }

    report = {
        "diagnostic_only": True,
        "warning": "Oracle localization discloses the ground-truth label and is not a benchmark score.",
        "model": args.model,
        "inference": health,
        "overall": aggregate(rows),
        "by_question_type": {
            key: aggregate(values) for key, values in sorted(groups.items())
        },
        "predictions": str(args.out),
    }
    report_path = args.out.with_name("diagnostic_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"diagnostic_report={report_path}")


if __name__ == "__main__":
    main()
