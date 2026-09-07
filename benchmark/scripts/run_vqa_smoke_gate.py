#!/usr/bin/env python3
"""Parse model JSONL outputs, score them, and apply mechanical smoke gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rpx_benchmark.vqa.contract import ATTRIBUTE_TYPES, BBOX_TYPES, load_manifest
from rpx_benchmark.vqa.metrics import bbox_iou, score_predictions
from rpx_benchmark.vqa.outputs import parse_output
from rpx_benchmark.vqa.roster import get_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-parse-rate", type=float, default=0.95)
    args = parser.parse_args()
    model = get_model(args.model)
    samples = {sample.sample_id: sample for sample in load_manifest(args.manifest)}
    raw_by_id = {}
    latency_by_id = {}
    provenance_by_id = {}
    with args.predictions.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in raw_by_id:
                raise SystemExit(f"duplicate prediction at line {line_number}: {sample_id}")
            if row.get("backend") != "vllm":
                raise SystemExit(
                    f"non-vLLM prediction at line {line_number}: {row.get('backend')!r}"
                )
            if row.get("model") != model.key:
                raise SystemExit(
                    f"model mismatch at line {line_number}: {row.get('model')!r}"
                )
            raw_by_id[sample_id] = str(row["raw_output"])
            latency_by_id[sample_id] = float(row["latency_ms"])
            provenance_by_id[sample_id] = {
                key: row.get(key)
                for key in ("backend", "vllm_version", "checkpoint", "revision")
            }
    expected = {
        sample_id: sample
        for sample_id, sample in samples.items()
        if (
            "bbox"
            if sample.question_type in BBOX_TYPES
            else "attribute"
            if sample.question_type in ATTRIBUTE_TYPES
            else "binary"
        )
        in model.capabilities
    }
    missing = sorted(set(expected) - set(raw_by_id))
    extra = sorted(set(raw_by_id) - set(expected))
    if missing or extra:
        raise SystemExit(f"prediction coverage mismatch: missing={missing}, extra={extra}")
    parsed_by_id = {
        sample_id: parse_output(sample, raw_by_id[sample_id], model.key)
        for sample_id, sample in expected.items()
    }
    report = score_predictions(
        (sample, parsed_by_id[sample_id]) for sample_id, sample in expected.items()
    )
    provenance_values = {json.dumps(value, sort_keys=True) for value in provenance_by_id.values()}
    if len(provenance_values) != 1:
        raise SystemExit("prediction provenance changed within one run")
    report["inference"] = next(iter(provenance_by_id.values()))
    latencies = [latency_by_id[sample_id] for sample_id in expected]
    report["latency_ms"] = {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }
    report["samples"] = []
    for sample_id, sample in expected.items():
        parsed = parsed_by_id[sample_id]
        iou = 0.0
        if parsed.valid and parsed.bbox is not None and sample.answer_bbox is not None:
            iou = bbox_iou(parsed.bbox, sample.answer_bbox)
        report["samples"].append(
            {
                "sample_id": sample_id,
                "question": sample.question,
                "question_type": sample.question_type,
                "kind": sample.kind,
                "phase": sample.phase,
                "ground_truth_bbox": sample.answer_bbox,
                "raw_output": raw_by_id[sample_id],
                "predicted_bbox": parsed.bbox,
                "valid": parsed.valid,
                "parse_error": parsed.error,
                "iou": iou,
                "latency_ms": latency_by_id[sample_id],
            }
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if report["parse_rate"] < args.min_parse_rate:
        raise SystemExit(
            f"FAIL: parse_rate {report['parse_rate']:.3f} < {args.min_parse_rate:.3f}"
        )


if __name__ == "__main__":
    main()
