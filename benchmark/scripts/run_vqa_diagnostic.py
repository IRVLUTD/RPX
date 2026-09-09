#!/usr/bin/env python3
"""Separate VQA semantic selection, oracle localization, and end-to-end errors."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rpx_benchmark.vqa.contract import BBOX_TYPES, load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images_many
from rpx_benchmark.vqa.metrics import bbox_iou
from rpx_benchmark.vqa.outputs import normalize_label, parse_output
from rpx_benchmark.vqa.prompts import (
    build_label_localization_prompt,
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


def answer_phrase(raw: str) -> str:
    """Extract a short referring phrase without mapping it to a GT label."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            for key in ("label", "answer", "object"):
                if value.get(key):
                    text = str(value[key])
                    break
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    text = re.sub(
        r"^(?:the\s+)?(?:answer|object)(?:\s+object)?\s+is\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip("` \t\"'.,:;!?()[]{}")[:160]


def _loose_bbox_object(raw: str) -> dict | None:
    """Read JSON or a Python-literal dict for diagnostic analysis only."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    payload = text[start : end + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(payload)
        except (SyntaxError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def diagnostic_bbox(sample, raw: str, model_key: str) -> dict:
    """Score protocol-strict and format-tolerant bbox interpretations.

    `best_*` deliberately uses GT to select between known coordinate
    conventions and is therefore diagnostic-only. The official acceptance
    report remains untouched and protocol-strict.
    """
    strict = parse_output(sample, raw, model_key)
    hypotheses: dict[str, dict] = {}
    if strict.valid and strict.bbox is not None:
        iou = bbox_iou(strict.bbox, sample.answer_bbox)
        hypotheses[strict.coordinate_format or "native_model_format"] = {
            "bbox": strict.bbox,
            "iou": iou,
        }
    if not model_key.startswith("paligemma2-"):
        value = _loose_bbox_object(raw)
        raw_bbox = None if value is None else value.get("bbox")
        if (
            isinstance(raw_bbox, list)
            and len(raw_bbox) == 1
            and isinstance(raw_bbox[0], list)
        ):
            raw_bbox = raw_bbox[0]
        try:
            bbox = tuple(float(v) for v in raw_bbox) if len(raw_bbox) == 4 else None
        except (TypeError, ValueError):
            bbox = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if (
                0 <= x0 <= x1 <= 1000
                and 0 <= y0 <= y1 <= 1000
            ):
                denominator = 1 if all(0 <= v <= 1 for v in bbox) else 1000
                name = (
                    "normalized_0_1_loose_syntax"
                    if denominator == 1
                    else "normalized_0_1000_loose_syntax"
                )
                normalized_bbox = (
                    x0 * (sample.img_w - 1) / denominator,
                    y0 * (sample.img_h - 1) / denominator,
                    x1 * (sample.img_w - 1) / denominator,
                    y1 * (sample.img_h - 1) / denominator,
                )
                hypotheses.setdefault(
                    name,
                    {
                        "bbox": normalized_bbox,
                        "iou": bbox_iou(normalized_bbox, sample.answer_bbox),
                    },
                )
            if 0 <= x0 <= x1 < sample.img_w and 0 <= y0 <= y1 < sample.img_h:
                hypotheses["original_pixel_xyxy"] = {
                    "bbox": bbox,
                    "iou": bbox_iou(bbox, sample.answer_bbox),
                }
    best_format = max(
        hypotheses, key=lambda key: hypotheses[key]["iou"], default=None
    )
    best = hypotheses.get(best_format) if best_format else None
    return {
        "strict_valid": strict.valid,
        "strict_parse_error": strict.error,
        "strict_bbox": strict.bbox,
        "strict_coordinate_format": strict.coordinate_format,
        "hypotheses": hypotheses,
        "best_coordinate_format": best_format,
        "best_bbox": None if best is None else best["bbox"],
        "best_iou": 0.0 if best is None else best["iou"],
        "best_hit_at_0_5": bool(best is not None and best["iou"] >= 0.5),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_acceptance_rows(
    path: Path, samples: list, health: dict, model_key: str
) -> dict[str, dict]:
    """Load a completed acceptance report for reuse by the diagnostic.

    Reusing these rows avoids spending a third model call on an end-to-end
    prediction that was already measured.  Fail closed on provenance or sample
    mismatches so a stale report cannot silently contaminate the attribution.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    inference = report.get("inference") or {}
    if inference.get("model") not in (None, model_key):
        raise SystemExit(
            "acceptance report provenance mismatch for model: "
            f"report={inference.get('model')!r}, expected={model_key!r}"
        )
    for key in ("backend", "checkpoint", "revision"):
        if inference.get(key) != health.get(key):
            raise SystemExit(
                f"acceptance report provenance mismatch for {key}: "
                f"report={inference.get(key)!r}, resident={health.get(key)!r}"
            )
    rows: dict[str, dict] = {}
    for row in report.get("samples") or []:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in rows:
            raise SystemExit(f"invalid or duplicate acceptance sample_id: {sample_id!r}")
        rows[sample_id] = row
    for sample in samples:
        row = rows.get(sample.sample_id)
        if row is None:
            raise SystemExit(f"acceptance report is missing sample {sample.sample_id}")
        if row.get("question_type") != sample.question_type:
            raise SystemExit(f"acceptance question type mismatch for {sample.sample_id}")
        if row.get("ground_truth_bbox") != list(sample.answer_bbox or []):
            raise SystemExit(f"acceptance GT bbox mismatch for {sample.sample_id}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        help="reuse completed end-to-end rows instead of running them again",
    )
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
    acceptance_rows = (
        load_acceptance_rows(args.acceptance_report, samples, health, args.model)
        if args.acceptance_report
        else None
    )

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
            semantic_prediction = answer_phrase(semantic_raw)
            expected_label = comparable_label(sample.answer)
            semantic_exact_match = comparable_label(semantic_prediction) == expected_label

            predicted_grounding = None
            predicted_ms = 0.0
            predicted_raw = ""
            if semantic_prediction:
                predicted_spec = build_label_localization_prompt(
                    semantic_prediction, args.model
                )
                predicted_raw = runner.predict(
                    [paths[-1]],
                    predicted_spec.text,
                    predicted_spec.max_new_tokens,
                    predicted_spec.output_kind,
                )
                predicted_ms = runner.latency_ms
                predicted_grounding = diagnostic_bbox(
                    sample, predicted_raw, args.model
                )

            oracle_spec = build_oracle_localization_prompt(sample, args.model)
            oracle_raw = runner.predict(
                [paths[-1]],
                oracle_spec.text,
                oracle_spec.max_new_tokens,
                oracle_spec.output_kind,
            )
            oracle_ms = runner.latency_ms
            oracle_grounding = diagnostic_bbox(sample, oracle_raw, args.model)

            if acceptance_rows is None:
                end_spec = build_prompt(sample, args.model)
                end_raw = runner.predict(
                    paths, end_spec.text, end_spec.max_new_tokens, end_spec.output_kind
                )
                end_ms = runner.latency_ms
                end_source = "new_model_call"
            else:
                prior = acceptance_rows[sample.sample_id]
                end_raw = str(prior.get("raw_output") or "")
                end_ms = float(prior.get("latency_ms") or 0.0)
                end_source = str(args.acceptance_report)
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
                    "exact_match_informational_only": semantic_exact_match,
                    "latency_ms": semantic_ms,
                },
                "predicted_label_localization": {
                    "label_used": semantic_prediction,
                    "raw_output": predicted_raw,
                    "latency_ms": predicted_ms,
                    **(predicted_grounding or {
                        "strict_valid": False,
                        "strict_parse_error": "empty semantic answer",
                        "strict_bbox": None,
                        "strict_coordinate_format": None,
                        "hypotheses": {},
                        "best_coordinate_format": None,
                        "best_bbox": None,
                        "best_iou": 0.0,
                        "best_hit_at_0_5": False,
                    }),
                },
                "oracle_localization": {
                    "raw_output": oracle_raw,
                    "latency_ms": oracle_ms,
                    "ground_truth_label_disclosed": True,
                    **oracle_grounding,
                },
                "end_to_end": {
                    "source": end_source,
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
                f"label={semantic_prediction!r} "
                f"predicted_label_iou={row['predicted_label_localization']['best_iou']:.3f} "
                f"oracle_iou={oracle_grounding['best_iou']:.3f} "
                f"end_iou={end_iou:.3f}",
                flush=True,
            )

    groups = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)

    def aggregate(values: list[dict]) -> dict:
        oracle_hits = sum(
            row["oracle_localization"]["best_hit_at_0_5"] for row in values
        )
        joint_hits = sum(
            row["predicted_label_localization"]["best_hit_at_0_5"]
            and row["oracle_localization"]["best_hit_at_0_5"]
            for row in values
        )
        return {
            "count": len(values),
            "semantic_exact_match_informational_only": mean(
                [
                    float(row["semantic"]["exact_match_informational_only"])
                    for row in values
                ]
            ),
            "oracle_parse_rate": mean(
                [float(row["oracle_localization"]["strict_valid"]) for row in values]
            ),
            "oracle_bbox_mean_iou": mean(
                [row["oracle_localization"]["best_iou"] for row in values]
            ),
            "oracle_bbox_accuracy_at_0_5": mean(
                [float(row["oracle_localization"]["best_hit_at_0_5"]) for row in values]
            ),
            "predicted_label_bbox_mean_iou": mean(
                [row["predicted_label_localization"]["best_iou"] for row in values]
            ),
            "predicted_label_bbox_accuracy_at_0_5": mean(
                [
                    float(row["predicted_label_localization"]["best_hit_at_0_5"])
                    for row in values
                ]
            ),
            "identity_accuracy_when_oracle_localizable": (
                joint_hits / oracle_hits if oracle_hits else None
            ),
            "oracle_best_coordinate_formats": dict(
                sorted(
                    Counter(
                        row["oracle_localization"]["best_coordinate_format"]
                        or "unparseable"
                        for row in values
                    ).items()
                )
            ),
            "predicted_label_best_coordinate_formats": dict(
                sorted(
                    Counter(
                        row["predicted_label_localization"]["best_coordinate_format"]
                        or "unparseable"
                        for row in values
                    ).items()
                )
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
            "identity_attribution": {
                "identified_and_localizable": sum(
                    row["predicted_label_localization"]["best_hit_at_0_5"]
                    and row["oracle_localization"]["best_hit_at_0_5"]
                    for row in values
                ),
                "selection_failed_but_oracle_localizable": sum(
                    not row["predicted_label_localization"]["best_hit_at_0_5"]
                    and row["oracle_localization"]["best_hit_at_0_5"]
                    for row in values
                ),
                "predicted_phrase_hit_despite_oracle_label_miss": sum(
                    row["predicted_label_localization"]["best_hit_at_0_5"]
                    and not row["oracle_localization"]["best_hit_at_0_5"]
                    for row in values
                ),
                "both_localizations_missed": sum(
                    not row["predicted_label_localization"]["best_hit_at_0_5"]
                    and not row["oracle_localization"]["best_hit_at_0_5"]
                    for row in values
                ),
            },
        }

    report = {
        "diagnostic_only": True,
        "warning": (
            "All localization hypotheses and oracle results are diagnostic-only. "
            "Oracle localization discloses the GT label; best-coordinate scoring "
            "uses GT to compare known conventions. Neither is a benchmark score."
        ),
        "model": args.model,
        "inference": health,
        "end_to_end_source": (
            str(args.acceptance_report)
            if args.acceptance_report
            else "new_model_calls"
        ),
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
