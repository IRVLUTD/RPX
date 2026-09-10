#!/usr/bin/env python3
"""Separate VQA semantic selection, oracle localization, and end-to-end errors."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from run_vllm_vqa import RemoteRunner
from vqa_models.backend_registry import CHECKPOINTS

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
from rpx_benchmark.vqa.roster import get_model


def comparable_label(value: str) -> str:
    value = normalize_label(value)
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    return value


def extract_answer_phrase(raw: str) -> dict:
    """Extract at most five object-name words and audit format compliance."""
    text = raw.strip()
    format_errors: list[str] = []
    if text.startswith("```"):
        format_errors.append("markdown_fence")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            format_errors.append("structured_output_instead_of_keywords")
            for key in ("label", "answer", "object"):
                if value.get(key):
                    text = str(value[key])
                    break
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        format_errors.append("expected_one_nonempty_line")
    text = lines[0] if lines else ""
    stripped = re.sub(
        r"^(?:the\s+)?(?:answer|object)(?:\s+object)?\s+is\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"^(?:it|this)\s+is\s+", "", stripped, flags=re.IGNORECASE
    )
    if stripped != text:
        format_errors.append("sentence_wrapper")
    phrase = stripped.strip("` \t\"'.,:;!?()[]{}")
    if phrase != stripped:
        format_errors.append("surrounding_punctuation")
    without_article = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    if without_article != phrase:
        format_errors.append("leading_article")
        phrase = without_article
    words = phrase.split()
    if not 1 <= len(words) <= 5:
        format_errors.append("expected_1_to_5_words")
        phrase = ""
    elif not re.fullmatch(r"[\w/&+-]+(?:\s+[\w/&+-]+){0,4}", phrase):
        format_errors.append("non_keyword_punctuation")
        phrase = ""
    return {
        "phrase": phrase,
        "format_valid": not format_errors,
        "format_errors": format_errors,
    }


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
    strict = parse_output(
        sample, raw, model_key, paligemma_target_only=True
    )
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
        # Florence may correctly expose several regions for a non-unique noun
        # phrase. Retain every region and compute a GT-selected upper bound,
        # but never turn that selection into a scored prediction.
        raw_candidates = None if value is None else value.get("candidates")
        if isinstance(raw_candidates, list):
            for index, candidate in enumerate(raw_candidates):
                candidate_bbox = (
                    candidate.get("bbox") if isinstance(candidate, dict) else None
                )
                if not (
                    isinstance(candidate_bbox, list)
                    and len(candidate_bbox) == 4
                    and all(isinstance(v, (int, float)) for v in candidate_bbox)
                ):
                    continue
                x0, y0, x1, y1 = (float(v) for v in candidate_bbox)
                if not (0 <= x0 <= x1 <= 1000 and 0 <= y0 <= y1 <= 1000):
                    continue
                normalized_bbox = (
                    x0 * (sample.img_w - 1) / 1000,
                    y0 * (sample.img_h - 1) / 1000,
                    x1 * (sample.img_w - 1) / 1000,
                    y1 * (sample.img_h - 1) / 1000,
                )
                hypotheses[f"native_candidate_{index + 1}_gt_selected"] = {
                    "bbox": normalized_bbox,
                    "iou": bbox_iou(normalized_bbox, sample.answer_bbox),
                    "label": str(candidate.get("label") or ""),
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
        metadata = row.get("adapter_metadata") or {}
        if metadata.get("single_scored_model_call") is not True:
            raise SystemExit(
                f"acceptance row {sample.sample_id} is not a direct single-call "
                "prediction; rerun acceptance with the current image"
            )
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
    model = get_model(args.model)
    direct_supported = "bbox" in model.capabilities
    if args.acceptance_report and not direct_supported:
        raise SystemExit(
            f"{args.model} has no direct one-stage bbox acceptance protocol; "
            "do not reuse its historical acceptance report"
        )

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
            semantic_metadata = runner.prediction_metadata()
            semantic_extraction = extract_answer_phrase(semantic_raw)
            semantic_prediction = semantic_extraction["phrase"]
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
                predicted_metadata = runner.prediction_metadata()
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
            oracle_metadata = runner.prediction_metadata()
            oracle_grounding = diagnostic_bbox(sample, oracle_raw, args.model)

            if not direct_supported:
                end_raw = ""
                end_ms = None
                end_source = "unsupported_by_model"
                end_parsed = None
                end_iou = None
            elif acceptance_rows is None:
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
            if direct_supported:
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
                    "keyword_format_valid": semantic_extraction["format_valid"],
                    "keyword_format_errors": semantic_extraction["format_errors"],
                    "latency_ms": semantic_ms,
                    "adapter_metadata": semantic_metadata,
                },
                "predicted_label_localization": {
                    "label_used": semantic_prediction,
                    "raw_output": predicted_raw,
                    "latency_ms": predicted_ms,
                    "adapter_metadata": (
                        predicted_metadata if semantic_prediction else {}
                    ),
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
                    "adapter_metadata": oracle_metadata,
                    "ground_truth_label_disclosed": True,
                    **oracle_grounding,
                },
                "end_to_end": {
                    "supported": direct_supported,
                    "source": end_source,
                    "raw_output": end_raw,
                    "valid": None if end_parsed is None else end_parsed.valid,
                    "parse_error": (
                        "unsupported by the model adapter"
                        if end_parsed is None
                        else end_parsed.error
                    ),
                    "predicted_label": None if end_parsed is None else end_parsed.label,
                    "predicted_bbox": None if end_parsed is None else end_parsed.bbox,
                    "iou": end_iou,
                    "hit_at_0_5": None if end_iou is None else end_iou >= 0.5,
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
                f"end_iou={'unsupported' if end_iou is None else f'{end_iou:.3f}'}",
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
            "semantic_keyword_format_rate": mean(
                [float(row["semantic"]["keyword_format_valid"]) for row in values]
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
            "end_to_end_supported": direct_supported,
            "end_to_end_parse_rate": (
                mean([float(row["end_to_end"]["valid"]) for row in values])
                if direct_supported else None
            ),
            "end_to_end_bbox_mean_iou": (
                mean([row["end_to_end"]["iou"] for row in values])
                if direct_supported else None
            ),
            "end_to_end_bbox_accuracy_at_0_5": (
                mean([float(row["end_to_end"]["hit_at_0_5"]) for row in values])
                if direct_supported else None
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
            "uses GT to compare known conventions and ambiguous native "
            "candidates. Neither is a benchmark score."
        ),
        "model": args.model,
        "inference": health,
        "scored_path_contract": (
            "one model generation receives the original question and all required "
            "images and returns the final bbox; any generated label is informational, "
            "and diagnostic answer/oracle calls are unscored and never replace that "
            "prediction"
        ),
        "scored_path_supported": direct_supported,
        "end_to_end_source": (
            str(args.acceptance_report)
            if args.acceptance_report
            else "new_model_calls" if direct_supported else "unsupported_by_model"
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
