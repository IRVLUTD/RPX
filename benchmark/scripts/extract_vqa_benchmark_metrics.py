#!/usr/bin/env python3
"""Rescore completed RPX VQA benchmark runs without rerunning inference.

The command discovers a complete shard set for each requested model, reparses
the retained raw outputs with the current model adapter, and writes comparable
JSON, CSV, and Markdown summaries. Terminal inference failures remain in the
denominator as invalid predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rpx_benchmark.vqa.contract import BBOX_TYPES, VQASample, load_manifest
from rpx_benchmark.vqa.metrics import score_predictions
from rpx_benchmark.vqa.outputs import ParsedOutput, parse_output
from rpx_benchmark.vqa.roster import get_model

SHARD_RE = re.compile(r"shard-(?P<index>[0-9]+)-of-(?P<count>[0-9]+)$")
SUMMARY_FIELDS = (
    "model",
    "rows",
    "predictions",
    "inference_failures",
    "bbox_validity_rate",
    "bbox_accuracy_at_0_25",
    "bbox_accuracy_at_0_5",
    "bbox_mean_giou",
    "bbox_center_in_gt_accuracy",
    "bbox_mean_iou",
    "run_dir",
)
SCENE_FIELDS = (
    "model",
    "scene",
    "kind",
    "phase",
    "capture",
    "rows",
    "predictions",
    "inference_failures",
    "bbox_validity_rate",
    "bbox_accuracy_at_0_25",
    "bbox_accuracy_at_0_5",
    "bbox_accuracy_at_0_75",
    "bbox_mean_accuracy_50_95",
    "bbox_mean_giou",
    "bbox_mean_giou_valid",
    "bbox_center_in_gt_accuracy",
    "bbox_mean_iou",
    "parse_rate",
)


@dataclass(frozen=True)
class RunCandidate:
    benchmark_dir: Path
    shard_count: int
    shard_dirs: tuple[Path, ...]
    predictions: dict[str, dict]
    failures: dict[str, dict]
    modified: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def collect_rows(shard_dirs: tuple[Path, ...], filename: str) -> dict[str, dict]:
    collected: dict[str, dict] = {}
    for shard_dir in shard_dirs:
        for row in read_jsonl(shard_dir / filename):
            sample_id = str(row["sample_id"])
            if sample_id in collected:
                raise SystemExit(
                    f"duplicate {sample_id} across {filename} files under "
                    f"{shard_dir.parent}"
                )
            collected[sample_id] = row
    return collected


def candidate_groups(benchmark_dir: Path) -> list[tuple[int, tuple[Path, ...]]]:
    grouped: dict[int, dict[int, Path]] = {}
    for path in benchmark_dir.glob("shard-*-of-*"):
        if not path.is_dir():
            continue
        match = SHARD_RE.fullmatch(path.name)
        if not match:
            continue
        index = int(match.group("index"))
        count = int(match.group("count"))
        if not 0 <= index < count:
            continue
        grouped.setdefault(count, {})[index] = path
    return [
        (count, tuple(by_index[index] for index in range(count)))
        for count, by_index in grouped.items()
        if set(by_index) == set(range(count))
    ]


def newest_mtime(paths: tuple[Path, ...]) -> float:
    files = [
        shard_dir / filename
        for shard_dir in paths
        for filename in ("predictions.jsonl", "failures.jsonl", "report.json")
        if (shard_dir / filename).exists()
    ]
    return max((path.stat().st_mtime for path in files), default=0.0)


def discover_complete_run(
    outputs_root: Path,
    model: str,
    expected_ids: set[str],
    manifest_sha256: str,
) -> RunCandidate:
    base = outputs_root / model
    candidates: list[RunCandidate] = []
    diagnostics: list[str] = []
    for benchmark_dir in sorted(base.glob("sha-*/benchmark")):
        for shard_count, shard_dirs in candidate_groups(benchmark_dir):
            config_mismatch = False
            for shard_dir in shard_dirs:
                config_path = shard_dir / "run_config.json"
                if config_path.is_file():
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    recorded_sha = config.get("manifest_sha256")
                    if recorded_sha and recorded_sha != manifest_sha256:
                        config_mismatch = True
                        break
            if config_mismatch:
                diagnostics.append(
                    f"{benchmark_dir} ({shard_count} shards): manifest SHA mismatch"
                )
                continue
            predictions = collect_rows(shard_dirs, "predictions.jsonl")
            failures = collect_rows(shard_dirs, "failures.jsonl")
            overlap = set(predictions) & set(failures)
            if overlap:
                raise SystemExit(
                    f"{benchmark_dir} contains IDs in predictions and failures: "
                    f"{sorted(overlap)[:5]}"
                )
            covered = set(predictions) | set(failures)
            missing = expected_ids - covered
            extra = covered - expected_ids
            diagnostics.append(
                f"{benchmark_dir} ({shard_count} shards): predictions={len(predictions)}, "
                f"failures={len(failures)}, missing={len(missing)}, extra={len(extra)}"
            )
            if not missing and not extra:
                candidates.append(
                    RunCandidate(
                        benchmark_dir=benchmark_dir,
                        shard_count=shard_count,
                        shard_dirs=shard_dirs,
                        predictions=predictions,
                        failures=failures,
                        modified=newest_mtime(shard_dirs),
                    )
                )
    if not candidates:
        detail = "\n".join(f"  - {line}" for line in diagnostics) or "  - no runs found"
        raise SystemExit(f"no complete run found for {model} under {base}:\n{detail}")
    selected = max(candidates, key=lambda item: item.modified)
    print(
        f"selected {model}: {selected.benchmark_dir} "
        f"({selected.shard_count} shards, {len(selected.predictions)} predictions, "
        f"{len(selected.failures)} failures)",
        flush=True,
    )
    return selected


def score_subset(
    samples: list[VQASample], parsed_by_id: dict[str, ParsedOutput]
) -> dict:
    return score_predictions(
        (sample, parsed_by_id[sample.sample_id]) for sample in samples
    )


def score_model(
    model: str,
    candidate: RunCandidate,
    samples: list[VQASample],
) -> dict:
    get_model(model)
    parsed_by_id: dict[str, ParsedOutput] = {}
    provenance: set[str] = set()
    for sample in samples:
        row = candidate.predictions.get(sample.sample_id)
        if row is None:
            failure = candidate.failures[sample.sample_id]
            parsed_by_id[sample.sample_id] = ParsedOutput(
                False,
                error=f"inference failure: {failure.get('error', 'unknown error')}",
            )
            continue
        if row.get("model") != model:
            raise SystemExit(
                f"model mismatch for {sample.sample_id}: {row.get('model')!r} != {model!r}"
            )
        provenance.add(
            json.dumps(
                {
                    key: row.get(key)
                    for key in ("backend", "backend_version", "checkpoint", "revision")
                },
                sort_keys=True,
            )
        )
        parsed_by_id[sample.sample_id] = parse_output(
            sample, str(row["raw_output"]), model
        )
    if len(provenance) > 1:
        raise SystemExit(f"provenance changed within selected {model} run")

    bbox_samples = [sample for sample in samples if sample.question_type in BBOX_TYPES]
    metrics = score_subset(bbox_samples, parsed_by_id)
    metrics_by_context = {
        name: score_subset(subset, parsed_by_id)
        for name, subset in (
            ("regular", [sample for sample in bbox_samples if not sample.is_in_context]),
            ("in_context", [sample for sample in bbox_samples if sample.is_in_context]),
        )
        if subset
    }
    metrics_by_capture = {
        name: score_subset(subset, parsed_by_id)
        for name, subset in (
            ("mos_phase_0", [s for s in bbox_samples if s.kind == "mos" and s.phase == 0]),
            ("mos_phase_1", [s for s in bbox_samples if s.kind == "mos" and s.phase == 1]),
            ("mos_phase_2", [s for s in bbox_samples if s.kind == "mos" and s.phase == 2]),
            ("ego", [s for s in bbox_samples if s.kind == "ego"]),
        )
        if subset
    }
    grouped_by_scene: dict[tuple[str, str, int | None], list[VQASample]] = defaultdict(list)
    for sample in bbox_samples:
        grouped_by_scene[(sample.scene_id, sample.kind, sample.phase)].append(sample)
    metrics_by_scene = []
    for (scene_id, kind, phase), subset in sorted(
        grouped_by_scene.items(),
        key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2]),
    ):
        metrics_by_scene.append(
            {
                "scene": scene_id,
                "kind": kind,
                "phase": phase,
                "capture": "ego" if kind == "ego" else f"mos_phase_{phase}",
                "rows": len(subset),
                "predictions": sum(
                    sample.sample_id in candidate.predictions for sample in subset
                ),
                "inference_failures": sum(
                    sample.sample_id in candidate.failures for sample in subset
                ),
                "metrics": score_subset(subset, parsed_by_id),
            }
        )
    return {
        "model": model,
        "run_dir": str(candidate.benchmark_dir),
        "shard_count": candidate.shard_count,
        "rows": len(bbox_samples),
        "predictions": len(candidate.predictions),
        "inference_failures": len(candidate.failures),
        "inference": json.loads(next(iter(provenance))) if provenance else None,
        "metrics": metrics,
        "metrics_by_context": metrics_by_context,
        "metrics_by_capture": metrics_by_capture,
        "metrics_by_scene": metrics_by_scene,
    }


def summary_row(result: dict) -> dict:
    metrics = result["metrics"]
    return {
        "model": result["model"],
        "rows": result["rows"],
        "predictions": result["predictions"],
        "inference_failures": result["inference_failures"],
        "bbox_validity_rate": metrics["bbox_validity_rate"],
        "bbox_accuracy_at_0_25": metrics["bbox_accuracy_at_0_25"],
        "bbox_accuracy_at_0_5": metrics["bbox_accuracy_at_0_5"],
        "bbox_mean_giou": metrics["bbox_mean_giou"],
        "bbox_center_in_gt_accuracy": metrics["bbox_center_in_gt_accuracy"],
        "bbox_mean_iou": metrics["bbox_mean_iou"],
        "run_dir": result["run_dir"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def scene_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for scene in result["metrics_by_scene"]:
            metrics = scene["metrics"]
            rows.append(
                {
                    "model": result["model"],
                    "scene": scene["scene"],
                    "kind": scene["kind"],
                    "phase": "" if scene["phase"] is None else scene["phase"],
                    "capture": scene["capture"],
                    "rows": scene["rows"],
                    "predictions": scene["predictions"],
                    "inference_failures": scene["inference_failures"],
                    "bbox_validity_rate": metrics["bbox_validity_rate"],
                    "bbox_accuracy_at_0_25": metrics["bbox_accuracy_at_0_25"],
                    "bbox_accuracy_at_0_5": metrics["bbox_accuracy_at_0_5"],
                    "bbox_accuracy_at_0_75": metrics["bbox_accuracy_at_0_75"],
                    "bbox_mean_accuracy_50_95": metrics[
                        "bbox_mean_accuracy_50_95"
                    ],
                    "bbox_mean_giou": metrics["bbox_mean_giou"],
                    "bbox_mean_giou_valid": metrics["bbox_mean_giou_valid"],
                    "bbox_center_in_gt_accuracy": metrics[
                        "bbox_center_in_gt_accuracy"
                    ],
                    "bbox_mean_iou": metrics["bbox_mean_iou"],
                    "parse_rate": metrics["parse_rate"],
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict], manifest_sha256: str) -> None:
    lines = [
        "# RPX VQA full-benchmark localization metrics",
        "",
        f"Manifest SHA-256: `{manifest_sha256}`",
        "",
        "| Model | Rows | Predictions | Infra failures | BBox valid | Acc@.25 | Acc@.50 | Mean GIoU | Center in GT | Mean IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row['rows']} | {row['predictions']} | "
            f"{row['inference_failures']} | {row['bbox_validity_rate']:.4f} | "
            f"{row['bbox_accuracy_at_0_25']:.4f} | "
            f"{row['bbox_accuracy_at_0_5']:.4f} | {row['bbox_mean_giou']:.4f} | "
            f"{row['bbox_center_in_gt_accuracy']:.4f} | "
            f"{row['bbox_mean_iou']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- `BBox valid`: parsed, finite, ordered inclusive-XYXY box wholly inside the target image.",
            "- `Acc@.25` and `Acc@.50`: fraction of all bbox rows with IoU greater than or equal to the threshold.",
            "- `Mean GIoU`: all-row mean; invalid/missing boxes receive -1.0.",
            "- `Center in GT`: predicted-box center lies inside the inclusive GT box; invalid/missing boxes are misses.",
            "- Infrastructure failures remain in every denominator.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    sample_ids = {sample.sample_id for sample in samples}
    if len(sample_ids) != len(samples):
        raise SystemExit("manifest contains duplicate sample IDs")
    manifest_sha256 = file_sha256(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model in args.model:
        candidate = discover_complete_run(
            args.outputs_root, model, sample_ids, manifest_sha256
        )
        results.append(score_model(model, candidate, samples))

    report = {
        "schema_version": "rpx-vqa-localization-metrics-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_rows": len(samples),
        "models": results,
    }
    summary = [summary_row(result) for result in results]
    json_path = args.out_dir / "vqa_localization_metrics.json"
    csv_path = args.out_dir / "vqa_localization_metrics.csv"
    markdown_path = args.out_dir / "vqa_localization_metrics.md"
    scene_path = args.out_dir / "vqa_scene_metrics.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary)
    with scene_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCENE_FIELDS)
        writer.writeheader()
        writer.writerows(scene_rows(results))
    write_markdown(markdown_path, summary, manifest_sha256)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {scene_path}")


if __name__ == "__main__":
    main()
