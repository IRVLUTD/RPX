#!/usr/bin/env python3
"""Audit RPX depth shards and smoke-test depth-estimation metrics.

Depth PNGs are stored as uint16 millimeters. This checker converts them to
meters, treats 0 and 65535 as invalid holes, and verifies that the benchmark
metric path behaves correctly on sampled ground-truth frames.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


DEPTH_MM_TO_M = 0.001
DEPTH_SATURATED_MM = np.iinfo(np.uint16).max
EPS = 1e-6


@dataclass(frozen=True)
class DepthUnit:
    kind: str
    unit_id: str
    depth_tar: Path
    expected_frames: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="RPX dataset repository root or downloaded local_dir.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("rerun_check/out_depth"))
    parser.add_argument(
        "--kinds",
        default="mos,ego",
        help="Comma-separated unit kinds to check: mos, sos, ego.",
    )
    parser.add_argument(
        "--sample-policy",
        choices=["middle", "endpoints", "all"],
        default="endpoints",
        help="'endpoints' samples first/middle/last frame per depth shard.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser.parse_args()


def repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def frame_indices_for_count(count: int, policy: str) -> list[int]:
    if count <= 0:
        return []
    if policy == "all":
        return list(range(count))
    if policy == "middle":
        return [count // 2]
    return sorted({0, count // 2, count - 1})


def depth_png_names(path: Path) -> list[str]:
    with tarfile.open(path) as tar:
        return sorted(
            member.name
            for member in tar.getmembers()
            if member.isfile()
            and member.name.startswith("depth/")
            and member.name.lower().endswith(".png")
        )


def read_depth_png(tar: tarfile.TarFile, name: str) -> tuple[np.ndarray, str, tuple[int, int]]:
    extracted = tar.extractfile(name)
    if extracted is None:
        raise ValueError(f"could not extract {name}")
    image = Image.open(io.BytesIO(extracted.read())).copy()
    arr = np.asarray(image)
    return arr, image.mode, image.size


def valid_depth_mask(depth_mm: np.ndarray) -> np.ndarray:
    return (depth_mm > 0) & (depth_mm < DEPTH_SATURATED_MM)


def depth_to_meters(depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_depth_mask(depth_mm)
    depth_m = depth_mm.astype(np.float32) * DEPTH_MM_TO_M
    depth_m[~valid] = 0.0
    return depth_m, valid


def depth_quality_stats(depth_mm: np.ndarray) -> dict[str, Any]:
    valid = valid_depth_mask(depth_mm)
    total = depth_mm.size
    stats: dict[str, Any] = {
        "dtype": str(depth_mm.dtype),
        "raw_min_mm": int(np.min(depth_mm)) if total else None,
        "raw_max_mm": int(np.max(depth_mm)) if total else None,
        "zero_fraction": float(np.count_nonzero(depth_mm == 0) / total) if total else 0.0,
        "saturated_fraction": float(np.count_nonzero(depth_mm == DEPTH_SATURATED_MM) / total) if total else 0.0,
        "valid_fraction": float(np.count_nonzero(valid) / total) if total else 0.0,
    }
    if valid.any():
        depth_m = depth_mm[valid].astype(np.float32) * DEPTH_MM_TO_M
        stats.update(
            {
                "valid_min_m": float(np.min(depth_m)),
                "valid_max_m": float(np.max(depth_m)),
                "valid_mean_m": float(np.mean(depth_m)),
                "valid_std_m": float(np.std(depth_m)),
                "valid_p01_m": float(np.percentile(depth_m, 1)),
                "valid_p50_m": float(np.percentile(depth_m, 50)),
                "valid_p99_m": float(np.percentile(depth_m, 99)),
            }
        )
    return stats


def depth_metrics(pred_m: np.ndarray, target_m: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    metric_mask = valid & np.isfinite(pred_m) & (pred_m > 0) & np.isfinite(target_m) & (target_m > 0)
    if not metric_mask.any():
        return {
            "valid_pixels": 0.0,
            "abs_rel": math.nan,
            "sq_rel": math.nan,
            "rmse": math.nan,
            "rmse_log": math.nan,
            "mae": math.nan,
            "log10": math.nan,
            "silog": math.nan,
            "delta1": math.nan,
            "delta2": math.nan,
            "delta3": math.nan,
        }

    pred = pred_m[metric_mask].astype(np.float64)
    target = target_m[metric_mask].astype(np.float64)
    diff = pred - target
    abs_diff = np.abs(diff)
    log_diff = np.log(pred + EPS) - np.log(target + EPS)
    ratio = np.maximum(pred / target, target / pred)
    return {
        "valid_pixels": float(pred.size),
        "abs_rel": float(np.mean(abs_diff / target)),
        "sq_rel": float(np.mean((diff**2) / target)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean(log_diff**2))),
        "mae": float(np.mean(abs_diff)),
        "log10": float(np.mean(np.abs(np.log10(pred + EPS) - np.log10(target + EPS)))),
        "silog": float(np.sqrt(max(np.mean(log_diff**2) - np.mean(log_diff) ** 2, 0.0)) * 100.0),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def aggregate_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None and math.isfinite(float(row[key]))]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def build_units(repo_root: Path, kinds: set[str]) -> list[DepthUnit]:
    units: list[DepthUnit] = []
    if "mos" in kinds:
        seen: set[tuple[str, str]] = set()
        for split in ("easy", "medium", "hard"):
            csv_path = repo_root / "splits" / f"{split}.csv"
            if not csv_path.exists():
                continue
            for row in read_csv_rows(csv_path):
                key = (row["scene_id"], row["phase_index"])
                if key in seen:
                    continue
                seen.add(key)
                units.append(
                    DepthUnit(
                        kind="mos",
                        unit_id=f"{row['scene_id']}.phase{row['phase_index']}",
                        depth_tar=repo_root / "scenes" / row["scene_id"] / row["phase_index"] / "depth.tar",
                        expected_frames=250,
                    )
                )
    if "sos" in kinds:
        csv_path = repo_root / "manifest" / "selected_sos_objects_v1.csv"
        if csv_path.exists():
            for row in read_csv_rows(csv_path):
                units.append(
                    DepthUnit(
                        kind="sos",
                        unit_id=f"object:{row['object_id']}",
                        depth_tar=repo_root / "objects" / row["object_id"] / "0" / "depth.tar",
                        expected_frames=500,
                    )
                )
    if "ego" in kinds:
        for path in sorted((repo_root / "scenes").glob("scene*/ego/depth.tar")):
            units.append(
                DepthUnit(
                    kind="ego",
                    unit_id=f"{path.parents[1].name}.ego",
                    depth_tar=path,
                    expected_frames=None,
                )
            )
    return units


def check_unit(repo_root: Path, unit: DepthUnit, policy: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sampled: list[dict[str, Any]] = []
    identity_metrics: list[dict[str, float]] = []
    if not unit.depth_tar.exists():
        return {
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "depth_tar": as_rel(unit.depth_tar, repo_root),
            "status": "error",
            "errors": [f"missing depth shard: {unit.depth_tar}"],
            "warnings": warnings,
            "frame_count": 0,
            "sampled": sampled,
        }
    try:
        names = depth_png_names(unit.depth_tar)
        if unit.expected_frames is not None and len(names) != unit.expected_frames:
            errors.append(f"depth frame count {len(names)}, expected {unit.expected_frames}")
        if not names:
            errors.append("depth shard has no PNG frames")
    except Exception as exc:
        return {
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "depth_tar": as_rel(unit.depth_tar, repo_root),
            "status": "error",
            "errors": [f"could not list depth shard: {exc}"],
            "warnings": warnings,
            "frame_count": 0,
            "sampled": sampled,
        }

    sample_names = [names[index] for index in frame_indices_for_count(len(names), policy)]
    try:
        with tarfile.open(unit.depth_tar) as tar:
            for name in sample_names:
                try:
                    depth_mm, mode, size = read_depth_png(tar, name)
                except Exception as exc:
                    errors.append(f"could not decode {name}: {exc}")
                    continue
                stats = depth_quality_stats(depth_mm)
                stats.update({"name": name, "mode": mode, "size": list(size)})
                if mode != "I;16" or str(depth_mm.dtype) != "uint16":
                    warnings.append(f"{name} is {mode}/{depth_mm.dtype}, expected I;16/uint16")
                if stats["valid_fraction"] <= 0:
                    errors.append(f"{name} has no valid metric depth pixels")
                target_m, valid = depth_to_meters(depth_mm)
                identity = depth_metrics(target_m.copy(), target_m, valid)
                identity_metrics.append(identity)
                sampled.append(stats)
    except (tarfile.TarError, OSError) as exc:
        errors.append(f"could not read samples from depth shard: {exc}")

    metric_keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "mae", "log10", "silog", "delta1", "delta2", "delta3"]
    metric_summary = aggregate_numeric(identity_metrics, metric_keys)
    if identity_metrics:
        finite_error_values = [
            max(identity.get("abs_rel", 0.0), identity.get("rmse", 0.0), identity.get("mae", 0.0))
            for identity in identity_metrics
            if math.isfinite(identity.get("abs_rel", math.nan))
        ]
        finite_delta_gaps = [
            1.0 - identity.get("delta1", 0.0)
            for identity in identity_metrics
            if math.isfinite(identity.get("delta1", math.nan))
        ]
        worst_error = max(finite_error_values) if finite_error_values else math.inf
        worst_delta_gap = max(finite_delta_gaps) if finite_delta_gaps else math.inf
        if math.isinf(worst_error) or math.isinf(worst_delta_gap):
            errors.append("identity depth metric smoke test had no valid pixels")
        elif worst_error > 1e-8 or worst_delta_gap > 1e-8:
            errors.append("identity depth metric smoke test failed")
    status = "error" if errors else ("warning" if warnings else "ok")
    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "depth_tar": as_rel(unit.depth_tar, repo_root),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "frame_count": len(names),
        "sampled": sampled,
        "identity_metric_summary": metric_summary,
    }


def write_markdown_summary(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# RPX Depth Benchmark QA Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Units checked: `{summary['units_checked']}`",
        f"- Frames sampled: `{summary['frames_sampled']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        "- Ground-truth depth source: `uint16 millimeters`",
        "- Benchmark depth unit: `meters`",
        "- Invalid/hole values: `0` and `65535`",
        "",
        "## Depth Quality",
        "",
    ]
    for key, value in summary["depth_quality"].items():
        lines.append(f"- `{key}` min/mean/max: `{value['min']:.6g}` / `{value['mean']:.6g}` / `{value['max']:.6g}`")
    lines.extend(["", "## Identity Metric Smoke", ""])
    for key, value in summary["identity_metrics"].items():
        lines.append(f"- `{key}` min/mean/max: `{value['min']:.6g}` / `{value['mean']:.6g}` / `{value['max']:.6g}`")
    if summary["errors_by_unit"]:
        lines.extend(["", "## Error Examples", ""])
        for item in summary["errors_by_unit"][:20]:
            lines.append(f"- `{item['unit_id']}`: {item['errors'][0]}")
    if summary["warnings_by_unit"]:
        lines.extend(["", "## Warning Examples", ""])
        for item in summary["warnings_by_unit"][:20]:
            lines.append(f"- `{item['unit_id']}`: {item['warnings'][0]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = {item.strip() for item in args.kinds.split(",") if item.strip()}
    invalid_kinds = kinds - {"mos", "sos", "ego"}
    if invalid_kinds:
        raise SystemExit(f"invalid --kinds values: {sorted(invalid_kinds)}")

    units = build_units(repo_root, kinds)
    if args.limit is not None:
        units = units[: args.limit]
    results = [check_unit(repo_root, unit, args.sample_policy) for unit in tqdm(units, desc="checking depth")]
    errors_by_unit = [{"unit_id": item["unit_id"], "errors": item["errors"]} for item in results if item["errors"]]
    warnings_by_unit = [{"unit_id": item["unit_id"], "warnings": item["warnings"]} for item in results if item["warnings"]]
    error_count = sum(len(item["errors"]) for item in results)
    warning_count = sum(len(item["warnings"]) for item in results)
    quality_rows = [sample for result in results for sample in result["sampled"]]
    metric_rows = [
        metric
        for result in results
        for metric in [
            {key: value["mean"] for key, value in result.get("identity_metric_summary", {}).items()}
        ]
        if metric
    ]
    summary = {
        "repo_root": str(repo_root),
        "sample_policy": args.sample_policy,
        "status": "error" if error_count else ("warning" if warning_count else "ok"),
        "units_checked": len(results),
        "unit_counts": dict(Counter(item["kind"] for item in results)),
        "frames_sampled": len(quality_rows),
        "error_count": error_count,
        "warning_count": warning_count,
        "depth_quality": aggregate_numeric(
            quality_rows,
            [
                "zero_fraction",
                "saturated_fraction",
                "valid_fraction",
                "valid_min_m",
                "valid_max_m",
                "valid_mean_m",
                "valid_std_m",
                "valid_p01_m",
                "valid_p50_m",
                "valid_p99_m",
            ],
        ),
        "identity_metrics": aggregate_numeric(
            metric_rows,
            ["abs_rel", "sq_rel", "rmse", "rmse_log", "mae", "log10", "silog", "delta1", "delta2", "delta3"],
        ),
        "errors_by_unit": errors_by_unit,
        "warnings_by_unit": warnings_by_unit,
        "results": results,
    }
    report_path = out_dir / "depth_benchmark_report.json"
    markdown_path = out_dir / "depth_benchmark_summary.md"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, markdown_path)
    print(f"status: {summary['status']}")
    print(f"units checked: {len(results)}")
    print(f"frames sampled: {len(quality_rows)}")
    print(f"errors: {error_count}")
    print(f"warnings: {warning_count}")
    print(f"wrote {report_path}")
    print(f"wrote {markdown_path}")
    if error_count or (warning_count and args.fail_on_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
