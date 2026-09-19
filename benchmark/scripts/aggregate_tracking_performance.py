#!/usr/bin/env python3
"""Collect D3 accuracy and hardware measurements across tracking models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = ("hota", "deta", "assa", "idf1", "mota")
HARDWARE_COLUMNS = (
    "latency_ms",
    "throughput_fps",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
    "clip_wall_time_s",
    "parameter_count",
)


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _summary(model_dir: Path) -> dict[str, Any]:
    analysis_path = model_dir / "paper_analysis.json"
    cells_path = model_dir / "combined_cells.parquet"
    if not analysis_path.is_file() or not cells_path.is_file():
        raise FileNotFoundError(
            f"{model_dir.name}: expected paper_analysis.json and combined_cells.parquet"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    cells = pd.read_parquet(cells_path)
    if len(cells) != 300:
        raise ValueError(f"{model_dir.name}: expected 300 cells, found {len(cells)}")

    means = analysis["metric_means"]["overall"]
    result: dict[str, Any] = {
        "model": str(analysis["model"]),
        "cells": int(len(cells)),
        "frames": int(pd.to_numeric(cells["n_frames"]).sum()),
        **{metric: float(means[metric]) for metric in METRICS},
        "idsw_total": int(round(float(pd.to_numeric(cells["metric:idsw"]).sum()))),
        "idsw_mean_per_clip": float(means["idsw"]),
    }
    for column in HARDWARE_COLUMNS:
        values = _finite(cells[column]) if column in cells else np.asarray([])
        if column == "latency_ms":
            result["latency_ms_median"] = float(np.median(values)) if values.size else None
            result["latency_ms_p95"] = (
                float(np.percentile(values, 95)) if values.size else None
            )
        elif column == "throughput_fps":
            result["propagation_fps_median"] = (
                float(np.median(values)) if values.size else None
            )
        elif column == "peak_gpu_memory_allocated_mb":
            result["peak_gpu_memory_allocated_mb"] = (
                float(np.max(values)) if values.size else None
            )
        elif column == "peak_gpu_memory_reserved_mb":
            result["peak_gpu_memory_reserved_mb"] = (
                float(np.max(values)) if values.size else None
            )
        elif column == "clip_wall_time_s":
            result["inference_wall_time_s"] = (
                float(np.sum(values)) if values.size else None
            )
        elif column == "parameter_count":
            result["parameter_count"] = (
                int(round(float(np.max(values)))) if values.size else None
            )

    hardware_cards = []
    for split in ("easy", "medium", "hard"):
        result_path = model_dir / split / "result.json"
        if result_path.is_file():
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if value.get("hardware"):
                hardware_cards.append(value["hardware"])
    result["hardware"] = hardware_cards[0] if hardware_cards else None
    return result


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing one subdir/model")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = [_summary(root / model) for model in args.models]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tracking-performance.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).drop(columns=["hardware"]).to_csv(
        output / "tracking-performance.csv", index=False
    )

    lines = [
        "# RPX D3 tracking performance",
        "",
        "Accuracy values are fractions (1.0 is perfect). IDSw is summed across all 300 clips.",
        "",
        (
            "| Model | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | IDSw ↓ | "
            "Latency ms ↓ | FPS ↑ | Peak VRAM MiB ↓ | Params |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {_display(row['hota'])} | {_display(row['deta'])} | "
            f"{_display(row['assa'])} | {_display(row['idf1'])} | "
            f"{_display(row['mota'])} | {row['idsw_total']} | "
            f"{_display(row['latency_ms_median'])} | "
            f"{_display(row['propagation_fps_median'])} | "
            f"{_display(row['peak_gpu_memory_allocated_mb'], 1)} | "
            f"{row['parameter_count'] if row['parameter_count'] is not None else 'n/a'} |"
        )
    (output / "tracking-performance.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    outputs = (
        "tracking-performance.csv",
        "tracking-performance.json",
        "tracking-performance.md",
    )
    for name in outputs:
        print(output / name)


if __name__ == "__main__":
    main()
