#!/usr/bin/env python3
"""Run the complete resumable DA-V2 Large RPX D1-F paper workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Support direct source-tree execution as well as the installed Docker package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpx_benchmark.api import TaskType
from rpx_benchmark.hub import download_split
from rpx_benchmark.metrics.depth_paper import FAST_PAPER_METRIC_KEYS, PAPER_METRIC_KEYS

PINNED_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
EXPECTED = {
    "easy": (24_750, 99),
    "medium": (24_750, 99),
    "hard": (25_500, 102),
}
MODEL_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def _validate_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("CUDA preflight failed: torch is not installed") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA preflight failed: torch.cuda.is_available() is false")
    print(f"CUDA: {torch.cuda.get_device_name(0)}; torch={torch.__version__}")


def _validate_storage(path: Path, minimum_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < minimum_gb:
        raise SystemExit(
            f"Storage preflight failed: {path} has {free_gb:.1f} GiB free; "
            f"requires at least {minimum_gb:.1f} GiB"
        )
    print(f"Storage: {path} has {free_gb:.1f} GiB free")


def _prefetch(split: str, cache_dir: Path) -> Path:
    manifest = download_split(
        task=TaskType.MONOCULAR_DEPTH,
        split=split,
        repo_id="IRVLUTD/RPX",
        cache_dir=cache_dir,
        revision=PINNED_REVISION,
    )
    payload = json.loads(manifest.read_text())
    samples = payload.get("samples") or []
    cells = {(str(row.get("scene_id")), str(row.get("phase"))) for row in samples}
    expected_samples, expected_cells = EXPECTED[split]
    if len(samples) != expected_samples or len(cells) != expected_cells:
        raise SystemExit(
            f"Dataset contract failed for {split}: got {len(samples)} samples/{len(cells)} cells; "
            f"expected {expected_samples}/{expected_cells}"
        )
    root = Path(payload.get("root") or "")
    if not root.is_dir():
        raise SystemExit(f"Resolved manifest root is unavailable: {root}")
    print(f"Dataset {split}: {len(samples)} frames, {len(cells)} cells, root={root}")
    return manifest


def _run_tee(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        status = process.wait()
    if status != 0:
        raise SystemExit(f"Command failed with status {status}: {' '.join(command)}")


def _valid_latency_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        return (
            payload.get("schema_version") == "rpx-d1f-latency-v1"
            and payload.get("model_checkpoint") == MODEL_CHECKPOINT
            and payload.get("actual_torch_dtype") == "torch.float16"
            and payload.get("warmup_forwards") == 3
            and payload.get("measured_forwards") == 100
            and payload.get("prediction_writes") is False
            and float(payload.get("median_ms")) > 0.0
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _validate_split_outputs(
    out_dir: Path,
    expected_samples: int,
    expected_cells: int,
    metric_keys: tuple[str, ...] = PAPER_METRIC_KEYS,
) -> None:
    import pyarrow.parquet as pq

    per_sample = out_dir / "per_sample_metrics.parquet"
    cells_path = out_dir / "cells.parquet"
    if pq.read_metadata(per_sample).num_rows != expected_samples:
        raise SystemExit(f"Run contract failed for {out_dir.name}: wrong per-sample row count")
    cells = pq.read_table(cells_path)
    if cells.num_rows != expected_cells:
        raise SystemExit(f"Run contract failed for {out_dir.name}: wrong cell row count")
    metric_columns = {name for name in cells.column_names if name.startswith("metric:")}
    expected_metrics = {f"metric:{key}" for key in metric_keys}
    if metric_columns != expected_metrics:
        raise SystemExit(
            f"Run contract failed for {out_dir.name}: metric columns {sorted(metric_columns)}"
        )
    rows = cells.select(["scene_id", "phase"]).to_pylist()
    keys = {(str(row["scene_id"]), str(row["phase"])) for row in rows}
    if len(keys) != expected_cells:
        raise SystemExit(f"Run contract failed for {out_dir.name}: duplicate cells")


def _write_fast_phase_summary(cell_paths: list[Path], output_root: Path) -> None:
    """Merge deferred-F-score cells and report equal-scene phase means."""
    import csv

    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(path) for path in cell_paths]
    combined = pa.concat_tables(tables, promote_options="default")
    combined_path = output_root / "fast_combined_cells.parquet"
    pq.write_table(combined, combined_path)
    rows = combined.to_pylist()
    summary: list[dict[str, object]] = []
    for phase in ("clutter", "interaction", "clean"):
        phase_rows = [row for row in rows if str(row.get("phase")) == phase]
        if len(phase_rows) != 100:
            raise SystemExit(
                f"Fast phase summary expected 100 cells for {phase}; got {len(phase_rows)}"
            )
        entry: dict[str, object] = {"phase": phase, "n_scenes": len(phase_rows)}
        for key in FAST_PAPER_METRIC_KEYS:
            values = [float(row[f"metric:{key}"]) for row in phase_rows]
            entry[key] = sum(values) / len(values)
        summary.append(entry)
    json_path = output_root / "fast_phase_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = output_root / "fast_phase_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Deferred F-Score. Five-metric combined cells: {combined_path}")
    print(f"Phase-wise five-metric summary: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/huggingface"))
    parser.add_argument("--output-root", type=Path, default=Path("/outputs/da-v2-large"))
    parser.add_argument("--jedi-bounds", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--defer-fscore",
        action="store_true",
        help="finish predictions plus five paper metrics now; evaluate exact F-Score later",
    )
    parser.add_argument("--min-free-gb", type=float, default=120.0)
    args = parser.parse_args()

    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    _validate_cuda()
    _validate_storage(args.cache_dir, args.min_free_gb)
    _validate_storage(args.output_root, args.min_free_gb)

    env = dict(os.environ)
    env["HF_HOME"] = str(args.cache_dir)
    env["RPX_CACHE_DIR"] = str(args.cache_dir / "rpx-resolved")
    env["PYTHONUNBUFFERED"] = "1"
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"

    # download_split() runs in this launcher process before child commands.
    # Give it the same cache/offline contract as the split runs.
    os.environ["HF_HOME"] = env["HF_HOME"]
    os.environ["RPX_CACHE_DIR"] = env["RPX_CACHE_DIR"]
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    manifests = {split: _prefetch(split, args.cache_dir) for split in EXPECTED}
    script_dir = Path(__file__).resolve().parent
    cells: list[Path] = []
    for split, (expected_samples, expected_cells) in EXPECTED.items():
        out_dir = args.output_root / split
        command = [
            sys.executable,
            str(script_dir / "run_depth.py"),
            "--model",
            "da-v2-large",
            "--split",
            split,
            "--manifest-path",
            str(manifests[split]),
            "--revision",
            PINNED_REVISION,
            "--device",
            "cuda",
            "--precision",
            "fp16",
            "--batch-size",
            "1",
            "--skip-flops",
            "--save-predictions",
            "--resume-predictions",
            "--paper-protocol",
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            str(out_dir),
        ]
        if args.defer_fscore:
            command.append("--defer-fscore")
        _run_tee(command, out_dir / "run.log", env)
        metadata = json.loads((out_dir / "run_metadata.json").read_text())
        if int(metadata.get("num_samples") or 0) != expected_samples:
            raise SystemExit(f"Run contract failed for {split}: wrong sample count")
        if metadata.get("model_checkpoint") != MODEL_CHECKPOINT:
            raise SystemExit(f"Run contract failed for {split}: wrong checkpoint")
        if metadata.get("actual_torch_dtype") != "torch.float16":
            raise SystemExit(f"Run contract failed for {split}: model is not FP16")
        if metadata.get("alignment") != "none" or metadata.get("batch_size") != 1:
            raise SystemExit(f"Run contract failed for {split}: operating point drift")
        if metadata.get("paper_protocol") is not True:
            raise SystemExit(f"Run contract failed for {split}: paper protocol was not active")
        prediction_count = sum(1 for _ in (out_dir / "predictions").rglob("*.npz"))
        if prediction_count != expected_samples:
            raise SystemExit(
                f"Run contract failed for {split}: {prediction_count} predictions; "
                f"expected {expected_samples}"
            )
        expected_metric_keys = FAST_PAPER_METRIC_KEYS if args.defer_fscore else PAPER_METRIC_KEYS
        _validate_split_outputs(
            out_dir,
            expected_samples,
            expected_cells,
            expected_metric_keys,
        )
        cells.append(out_dir / "cells.parquet")

    if args.defer_fscore:
        _write_fast_phase_summary(cells, args.output_root)
        print(
            "Fast production stage complete. Run evaluate_depth_paper_predictions.py "
            "later to add exact F-Score and paper analysis."
        )
        return

    latency_path = args.output_root / "latency.json"
    if _valid_latency_file(latency_path):
        print(f"Latency: reusing validated {latency_path}")
    else:
        _run_tee(
            [
                sys.executable,
                str(script_dir / "measure_depth_latency.py"),
                "--manifest-path",
                str(manifests["easy"]),
                "--output",
                str(latency_path),
            ],
            args.output_root / "latency.log",
            env,
        )

    analysis_command = [
        sys.executable,
        str(script_dir / "analyze_depth_paper.py"),
        *(str(path) for path in cells),
        "--output-dir",
        str(args.output_root / "paper"),
    ]
    if args.jedi_bounds:
        analysis_command.extend(("--jedi-bounds", str(args.jedi_bounds)))
    _run_tee(analysis_command, args.output_root / "paper" / "analysis.log", env)
    print(f"Complete: {args.output_root}")


if __name__ == "__main__":
    main()
