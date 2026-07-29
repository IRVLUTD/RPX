#!/usr/bin/env python3
"""Complete exact D1-F metrics from cached predictions, without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpx_benchmark.cell_log import cells_from_per_sample, write_cells
from rpx_benchmark.metrics.depth_paper import (
    FAST_PAPER_METRIC_KEYS,
    PAPER_METRIC_KEYS,
    PREDICTION_EVALUATION_POLICY,
)

PINNED_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
EXPECTED = {"easy": (24_750, 99), "medium": (24_750, 99), "hard": (25_500, 102)}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _cached_fscore(
    path: Path,
    sample_id: str,
    alignment_signature: str = "none",
) -> float | None:
    try:
        payload = json.loads(path.read_text())
        value = float(payload["fscore_5cm"])
        if payload.get("schema_version") != "rpx-d1f-fscore-v1":
            return None
        cached_signature = str(payload.get("alignment_signature", "none"))
        if cached_signature != alignment_signature:
            return None
        if payload.get("id") != sample_id or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _evaluate_one(work: tuple) -> tuple[str, float]:
    """Worker entry point: load one prediction/GT pair and compute exact F-score."""
    import numpy as np
    from PIL import Image

    from rpx_benchmark.metrics.depth_paper import compute_d1_paper_metrics

    if len(work) == 3:  # Backward-compatible DA-V2/fixture form.
        sample_id, prediction_path, gt_path = work
        alignment, scale, shift = "none", 1.0, 0.0
    else:
        sample_id, prediction_path, gt_path, alignment, scale, shift = work
    with np.load(prediction_path, allow_pickle=False) as payload:
        if payload.files != ["depth"]:
            raise ValueError(f"{prediction_path}: expected one 'depth' array")
        prediction = np.asarray(payload["depth"])
    if prediction.dtype != np.float32:
        raise ValueError(f"{prediction_path}: expected float32, got {prediction.dtype}")
    ground_truth = np.asarray(Image.open(gt_path), dtype=np.float32) / 1000.0
    if alignment == "ls_affine":
        prediction = (float(scale) * prediction + float(shift)).astype(np.float32)
    elif alignment == "ls_disparity":
        aligned_disparity = (
            float(scale) * prediction.astype(np.float64) + float(shift)
        )
        prediction = (
            1.0 / np.maximum(aligned_disparity, 1e-3)
        ).astype(np.float32)
    elif alignment == "ls_log":
        log_depth = np.minimum(
            float(scale) * prediction.astype(np.float64) + float(shift),
            5.0,
        )
        prediction = np.exp(log_depth).astype(np.float32)
    elif alignment != "none":
        raise ValueError(f"Deferred F-score does not support alignment {alignment!r}")
    metrics = compute_d1_paper_metrics(prediction, ground_truth)
    return sample_id, float(metrics["fscore_5cm"])


def _prediction_path(predictions: Path, sample: dict[str, object]) -> Path:
    scene = str(sample.get("scene_id") or "")
    phase = str(sample.get("phase") if sample.get("phase") is not None else "")
    frame = Path(str(sample.get("rgb") or sample["id"])).stem
    return predictions / scene / phase / f"{frame}.npz"


def _fscore_cache_path(root: Path, sample: dict[str, object]) -> Path:
    scene = str(sample.get("scene_id") or "")
    phase = str(sample.get("phase") if sample.get("phase") is not None else "")
    frame = Path(str(sample.get("rgb") or sample["id"])).stem
    return root / scene / phase / f"{frame}.json"


def _cell_key(sample: dict[str, object]) -> tuple[str, str]:
    phase = sample.get("phase")
    return str(sample.get("scene_id") or ""), str(phase if phase is not None else "")


def _alignment_parameters(
    samples: list[dict[str, object]],
    dataset_root: Path,
    predictions: Path,
    alignment: str,
) -> dict[tuple[str, str], tuple[float, float, str]]:
    """Compute one RPX pooled alignment transform per scene-phase cell."""
    if alignment == "none":
        return {}
    if alignment not in {"ls_affine", "ls_disparity", "ls_log"}:
        raise SystemExit(
            "Deferred exact F-score supports none, ls_affine, "
            f"ls_disparity and ls_log; got {alignment!r}"
        )

    import numpy as np
    from PIL import Image

    stats: dict[tuple[str, str], list[float]] = {}
    for index, sample in enumerate(samples, start=1):
        prediction_path = _prediction_path(predictions, sample)
        gt_path = dataset_root / str(sample["depth"])
        with np.load(prediction_path, allow_pickle=False) as payload:
            prediction = np.asarray(payload["depth"], dtype=np.float32)
        ground_truth = np.asarray(Image.open(gt_path), dtype=np.float32) / 1000.0
        valid = (
            np.isfinite(ground_truth)
            & (ground_truth > 0.3)
            & (ground_truth < 5.0)
            & np.isfinite(prediction)
            & (prediction > 0)
        )
        if not valid.any():
            continue
        x = prediction[valid].astype(np.float64)
        y = ground_truth[valid].astype(np.float64)
        if alignment == "ls_disparity":
            y = 1.0 / np.maximum(y, 1e-3)
        elif alignment == "ls_log":
            y = np.log(y)
        row = stats.setdefault(_cell_key(sample), [0.0] * 5)
        row[0] += float(x.size)
        row[1] += float(np.sum(x, dtype=np.float64))
        row[2] += float(np.sum(y, dtype=np.float64))
        row[3] += float(np.dot(x, x))
        row[4] += float(np.dot(x, y))
        if index % 1000 == 0 or index == len(samples):
            print(f"alignment pass: {index}/{len(samples)}", flush=True)

    parameters: dict[tuple[str, str], tuple[float, float, str]] = {}
    for key, (count, sum_x, sum_y, sum_xx, sum_xy) in stats.items():
        denominator = count * sum_xx - sum_x * sum_x
        if count < 2 or abs(denominator) <= 1e-12:
            raise SystemExit(f"Degenerate {alignment} alignment fit for cell {key}")
        scale = (count * sum_xy - sum_x * sum_y) / denominator
        shift = (sum_y - scale * sum_x) / count
        signature_payload = f"{alignment}:{scale:.17g}:{shift:.17g}"
        signature = hashlib.sha256(signature_payload.encode()).hexdigest()
        parameters[key] = (scale, shift, signature)
    expected_cells = len({_cell_key(sample) for sample in samples})
    if len(parameters) != expected_cells:
        raise SystemExit(
            f"Alignment produced {len(parameters)} cell transforms; expected {expected_cells}"
        )
    return parameters


def _complete_split(
    split: str,
    manifest_path: Path,
    output_root: Path,
    workers: int,
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = output_root / split
    predictions = out_dir / "predictions"
    fast_path = out_dir / "per_sample_metrics.parquet"
    if not fast_path.is_file():
        raise SystemExit(f"Missing fast metric table: {fast_path}")

    manifest = json.loads(manifest_path.read_text())
    samples = list(manifest.get("samples") or [])
    expected_samples, expected_cells = EXPECTED[split]
    if len(samples) != expected_samples:
        raise SystemExit(
            f"Manifest contract failed for {split}: {len(samples)} != {expected_samples}"
        )
    dataset_root = Path(str(manifest["root"]))
    metadata_path = out_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    required_metadata = {
        "dataset_revision": PINNED_REVISION,
        "paper_protocol": True,
        "num_samples": expected_samples,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise SystemExit(
                f"Run metadata contract failed for {split}: "
                f"{key}={metadata.get(key)!r}, expected {expected!r}"
            )
    if not metadata.get("model") or not metadata.get("model_checkpoint"):
        raise SystemExit(f"Run metadata for {split} lacks model/checkpoint provenance")
    alignment = str(metadata.get("alignment") or "none")
    rows = pq.read_table(fast_path).to_pylist()
    if len(rows) != expected_samples:
        raise SystemExit(f"Fast metric table for {split} has {len(rows)} rows")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != expected_samples:
        raise SystemExit(f"Fast metric table for {split} contains duplicate IDs")
    missing_fast = [key for key in FAST_PAPER_METRIC_KEYS if key not in rows[0]]
    if missing_fast:
        raise SystemExit(f"Fast metric table for {split} lacks {missing_fast}")

    cache_root = out_dir / "fscore_cache"
    parameters = _alignment_parameters(samples, dataset_root, predictions, alignment)
    fscore_by_id: dict[str, float] = {}
    pending: list[tuple[tuple, Path, str]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id not in by_id:
            raise SystemExit(f"Fast metric table for {split} lacks sample {sample_id}")
        prediction = _prediction_path(predictions, sample)
        gt_path = dataset_root / str(sample["depth"])
        if not prediction.is_file() or not gt_path.is_file():
            raise SystemExit(f"Missing prediction or GT for {sample_id}")
        cache_path = _fscore_cache_path(cache_root, sample)
        if alignment == "none":
            scale, shift, signature = 1.0, 0.0, "none"
        else:
            scale, shift, signature = parameters[_cell_key(sample)]
        cached = _cached_fscore(cache_path, sample_id, signature)
        if cached is None:
            pending.append(
                (
                    (
                        sample_id,
                        str(prediction),
                        str(gt_path),
                        alignment,
                        scale,
                        shift,
                    ),
                    cache_path,
                    signature,
                )
            )
        else:
            fscore_by_id[sample_id] = cached

    print(
        f"{split}: {len(fscore_by_id)} cached F-scores, "
        f"{len(pending)} pending; workers={workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_evaluate_one, work): (cache_path, signature)
                for work, cache_path, signature in pending
            }
            done = len(fscore_by_id)
            for future in as_completed(futures):
                sample_id, value = future.result()
                cache_path, signature = futures[future]
                _atomic_json(
                    cache_path,
                    {
                        "schema_version": "rpx-d1f-fscore-v1",
                        "id": sample_id,
                        "threshold_m": 0.05,
                        "fscore_5cm": value,
                        "alignment": alignment,
                        "alignment_signature": signature,
                    },
                )
                fscore_by_id[sample_id] = value
                done += 1
                if done % 25 == 0 or done == expected_samples:
                    print(f"{split}: exact F-score {done}/{expected_samples}", flush=True)

    for row in rows:
        row["fscore_5cm"] = fscore_by_id[str(row["id"])]
    if any(
        not math.isfinite(float(row[key]))
        for row in rows
        for key in PAPER_METRIC_KEYS
    ):
        raise SystemExit(f"Non-finite paper metric produced for {split}")

    old_cells_path = out_dir / "cells.parquet"
    old_cells = pq.read_table(old_cells_path).to_pylist()
    system_card = {
        key: old_cells[0].get(key)
        for key in ("gpu_name", "gpu_memory_gb", "precision", "batch_size")
    }
    cells = cells_from_per_sample(
        rows,
        model_name=str(old_cells[0].get("model_name") or "da-v2-large"),
        task="monocular_depth",
        metric_keys=PAPER_METRIC_KEYS,
        system_card=system_card,
    )
    if len(cells) != expected_cells:
        raise SystemExit(f"Exact evaluation for {split} produced {len(cells)} cells")

    per_sample_tmp = out_dir / ".per_sample_metrics.parquet.tmp"
    pq.write_table(pa.Table.from_pylist(rows), per_sample_tmp)
    os.replace(per_sample_tmp, fast_path)
    cells_tmp = out_dir / ".cells.parquet.tmp"
    write_cells(cells, cells_tmp, format="parquet")
    os.replace(cells_tmp, old_cells_path)

    aggregate = {
        key: sum(float(row[key]) for row in rows) / len(rows) for key in PAPER_METRIC_KEYS
    }
    _atomic_json(
        out_dir / "paper_metrics_result.json",
        {
            "schema_version": "rpx-d1f-exact-metrics-v1",
            "split": split,
            "num_samples": len(rows),
            "num_cells": len(cells),
            "metrics": aggregate,
            "fscore_workers": workers,
            "prediction_evaluation_policy": PREDICTION_EVALUATION_POLICY,
            "alignment": alignment,
        },
    )
    metadata["metrics"] = list(PAPER_METRIC_KEYS)
    metadata["fscore_status"] = "complete"
    metadata["fscore_workers"] = workers
    _atomic_json(metadata_path, metadata)
    return old_cells_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/huggingface"))
    parser.add_argument("--output-root", type=Path, default=Path("/outputs/da-v2-large"))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="parallel CPU processes; each cKDTree query remains exact and single-worker",
    )
    parser.add_argument("--jedi-bounds", type=Path)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    manifest_root = (
        args.cache_dir
        / "rpx-resolved"
        / "IRVLUTD__RPX"
        / "manifests"
        / "monocular_depth"
    )
    cells = [
        _complete_split(split, manifest_root / f"{split}.json", args.output_root, args.workers)
        for split in EXPECTED
    ]
    if not args.skip_analysis:
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "analyze_depth_paper.py"),
            *(str(path) for path in cells),
            "--output-dir",
            str(args.output_root / "paper"),
        ]
        if args.jedi_bounds:
            command.extend(("--jedi-bounds", str(args.jedi_bounds)))
        subprocess.run(command, check=True)
    print(f"Exact D1-F evaluation complete: {args.output_root}")


if __name__ == "__main__":
    main()
