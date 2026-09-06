#!/usr/bin/env python3
"""Build the deterministic one-row-per-cell VQA bbox acceptance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rpx_benchmark.vqa.contract import VQASample, write_manifest

GENERAL_TYPES = {
    "attr_single_color",
    "attr_single_material",
    "attr_single_function",
    "attr_composition",
}
SPATIAL_TYPES = {"spatial_lr_extreme", "depth_closest", "spatial_farthest"}


def rank(row: dict) -> str:
    identity = "\x1f".join(
        str(row.get(key, ""))
        for key in ("scene_id", "kind", "phase", "frame", "type", "question")
    )
    return hashlib.sha256(f"20260907:{identity}".encode()).hexdigest()


def cell(row: dict) -> tuple[str, str, int | None, str] | None:
    if row["type"] in GENERAL_TYPES and row.get("answer_bbox") is not None:
        phase = None if row["kind"] == "ego" else int(row["phase"])
        return ("general", row["kind"], phase, row["scene_id"])
    if row["type"] in SPATIAL_TYPES and row.get("answer_bbox") is not None:
        return ("spatial", row["kind"], int(row["phase"]), row["scene_id"])
    return None


def centered(row: dict) -> bool:
    x0, y0, x1, y1 = row["answer_bbox"]
    cx = (x0 + x1) / (2 * row["img_w"])
    cy = (y0 + y1) / (2 * row["img_h"])
    return 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required") from exc

    chosen: dict[tuple[str, str, int | None, str], tuple[str, dict]] = {}
    for filename in ("spatial_bbox.parquet", "attribute.parquet"):
        parquet = pq.ParquetFile(args.parquet_dir / filename)
        for batch in parquet.iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                key = cell(row)
                if key is None or not centered(row):
                    continue
                score = rank(row)
                if key not in chosen or score < chosen[key][0]:
                    chosen[key] = (score, row)

    counts = {"general": 0, "spatial": 0}
    for task, _, _, _ in chosen:
        counts[task] += 1
    if counts != {"general": 400, "spatial": 300}:
        raise SystemExit(f"incomplete acceptance coverage: {counts}")
    rows = [value[1] for _, value in sorted(chosen.items())]
    samples = [VQASample.from_dict(row) for row in rows]
    write_manifest(samples, args.out)
    print(f"wrote {len(samples)} acceptance rows: {json.dumps(counts, sort_keys=True)}")


if __name__ == "__main__":
    main()
