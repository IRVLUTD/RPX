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

    spatial_by_frame: dict[tuple[str, int, str], tuple[str, dict]] = {}
    parquet = pq.ParquetFile(args.parquet_dir / "spatial_bbox.parquet")
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if row["type"] not in SPATIAL_TYPES or not centered(row):
                continue
            key = (row["scene_id"], int(row["phase"]), row["frame"])
            score = rank(row)
            if key not in spatial_by_frame or score < spatial_by_frame[key][0]:
                spatial_by_frame[key] = (score, row)

    paired: dict[tuple[str, int], tuple[tuple[str, str], dict, dict]] = {}
    ego: dict[str, tuple[str, dict]] = {}
    parquet = pq.ParquetFile(args.parquet_dir / "attribute.parquet")
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if (
                row["type"] not in GENERAL_TYPES
                or row.get("answer_bbox") is None
                or not centered(row)
            ):
                continue
            if row["kind"] == "ego":
                key = row["scene_id"]
                score = rank(row)
                if key not in ego or score < ego[key][0]:
                    ego[key] = (score, row)
                continue
            frame_key = (row["scene_id"], int(row["phase"]), row["frame"])
            if frame_key not in spatial_by_frame:
                continue
            cell_key = (row["scene_id"], int(row["phase"]))
            pair_score = (
                hashlib.sha256(f"20260907:{frame_key}".encode()).hexdigest(),
                rank(row),
            )
            if cell_key not in paired or pair_score < paired[cell_key][0]:
                paired[cell_key] = (pair_score, row, spatial_by_frame[frame_key][1])

    counts = {"general": len(paired) + len(ego), "spatial": len(paired)}
    if counts != {"general": 400, "spatial": 300}:
        raise SystemExit(f"incomplete acceptance coverage: {counts}")
    rows = []
    for _, (_, general_row, spatial_row) in sorted(paired.items()):
        rows.extend((general_row, spatial_row))
    rows.extend(value[1] for _, value in sorted(ego.items()))
    samples = [VQASample.from_dict(row) for row in rows]
    write_manifest(samples, args.out)
    image_count = len({json.dumps(sample.image, sort_keys=True) for sample in samples})
    if image_count != 400:
        raise SystemExit(f"acceptance image reuse invariant failed: {image_count}")
    print(
        f"wrote {len(samples)} acceptance rows over {image_count} images: "
        f"{json.dumps(counts, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
