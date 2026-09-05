#!/usr/bin/env python3
"""Build the deterministic RPX VQA smoke manifest from the benchmark parquets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from rpx_benchmark.vqa.contract import VQASample, write_manifest

QUOTAS = {
    "spatial_lr_binary:yes": 4,
    "spatial_lr_binary:no": 4,
    "spatial_ud_binary:yes": 4,
    "spatial_ud_binary:no": 4,
    "spatial_lr_extreme": 4,
    "depth_closest": 4,
    "spatial_farthest": 4,
    "attr_composition:mos": 4,
    "attr_composition:ego": 4,
}


def _rank(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _stratum(row: dict) -> str | None:
    qtype = row["type"]
    if qtype in {"spatial_lr_binary", "spatial_ud_binary"}:
        return f"{qtype}:{row['answer']}"
    if qtype == "attr_composition":
        return f"{qtype}:{row['kind']}"
    if qtype in {"spatial_lr_extreme", "depth_closest", "spatial_farthest"}:
        return qtype
    return None


def _unambiguous_bbox(row: dict) -> bool:
    evidence = json.loads(row["evidence"])
    qtype = row["type"]
    if qtype == "spatial_farthest":
        values = sorted(evidence["distances_2d_px"].values(), reverse=True)
        return len(values) > 1 and values[0] - values[1] > 1.0
    values = sorted(float(value) for value in evidence.values())
    if len(values) < 2:
        return False
    if qtype == "depth_closest":
        return values[1] - values[0] > 1.0
    return min(values[-1] - values[-2], values[1] - values[0]) > 100.0


def read_candidates(parquet_dir: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("install the hub extra: pip install -e '.[hub]'") from exc

    rows: list[dict] = []
    for filename in ("spatial_binary.parquet", "spatial_bbox.parquet", "attribute.parquet"):
        path = parquet_dir / filename
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        for row in pq.read_table(path).to_pylist():
            stratum = _stratum(row)
            if stratum not in QUOTAS or row["frame"] != "00000":
                continue
            if row["type"] in {
                "spatial_lr_extreme",
                "depth_closest",
                "spatial_farthest",
            } and not _unambiguous_bbox(row):
                continue
            row["_source"] = filename
            row["_stratum"] = stratum
            rows.append(row)
    return rows


def select_rows(rows: list[dict], seed: int = 20260910) -> list[dict]:
    """Greedily cover quotas while minimizing distinct RGB tar shards."""
    by_image: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["scene_id"], row["kind"], row["phase"], row["frame"])
        by_image[key].append(row)
    remaining = dict(QUOTAS)
    selected: list[dict] = []
    used_ids: set[str] = set()

    def row_id(row: dict) -> str:
        return "\x1f".join(
            str(row[key])
            for key in ("_source", "scene_id", "kind", "phase", "frame", "type", "question")
        )

    while any(remaining.values()):
        scored = []
        for image_key, image_rows in by_image.items():
            available = defaultdict(int)
            for row in image_rows:
                identity = row_id(row)
                if identity not in used_ids:
                    available[row["_stratum"]] += 1
            gain = sum(min(remaining[key], count) for key, count in available.items())
            if gain:
                scored.append((-gain, _rank(seed, image_key), image_key))
        if not scored:
            missing = {key: value for key, value in remaining.items() if value}
            raise ValueError(f"unable to fill smoke quotas: {missing}")
        _, _, chosen_key = min(scored)
        chosen = sorted(by_image[chosen_key], key=lambda row: _rank(seed, row["question"]))
        for row in chosen:
            stratum = row["_stratum"]
            identity = row_id(row)
            if remaining[stratum] and identity not in used_ids:
                selected.append(row)
                used_ids.add(identity)
                remaining[stratum] -= 1

    return sorted(
        selected,
        key=lambda row: (
            row["type"],
            row["kind"],
            row["scene_id"],
            -1 if row["phase"] is None else row["phase"],
            row["question"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    selected = select_rows(read_candidates(args.parquet_dir), args.seed)
    samples = [VQASample.from_dict(row) for row in selected]
    write_manifest(samples, args.out)
    print(f"wrote {len(samples)} rows using {len({json.dumps(s.image) for s in samples})} images")


if __name__ == "__main__":
    main()
