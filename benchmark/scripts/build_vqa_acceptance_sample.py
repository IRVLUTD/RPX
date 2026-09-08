#!/usr/bin/env python3
"""Build the deterministic 104-row VQA acceptance manifest, covering both
normal (one-image) and in-context (two-image) tasks:

  normal general:      5 types x 4 conditions x 2 = 40
  normal spatial:       3 types x 3 MOS phases x 2 = 18
  in-context general:   5 types x 4 conditions x 2 = 40
  in-context spatial:   1 type  x 3 MOS phases x 2  =  6
  total                                             = 104

"conditions" = {mos phase 0, mos phase 1, mos phase 2, ego}. Every selected
row's answer bbox is centered (middle 50% of both image axes -- computed for
normal rows, read from the in-context Parquet's precomputed answer_is_centered
column for in-context rows). Selection is seeded (seed 20260908), prefers
rows that introduce a new (scene_id, frame) over reusing one already picked,
and rejects any row that would create a normalized-function-alias semantic
duplicate with an already-selected row (see rpx_benchmark.vqa.canonical).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rpx_benchmark.vqa.contract import VQASample, write_manifest
from rpx_benchmark.vqa.sampling import DeterministicSelector, is_centered

SEED = 20260908

NORMAL_GENERAL_TYPES = (
    "attr_single_color",
    "attr_single_material",
    "attr_single_function",
    "attr_composition",
    "attr_odd_one_out",
)
NORMAL_SPATIAL_TYPES = ("spatial_lr_extreme", "depth_closest", "spatial_farthest")
INCONTEXT_GENERAL_TYPES = (
    "inctx_attr_single_color",
    "inctx_attr_single_material",
    "inctx_attr_single_function",
    "inctx_attr_composition",
    "inctx_attr_odd_one_out",
)
INCONTEXT_SPATIAL_TYPES = ("inctx_spatial_farthest",)
CONDITIONS = ("mos:0", "mos:1", "mos:2", "ego")
MOS_PHASES = (0, 1, 2)

QUOTA_PER_CELL = 2
EXPECTED_TOTAL = (
    len(NORMAL_GENERAL_TYPES) * len(CONDITIONS) * QUOTA_PER_CELL
    + len(NORMAL_SPATIAL_TYPES) * len(MOS_PHASES) * QUOTA_PER_CELL
    + len(INCONTEXT_GENERAL_TYPES) * len(CONDITIONS) * QUOTA_PER_CELL
    + len(INCONTEXT_SPATIAL_TYPES) * len(MOS_PHASES) * QUOTA_PER_CELL
)
assert EXPECTED_TOTAL == 104


def _condition(row: dict) -> str:
    return "ego" if row["kind"] == "ego" else f"mos:{int(row['phase'])}"


def _iter_rows(path: Path, columns: list[str] | None = None):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        yield from batch.to_pylist()


def collect_normal_general(parquet_dir: Path) -> dict[tuple[str, str], list[dict]]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in _iter_rows(parquet_dir / "attribute.parquet"):
        if row["type"] not in NORMAL_GENERAL_TYPES or row.get("answer_bbox") is None:
            continue
        if not is_centered(row):
            continue
        by_cell[(row["type"], _condition(row))].append(row)
    return by_cell


def collect_normal_spatial(parquet_dir: Path) -> dict[tuple[str, int], list[dict]]:
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in _iter_rows(parquet_dir / "spatial_bbox.parquet"):
        if row["type"] not in NORMAL_SPATIAL_TYPES or row["kind"] != "mos":
            continue
        if not is_centered(row):
            continue
        by_cell[(row["type"], int(row["phase"]))].append(row)
    return by_cell


def collect_incontext_general(parquet_dir: Path) -> dict[tuple[str, str], list[dict]]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for filename in ("incontext_mos_attribute_bbox.parquet", "incontext_ego_attribute_bbox.parquet"):
        for row in _iter_rows(parquet_dir / filename):
            if row["type"] not in INCONTEXT_GENERAL_TYPES:
                continue
            if not is_centered(row):
                continue
            by_cell[(row["type"], _condition(row))].append(row)
    return by_cell


def collect_incontext_spatial(parquet_dir: Path) -> dict[tuple[str, int], list[dict]]:
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in _iter_rows(parquet_dir / "incontext_mos_spatial_bbox.parquet"):
        if row["type"] not in INCONTEXT_SPATIAL_TYPES:
            continue
        if not is_centered(row):
            continue
        by_cell[(row["type"], int(row["phase"]))].append(row)
    return by_cell


def build(parquet_dir: Path, seed: int = SEED) -> list[dict]:
    selector = DeterministicSelector(seed)
    groups = [
        collect_normal_general(parquet_dir),
        collect_normal_spatial(parquet_dir),
        collect_incontext_general(parquet_dir),
        collect_incontext_spatial(parquet_dir),
    ]
    labels = ("normal_general", "normal_spatial", "incontext_general", "incontext_spatial")
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for label, by_cell in zip(labels, groups):
        selected = selector.select(by_cell, lambda _cell: QUOTA_PER_CELL)
        under_quota = {cell: len(picked) for cell, picked in selected.items() if len(picked) < QUOTA_PER_CELL}
        if under_quota:
            raise SystemExit(f"{label}: under-quota cells {under_quota}")
        expected_cells = {
            "normal_general": len(NORMAL_GENERAL_TYPES) * len(CONDITIONS),
            "normal_spatial": len(NORMAL_SPATIAL_TYPES) * len(MOS_PHASES),
            "incontext_general": len(INCONTEXT_GENERAL_TYPES) * len(CONDITIONS),
            "incontext_spatial": len(INCONTEXT_SPATIAL_TYPES) * len(MOS_PHASES),
        }[label]
        if len(selected) != expected_cells:
            raise SystemExit(f"{label}: expected {expected_cells} cells, got {len(selected)}")
        for cell in sorted(selected, key=str):
            rows.extend(selected[cell])
        counts[label] = sum(len(picked) for picked in selected.values())
    return rows, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rows, counts = build(args.parquet_dir, args.seed)
    if len(rows) != EXPECTED_TOTAL:
        raise SystemExit(f"expected {EXPECTED_TOTAL} rows, built {len(rows)}")
    samples = [VQASample.from_dict(row) for row in rows]
    ids = {sample.sample_id for sample in samples}
    if len(ids) != len(samples):
        raise SystemExit("duplicate sample_id in acceptance manifest")
    write_manifest(samples, args.out)
    distinct_scene_frame = len({(s.scene_id, s.kind, s.phase, s.frame) for s in samples})
    print(
        f"wrote {len(samples)} acceptance rows ({json.dumps(counts, sort_keys=True)}) "
        f"over {distinct_scene_frame} distinct (scene,kind,phase,frame) cells, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
