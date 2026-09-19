"""Convert the in-context JSONL GT into the 3 required output Parquets:

  incontext_mos_attribute_bbox.parquet   (gt_incontext_general.jsonl, kind==mos)
  incontext_ego_attribute_bbox.parquet   (gt_incontext_general.jsonl, kind==ego)
  incontext_mos_spatial_bbox.parquet     (gt_incontext_spatial.jsonl, all mos, CAPPED -- see below)

Does not touch the existing published attribute.parquet/spatial_bbox.parquet/
spatial_binary.parquet -- entirely separate input and output paths.

---- Spatial cap ----

gt_incontext.gen_incontext_spatial_for_frame() is UNCAPPED by construction --
unlike gt_spatial.gen_spatial_questions()'s published spatial_farthest, which
calls `_cap(farthest_candidates, max_per_type=5, rng)` before emitting, the
in-context generator emits every entry compute_farthest_candidates() returns,
so a frame with N valid depth-ranked instances can produce up to N-1 rows
(confirmed empirically: scene001/mos/phase0's first frame alone has 6, one
more than the published cap). The raw JSONL (gt_incontext_spatial.jsonl) is
the ungapped candidate bank and is preserved as-is; THIS script caps it down
to at most 5 rows per (scene_id, phase, frame) when building the published
Parquet, without ever re-running generation.

Selection is deterministic, not random-per-run: within each (scene_id,
phase, frame) group, every row is ranked by
    sha256(f"{CAP_SEED}|{scene_id}|{phase}|{frame}|{reference_source_catalog_id}|{target_source_catalog_id}")
ascending, and the 5 lowest-hash rows are kept. This is the same
stable-hash-selection pattern gt_incontext.py's own reference picker uses
(`_selection_hash`/`_pick_reference`) -- reproducible from the raw bank
alone, independent of generation order or process/thread scheduling.
"""
import hashlib
import json
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "out/incontext"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "out/incontext_parquet"
CAP_SEED = 42          # same SEED constant run_incontext_gt.py generation used
SPATIAL_CAP_PER_FRAME = 5  # matches the published (single-image) spatial_farthest's max_per_type


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _cap_rank_key(row):
    key = (f"{CAP_SEED}|{row['scene_id']}|{row['phase']}|{row['frame']}|"
           f"{row['reference_source_catalog_id']}|{row['target_source_catalog_id']}")
    return hashlib.sha256(key.encode()).hexdigest()


def cap_spatial_deterministic(rows, cap=SPATIAL_CAP_PER_FRAME):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene_id"], r["phase"], r["frame"])].append(r)
    kept = []
    for group in groups.values():
        group.sort(key=_cap_rank_key)
        kept.extend(group[:cap])
    return kept


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    general = load_jsonl(f"{IN_DIR}/gt_incontext_general.jsonl")
    spatial_raw = load_jsonl(f"{IN_DIR}/gt_incontext_spatial.jsonl")

    mos_attr = [r for r in general if r["kind"] == "mos"]
    ego_attr = [r for r in general if r["kind"] == "ego"]
    assert all(r["kind"] == "mos" for r in spatial_raw), "inctx_spatial_farthest must be mos-only"

    spatial_capped = cap_spatial_deterministic(spatial_raw)
    print(f"spatial: {len(spatial_raw)} raw (uncapped) rows -> {len(spatial_capped)} after "
          f"deterministic cap of {SPATIAL_CAP_PER_FRAME}/frame "
          f"({len(spatial_raw) - len(spatial_capped)} dropped by the cap, raw bank preserved at "
          f"{IN_DIR}/gt_incontext_spatial.jsonl)")

    for name, rows in [("incontext_mos_attribute_bbox", mos_attr),
                        ("incontext_ego_attribute_bbox", ego_attr),
                        ("incontext_mos_spatial_bbox", spatial_capped)]:
        if not rows:
            print(f"{name}: 0 rows, skipping (nothing to write)")
            continue
        table = pa.Table.from_pylist(rows)
        out_path = f"{OUT_DIR}/{name}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"{name}: {len(rows)} rows -> {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
