"""Convert the in-context JSONL GT into the 3 required output Parquets:

  incontext_mos_attribute_bbox.parquet   (gt_incontext_general.jsonl, kind==mos)
  incontext_ego_attribute_bbox.parquet   (gt_incontext_general.jsonl, kind==ego)
  incontext_mos_spatial_bbox.parquet     (gt_incontext_spatial.jsonl, all mos)

Does not touch the existing published attribute.parquet/spatial_bbox.parquet/
spatial_binary.parquet -- entirely separate input and output paths.
"""
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "out/incontext"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "out/incontext_parquet"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    general = load_jsonl(f"{IN_DIR}/gt_incontext_general.jsonl")
    spatial = load_jsonl(f"{IN_DIR}/gt_incontext_spatial.jsonl")

    mos_attr = [r for r in general if r["kind"] == "mos"]
    ego_attr = [r for r in general if r["kind"] == "ego"]
    assert all(r["kind"] == "mos" for r in spatial), "inctx_spatial_farthest must be mos-only"

    for name, rows in [("incontext_mos_attribute_bbox", mos_attr),
                        ("incontext_ego_attribute_bbox", ego_attr),
                        ("incontext_mos_spatial_bbox", spatial)]:
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
