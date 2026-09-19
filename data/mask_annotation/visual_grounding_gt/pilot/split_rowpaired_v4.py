"""Split the v4 row-paired JSONL (function-canonicalization fix applied) into
mos/ego Parquets. Also checks for duplicate sample_ids as a pre-flight gate
before any further validation runs."""
import json
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

IN_PATH = "out/incontext_paired/gt_incontext_general_rowpaired_v4.jsonl"
OUT_DIR = "out/incontext_paired"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_jsonl(IN_PATH)
    print(f"loaded {len(rows)} rows from {IN_PATH}")

    ids = [r["sample_id"] for r in rows]
    id_counts = Counter(ids)
    dupes = {k: v for k, v in id_counts.items() if v > 1}
    print(f"duplicate sample_ids: {len(dupes)}")
    if dupes:
        for k, v in list(dupes.items())[:10]:
            print("  DUPLICATE:", k, v)
        print("ABORTING split due to duplicate sample_ids")
        sys.exit(1)

    mos = [r for r in rows if r["kind"] == "mos"]
    ego = [r for r in rows if r["kind"] == "ego"]
    assert len(mos) + len(ego) == len(rows)
    print(f"mos={len(mos)} ego={len(ego)}")

    for name, sub in [("incontext_mos_attribute_bbox_v4", mos), ("incontext_ego_attribute_bbox_v4", ego)]:
        table = pa.Table.from_pylist(sub)
        out_path = f"{OUT_DIR}/{name}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        print(f"{name}: {len(sub)} rows -> {out_path}")

    print("\n--- rows by type/kind/phase ---")
    from collections import defaultdict
    by_tkp = defaultdict(int)
    for r in rows:
        by_tkp[(r["type"], r["kind"], r["phase"])] += 1
    for k in sorted(by_tkp):
        print(" ", k, by_tkp[k])


if __name__ == "__main__":
    main()
