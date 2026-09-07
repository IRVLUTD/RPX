"""Q3 (finish)/Q4: split the row-paired JSONL into mos/ego Parquets (new
files, not overwriting the redistributed-capped ones), then compare the two
banks on the axes requested."""
import json
import sys
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

IN_PATH = "out/incontext_paired/gt_incontext_general_rowpaired.jsonl"
OUT_DIR = "out/incontext_paired"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_jsonl(IN_PATH)
    mos = [r for r in rows if r["kind"] == "mos"]
    ego = [r for r in rows if r["kind"] == "ego"]
    for name, sub in [("incontext_mos_attribute_bbox_rowpaired", mos), ("incontext_ego_attribute_bbox_rowpaired", ego)]:
        table = pa.Table.from_pylist(sub)
        out_path = f"{OUT_DIR}/{name}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        print(f"{name}: {len(sub)} rows -> {out_path}")

    import pyarrow.parquet as pq2
    redist_mos = pq2.read_table("out/incontext_parquet/incontext_mos_attribute_bbox.parquet").to_pandas()
    redist_ego = pq2.read_table("out/incontext_parquet/incontext_ego_attribute_bbox.parquet").to_pandas()
    import pandas as pd
    redist = pd.concat([redist_mos, redist_ego], ignore_index=True)
    rowpaired = pd.DataFrame(rows)

    print(f"\n{'='*80}\nQ4 comparison: row-paired vs redistributed-capped\n{'='*80}")
    print(f"total rows: row-paired={len(rowpaired)}  redistributed-capped={len(redist)}")

    print("\n--- rows by type/kind/phase ---")
    print("row-paired:\n", rowpaired.groupby(["type", "kind", "phase"], dropna=False).size().to_string())
    print("\nredistributed-capped:\n", redist.groupby(["type", "kind", "phase"], dropna=False).size().to_string())

    def frame_stats(df, label):
        cov = df.drop_duplicates(subset=["scene_id", "kind", "phase", "frame"])
        centered = df[df["answer_is_centered"] == True]  # noqa: E712
        ccov = centered.drop_duplicates(subset=["scene_id", "kind", "phase", "frame"])
        print(f"\n{label}: distinct target frames={len(cov)}  distinct CENTERED-covered frames={len(ccov)}")
        print(f"  distinct scenes={df.scene_id.nunique()}  distinct target objects={df.target_source_catalog_id.nunique()}  "
              f"distinct reference objects={df.reference_object_id.nunique()}")
        ref_freq = df.reference_object_id.value_counts()
        top_share = ref_freq.iloc[0] / len(df) if len(df) else 0
        print(f"  top reference share: {top_share:.2%} ({ref_freq.index[0]})")
        print(f"  reference-frequency: min={ref_freq.min()} median={ref_freq.median():.0f} max={ref_freq.max()}")

    frame_stats(rowpaired, "row-paired")
    frame_stats(redist, "redistributed-capped")

    print("\n--- drop reasons (row-paired) ---")
    drops = load_jsonl("out/incontext_paired/rowpaired_drops.jsonl")
    c = Counter((d["type"], d["reason"]) for d in drops)
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(" ", k, v)


if __name__ == "__main__":
    main()
