"""Q7/Q8: compare in-context spatial rows against the published
spatial_farthest rows (out/vqa_parquet/spatial_bbox.parquet), keyed on
(scene_id, phase, frame, anchor_name, answer_name, answer_bbox). Reports
exact intersection / only-in-normal / only-in-context counts for BOTH the
raw uncapped bank and the deterministically-capped Parquet already built.
If overlap is below 100% against the raw uncapped bank (the true candidate
superset), derives a paired Parquet containing EXACTLY the published
spatial_farthest rows (the intersection) -- a pure filter of the existing
raw bank, no scene reprocessing.
"""
import json
import sys

import pyarrow.parquet as pq

IN_RAW = "out/incontext/gt_incontext_spatial.jsonl"
IN_CAPPED = "out/incontext_parquet/incontext_mos_spatial_bbox.parquet"
PUBLISHED = "out/vqa_parquet/spatial_bbox.parquet"
OUT_PAIRED = "out/incontext_paired/incontext_mos_spatial_bbox_paired.parquet"


def key_of(scene, phase, frame, anchor, answer, bbox):
    return (scene, int(phase), frame, anchor, answer, tuple(bbox))


def load_published_keys():
    df = pq.read_table(PUBLISHED).to_pandas()
    sub = df[df["type"] == "spatial_farthest"]
    keys = {}
    for _, r in sub.iterrows():
        # published Parquet JSON-encodes evidence as a string (see
        # pilot/to_parquet.py) -- the in-context bank keeps it as a native
        # dict, so only this side needs decoding.
        ev = json.loads(r["evidence"]) if isinstance(r["evidence"], str) else r["evidence"]
        k = key_of(r["scene_id"], r["phase"], r["frame"], ev["reference"], r["answer"], r["answer_bbox"])
        keys.setdefault(k, []).append(r["question"])
    return keys


def load_incontext_raw():
    rows, keys = [], {}
    with open(IN_RAW) as f:
        for line in f:
            r = json.loads(line)
            k = key_of(r["scene_id"], r["phase"], r["frame"], r["evidence"]["reference"], r["answer"], r["answer_bbox"])
            keys.setdefault(k, []).append(r)
            rows.append((k, r))
    return rows, keys


def load_incontext_capped_keys():
    df = pq.read_table(IN_CAPPED).to_pandas()
    keys = {}
    for _, r in df.iterrows():
        k = key_of(r["scene_id"], r["phase"], r["frame"], r["evidence"]["reference"], r["answer"], r["answer_bbox"])
        keys.setdefault(k, []).append(r["sample_id"])
    return keys


def report(name, published_keys, incontext_keys):
    pub_set, ic_set = set(published_keys), set(incontext_keys)
    intersection = pub_set & ic_set
    only_pub = pub_set - ic_set
    only_ic = ic_set - pub_set
    multi_pub = {k: v for k, v in published_keys.items() if len(v) > 1}
    multi_ic = {k: v for k, v in incontext_keys.items() if len(v) > 1}
    print(f"\n=== {name} ===")
    print(f"published spatial_farthest distinct keys: {len(pub_set)}")
    print(f"in-context distinct keys: {len(ic_set)}")
    print(f"intersection: {len(intersection)}  ({100*len(intersection)/max(len(pub_set),1):.2f}% of published)")
    print(f"only-in-published (no in-context match): {len(only_pub)}")
    print(f"only-in-context (no published match): {len(only_ic)}")
    print(f"multiply-matched published keys (dup rows): {len(multi_pub)}")
    print(f"multiply-matched in-context keys (dup rows): {len(multi_ic)}")
    if only_pub:
        print("  example only-in-published keys:", list(only_pub)[:3])
    if only_ic:
        print("  example only-in-context keys:", list(only_ic)[:3])
    return pub_set, ic_set, intersection


def main():
    published_keys = load_published_keys()
    raw_rows, raw_keys = load_incontext_raw()
    capped_keys = load_incontext_capped_keys()

    pub_set, capped_set, capped_intersection = report("capped Parquet (current output) vs published", published_keys, capped_keys)
    _, raw_set, intersection = report("raw UNCAPPED bank vs published (true candidate superset)", published_keys, raw_keys)

    capped_coverage = len(capped_intersection) / max(len(pub_set), 1)
    raw_coverage = len(intersection) / max(len(pub_set), 1)
    print(f"\ncurrent-output (capped) overlap vs published: {capped_coverage:.4%}")
    print(f"raw-bank overlap vs published: {raw_coverage:.4%}")
    if capped_coverage < 1.0:
        print(f"\nQ8: current output's overlap ({capped_coverage:.2%}) is below 100% -- "
              f"deriving the exact-paired Parquet from the raw bank regardless of the raw bank's own "
              f"({raw_coverage:.2%}) coverage, since that's the number that actually matters for "
              f"row-for-row alignment with the published set.")
        print("Overlap below 100% -- deriving a paired Parquet with EXACTLY the published spatial_farthest rows "
              "(pure filter of the raw bank, no scene reprocessing).")
        paired_rows = []
        seen = set()
        for k, r in raw_rows:
            if k in pub_set and k not in seen:
                paired_rows.append(r)
                seen.add(k)  # one in-context row per published key -- first match, deterministic file order
        import pyarrow as pa
        import os
        os.makedirs("out/incontext_paired", exist_ok=True)
        table = pa.Table.from_pylist(paired_rows)
        pq.write_table(table, OUT_PAIRED, compression="zstd")
        print(f"wrote {len(paired_rows)} rows -> {OUT_PAIRED} "
              f"(target: {len(pub_set)} published spatial_farthest facts; "
              f"{len(pub_set) - len(paired_rows)} published facts have NO in-context counterpart at all in the raw bank)")
    else:
        print("100% overlap -- no paired derivation needed.")


if __name__ == "__main__":
    main()
