"""Q6: prove every proposed (capped) in-context general row maps to a
published normal attribute row (out/vqa_parquet/attribute.parquet) with the
same scene, kind, phase, frame, source type, target identity, and
answer_bbox. Reports matched/unmatched/multiply-matched counts by type, and
writes an augmented Parquet carrying a new `source_sample_id` field (a
stable hash of the matched published row's own key fields, since the
published Parquet has no row-id column of its own to reuse directly).
"""
import hashlib
import json
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

IN_CAPPED = sys.argv[1] if len(sys.argv) > 1 else "out/incontext_paired/gt_incontext_general_capped.jsonl"
PUBLISHED = "out/vqa_parquet/attribute.parquet"
OUT_PARQUET = "out/incontext_paired/incontext_general_capped_matched.parquet"
OUT_REPORT = "out/incontext_paired/match_report.json"


def pub_key(scene, kind, phase, frame, type_, target_oid, bbox):
    phase_key = None if (phase is None or (isinstance(phase, float) and phase != phase)) else int(phase)
    bbox_key = tuple(bbox) if bbox is not None else None
    return (scene, kind, phase_key, frame, type_, target_oid, bbox_key)


def source_sample_id(key):
    return hashlib.sha256("|".join(str(k) for k in key).encode()).hexdigest()[:24]


def main():
    print("loading published attribute.parquet ...", flush=True)
    pub = pq.read_table(PUBLISHED).to_pandas()
    pub_index = defaultdict(list)
    for i, r in pub.iterrows():
        k = pub_key(r["scene_id"], r["kind"], r["phase"], r["frame"], r["type"], r["target_oid"], r["answer_bbox"])
        pub_index[k].append(i)
    print(f"{len(pub)} published rows, {len(pub_index)} distinct keys", flush=True)

    print(f"loading capped in-context bank from {IN_CAPPED} ...", flush=True)
    matched, unmatched, multiply = defaultdict(int), defaultdict(int), defaultdict(int)
    out_rows = []
    unmatched_examples = defaultdict(list)
    n = 0
    with open(IN_CAPPED) as f:
        for line in f:
            row = json.loads(line)
            n += 1
            k = pub_key(row["scene_id"], row["kind"], row["phase"], row["frame"],
                        row["source_type"], row["target_source_catalog_id"], row["answer_bbox"])
            hits = pub_index.get(k, [])
            t = row["type"]
            if len(hits) == 0:
                unmatched[t] += 1
                if len(unmatched_examples[t]) < 3:
                    unmatched_examples[t].append({"sample_id": row["sample_id"], "key": [str(x) for x in k]})
                row["source_sample_id"] = None
            else:
                matched[t] += 1
                if len(hits) > 1:
                    multiply[t] += 1
                row["source_sample_id"] = source_sample_id(k)
            out_rows.append(row)
            if n % 200000 == 0:
                print(f"  ... {n} rows matched", flush=True)

    import os
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, OUT_PARQUET, compression="zstd")
    print(f"\nwrote {len(out_rows)} rows -> {OUT_PARQUET}")

    report = {"total": n, "matched_by_type": dict(matched), "unmatched_by_type": dict(unmatched),
              "multiply_matched_by_type": dict(multiply), "unmatched_examples": dict(unmatched_examples)}
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Q6 match report ===")
    all_types = sorted(set(matched) | set(unmatched))
    for t in all_types:
        m, u, mm = matched.get(t, 0), unmatched.get(t, 0), multiply.get(t, 0)
        print(f"{t:30s} matched={m:8d} unmatched={u:8d} multiply_matched={mm:6d} "
              f"match_rate={100*m/max(m+u,1):.2f}%")
        if u > 0:
            for ex in unmatched_examples[t]:
                print(f"    unmatched example: {ex}")


if __name__ == "__main__":
    main()
