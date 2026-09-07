"""Q1/Q2: corrected general-vs-published matching using keys that include
the actual attribute value (the previous match omitted it, which could
silently match a row to a published fact about a DIFFERENT value for the
same target -- e.g. two distinct colors of the same multi-color object).

Two distinct identifiers are assigned, per Q2:
  source_fact_id   -- phase-level identity: (scene, kind, phase, type,
                       target_oid, normalized attribute value(s)). Matches
                       whenever this fact was published SOMEWHERE in the
                       phase, regardless of which frame. Always assignable
                       when a match exists (this is the "is this a real
                       recognized fact" question).
  source_sample_id -- exact row-level identity: source_fact_id further
                       scoped to the SAME frame and SAME answer_bbox. Only
                       assignable when the published generator happened to
                       place this exact fact in this exact frame too (see
                       report -- phase-wide dedup.distribute() legitimately
                       scatters the same fact across different valid frames
                       between two independent runs, so this is expected to
                       be well below 100% even for a fully correct dataset).
Both are None when no published fact matches at all.
"""
import hashlib
import json
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
import sos_catalog as sc  # noqa: E402

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "out/incontext_paired/gt_incontext_general_capped.jsonl"
PUBLISHED = "out/vqa_parquet/attribute.parquet"
OUT_PARQUET = sys.argv[2] if len(sys.argv) > 2 else "out/incontext_paired/incontext_general_capped_matched_v2.parquet"
OUT_REPORT = "out/incontext_paired/match_report_v2.json"


def phase_key(phase):
    if phase is None or (isinstance(phase, float) and phase != phase):
        return None
    return int(phase)


def fact_key_for_row(scene, kind, phase, type_, target_oid, attribute_kind, attribute_value,
                      attribute_material, attribute_function):
    if attribute_kind == "composition":
        val_part = f"{sc.normalize_value(attribute_material)}|{sc.normalize_value(attribute_function)}"
    elif attribute_material is not None:  # odd_one_out
        val_part = sc.normalize_value(attribute_material)
    else:
        val_part = sc.normalize_value(attribute_value)
    return (scene, kind, phase_key(phase), type_, target_oid, val_part)


def stable_id(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


def published_fact_value(pub_type, row_dict, attrs_needed):
    """Extract the same val_part shape from a published row -- composition
    needs the (material,function) pair, which isn't a stored column, so it
    is parsed from the question text (the only place it's recorded in the
    published schema)."""
    if pub_type == "attr_composition":
        # "What is the object made of {m} that is used for {f}?"
        q = row_dict["question"]
        try:
            m = q.split("made of ", 1)[1].split(" that is used for ")[0]
            f = q.split(" that is used for ", 1)[1].rstrip("?")
        except IndexError:
            return None
        return f"{sc.normalize_value(m)}|{sc.normalize_value(f)}"
    if pub_type == "attr_odd_one_out":
        q = row_dict["question"]  # "Which object is NOT made of {m}?"
        try:
            m = q.split("NOT made of ", 1)[1].rstrip("?")
        except IndexError:
            return None
        return sc.normalize_value(m)
    # single_color/material/function: attr_value column has it directly
    return sc.normalize_value(row_dict["attr_value"])


def main():
    print("loading published attribute.parquet ...", flush=True)
    pub = pq.read_table(PUBLISHED).to_pandas()
    RETAINED = {"attr_single_color", "attr_single_material", "attr_single_function",
                "attr_composition", "attr_odd_one_out"}
    pub = pub[pub["type"].isin(RETAINED)]

    fact_index = defaultdict(list)   # fact_key -> [row_idx, ...]  (any frame)
    row_index = defaultdict(list)    # (fact_key, frame, bbox) -> [row_idx, ...]  (exact frame+bbox)
    records = pub.to_dict("records")
    for i, r in enumerate(records):
        val_part = published_fact_value(r["type"], r, None)
        if val_part is None:
            continue
        fk = (r["scene_id"], r["kind"], phase_key(r["phase"]), r["type"], r["target_oid"], val_part)
        fact_index[fk].append(i)
        row_index[(fk, r["frame"], tuple(r["answer_bbox"]))].append(i)
    print(f"{len(pub)} published rows -> {len(fact_index)} distinct facts, {len(row_index)} distinct (fact,frame,bbox) rows", flush=True)

    matched_fact, unmatched_fact, multi_fact = defaultdict(int), defaultdict(int), defaultdict(int)
    matched_row, multi_row = defaultdict(int), defaultdict(int)
    out_rows = []
    n = 0
    with open(IN_PATH) as f:
        for line in f:
            row = json.loads(line)
            n += 1
            t = row["type"]
            fk = fact_key_for_row(row["scene_id"], row["kind"], row["phase"], row["source_type"],
                                   row["target_source_catalog_id"], row["attribute_kind"],
                                   row["attribute_value"], row["attribute_material"], row["attribute_function"])
            fact_hits = fact_index.get(fk, [])
            if fact_hits:
                matched_fact[t] += 1
                if len(fact_hits) > 1:
                    multi_fact[t] += 1
                row["source_fact_id"] = stable_id(*fk)
                row_hits = row_index.get((fk, row["frame"], tuple(row["answer_bbox"])), [])
                if row_hits:
                    matched_row[t] += 1
                    if len(row_hits) > 1:
                        multi_row[t] += 1
                    row["source_sample_id"] = stable_id(*fk, row["frame"], tuple(row["answer_bbox"]))
                else:
                    row["source_sample_id"] = None
            else:
                unmatched_fact[t] += 1
                row["source_fact_id"] = None
                row["source_sample_id"] = None
            out_rows.append(row)
            if n % 200000 == 0:
                print(f"  ... {n} rows checked", flush=True)

    import os
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, OUT_PARQUET, compression="zstd")
    print(f"\nwrote {len(out_rows)} rows -> {OUT_PARQUET}")

    print("\n=== Q1: corrected (attribute-value-aware) fact-level + row-level match ===")
    for t in sorted(set(matched_fact) | set(unmatched_fact)):
        mf, uf, mmf = matched_fact.get(t, 0), unmatched_fact.get(t, 0), multi_fact.get(t, 0)
        mr, mmr = matched_row.get(t, 0), multi_row.get(t, 0)
        print(f"{t:30s} fact_matched={mf:8d} fact_unmatched={uf:8d} fact_multiply={mmf:6d} "
              f"fact_rate={100*mf/max(mf+uf,1):.2f}%  |  row_exact_matched={mr:8d} row_multiply={mmr:6d} "
              f"row_rate={100*mr/max(mf+uf,1):.2f}%")

    report = {"total": n, "matched_fact": dict(matched_fact), "unmatched_fact": dict(unmatched_fact),
              "multi_fact": dict(multi_fact), "matched_row_exact": dict(matched_row), "multi_row": dict(multi_row)}
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
