"""Follow-up to match_general_to_published.py: the frame-EXACT match rate
(60-80% for single_color/material/function/composition) looked low enough
to need explaining, not just reporting. Root cause, confirmed by direct
inspection (scene001/mos/phase0/frame00000's attr_composition rows): both
generators run the SAME phase-wide dedup.distribute() algorithm, which
explicitly does NOT pin a fact to one canonical frame -- it places each fact
in "whichever of its valid frames currently has the lightest load," a
choice that depends on RNG draw order and on what ELSE is competing for
frame slots that phase. The published generator's dedup key is bare
(material, function); this pipeline's in-context dedup key additionally
encodes the (target, reference) identity pair (needed since two different
references could theoretically serve the same fact). Different key
structure -> different RNG shuffle order -> a DIFFERENT, EQUALLY VALID
frame gets chosen for the same underlying fact in each independent run.
That is NOT a data-quality bug -- attr_composition's own eligibility
condition (see gt_attributes.py) is satisfied by the target's real,
static-within-phase attributes, so the SAME fact is valid across the whole
phase, and the published/in-context generators legitimately disagree only
on which of those many valid frames to render it in.

This script re-checks at the FACT level instead: does a published row exist
ANYWHERE in the same (scene, kind, phase) for this (type, target_oid),
dropping frame and answer_bbox (both frame-position-dependent) from the key.
This is the semantically meaningful invariant: "is this a real fact the
published generator also recognized," not "did it pick literally the same
frame to show it in."
"""
import json
import sys
from collections import defaultdict

import pyarrow.parquet as pq

IN_CAPPED = sys.argv[1] if len(sys.argv) > 1 else "out/incontext_paired/gt_incontext_general_capped.jsonl"
PUBLISHED = "out/vqa_parquet/attribute.parquet"


def fact_key(scene, kind, phase, type_, target_oid):
    phase_key = None if (phase is None or (isinstance(phase, float) and phase != phase)) else int(phase)
    return (scene, kind, phase_key, type_, target_oid)


def main():
    print("loading published attribute.parquet ...", flush=True)
    pub = pq.read_table(PUBLISHED).to_pandas()
    pub_facts = set()
    for _, r in pub.iterrows():
        pub_facts.add(fact_key(r["scene_id"], r["kind"], r["phase"], r["type"], r["target_oid"]))
    print(f"{len(pub_facts)} distinct published (scene,kind,phase,type,target) facts", flush=True)

    matched, unmatched = defaultdict(int), defaultdict(int)
    unmatched_examples = defaultdict(list)
    n = 0
    with open(IN_CAPPED) as f:
        for line in f:
            row = json.loads(line)
            n += 1
            k = fact_key(row["scene_id"], row["kind"], row["phase"], row["source_type"], row["target_source_catalog_id"])
            t = row["type"]
            if k in pub_facts:
                matched[t] += 1
            else:
                unmatched[t] += 1
                if len(unmatched_examples[t]) < 3:
                    unmatched_examples[t].append({"sample_id": row["sample_id"], "fact_key": [str(x) for x in k]})

    print(f"\n{n} in-context rows checked at FACT level (frame-independent)\n")
    for t in sorted(set(matched) | set(unmatched)):
        m, u = matched.get(t, 0), unmatched.get(t, 0)
        print(f"{t:30s} fact_matched={m:8d} fact_unmatched={u:8d} fact_match_rate={100*m/max(m+u,1):.2f}%")
        for ex in unmatched_examples[t]:
            print(f"    still-unmatched example: {ex}")


if __name__ == "__main__":
    main()
