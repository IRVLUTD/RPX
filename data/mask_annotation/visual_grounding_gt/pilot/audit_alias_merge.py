"""Q6/Q7:
Q6 -- merge obvious spelling equivalents (gray/grey) for a normalization
audit, report before/after distinct-value counts and re-run the compound-
style ambiguity question for the merged color space (using the already-
built 7-scene empirical frame data pattern, but for colors this time --
much rarer collision risk since colors are single atomic tokens already).

Q7 -- manual-review CSV for potential function aliases: for every function
value that SHARES A TOKEN with another function value ACROSS DIFFERENT
objects (not just within one object, which the earlier audit covered) --
this is the more consequential case, since a cross-object share is what
could make a reference match an unintended target. Never auto-merges;
produces a CSV for a human to decide.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
import sos_catalog as sc  # noqa: E402

STOPWORDS = {"a", "an", "the", "to", "of", "for", "in", "on", "with", "and", "or", "it", "is"}
ALIAS_MERGE_MAP = {"gray": "grey"}  # Q6: the one obvious, unambiguous spelling pair in this catalog


def tokenize(s):
    words = s.lower().replace("/", " ").replace("&", " ").replace(",", " ").split()
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def main():
    catalog = sc.load_catalog()

    # ---- Q6: gray/grey merge, before/after ----
    color_counter_before = defaultdict(int)
    color_counter_after = defaultdict(int)
    for obj in catalog.values():
        for c in obj.attrs_norm["color"]:
            color_counter_before[c] += 1
            color_counter_after[ALIAS_MERGE_MAP.get(c, c)] += 1

    print("=== Q6: gray/grey merge ===")
    print(f"distinct normalized colors BEFORE merge: {len(color_counter_before)}")
    print(f"distinct normalized colors AFTER merge:  {len(color_counter_after)}")
    print(f"  grey (before): {color_counter_before.get('grey', 0)} objects")
    print(f"  gray (before): {color_counter_before.get('gray', 0)} objects")
    print(f"  grey (after merge): {color_counter_after.get('grey', 0)} objects")

    # Which catalog objects have BOTH gray and grey in their own list (would
    # collapse to one value, changing len() of their own color list -- not
    # an ambiguity risk by itself, just a de-dup, but worth flagging)
    self_collisions = [o.object_id for o in catalog.values()
                        if "gray" in o.attrs_norm["color"] and "grey" in o.attrs_norm["color"]]
    print(f"objects with BOTH gray and grey in their own list (self-collision on merge): {self_collisions}")

    # Which objects currently uniquely own "gray" or "grey" that would gain
    # a same-object collision after merge (i.e., does merging ever change
    # who "uniquely owns" a color, catalog-wide -- this is the ambiguity-
    # relevant question, structurally identical to the material-compound
    # check, but for color aliasing instead of slash/ampersand compounds)
    gray_owners = {o.object_id for o in catalog.values() if "gray" in o.attrs_norm["color"]}
    grey_owners = {o.object_id for o in catalog.values() if "grey" in o.attrs_norm["color"]}
    print(f"catalog objects with 'gray': {sorted(gray_owners)}")
    print(f"catalog objects with 'grey': {sorted(grey_owners)}")
    print(f"combined 'grey' owners after merge: {sorted(gray_owners | grey_owners)} "
          f"({len(gray_owners | grey_owners)} objects, vs {len(grey_owners)} for 'grey' alone before)")

    # ---- Q7: cross-object function-token-overlap manual-review CSV ----
    print("\n=== Q7: cross-object function alias candidates ===")
    token_to_objs = defaultdict(set)  # token -> {(object_id, raw_function), ...}
    for obj in catalog.values():
        for raw_f in obj.attrs_raw["function"]:
            for tok in tokenize(raw_f):
                token_to_objs[tok].add((obj.object_id, raw_f))

    pairs_seen = set()
    csv_rows = []
    for tok, members in token_to_objs.items():
        objs_involved = {m[0] for m in members}
        if len(objs_involved) < 2:
            continue  # within-object only, already covered by the earlier audit
        members_list = sorted(members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                (obj_a, func_a), (obj_b, func_b) = members_list[i], members_list[j]
                if obj_a == obj_b:
                    continue
                if sc.normalize_value(func_a) == sc.normalize_value(func_b):
                    continue  # already identical strings -- not an aliasing question,
                               # normalize_value() already unifies these; a REAL alias
                               # candidate is two DIFFERENT strings that share meaning
                pair_key = tuple(sorted([(obj_a, func_a), (obj_b, func_b)]))
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)
                csv_rows.append({
                    "raw_function_a": func_a, "object_a": obj_a,
                    "raw_function_b": func_b, "object_b": obj_b,
                    "shared_token": tok,
                    "example_question_a": f"Which object is used for {func_a}?",
                    "example_question_b": f"Which object is used for {func_b}?",
                })

    out_csv = "out/function_alias_review.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_function_a", "object_a", "raw_function_b", "object_b",
                                           "shared_token", "example_question_a", "example_question_b"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"{len(csv_rows)} cross-object function-alias candidate pairs -> {out_csv}")
    print("(NOT auto-merged -- for human review. 'affected references'/'affected target frames' "
          "require joining against the generated Parquets per candidate pair; see report for the "
          "top candidates joined against real data.)")
    for r in csv_rows[:10]:
        print(f"  {r['object_a']:20s} {r['raw_function_a']!r:45s} <-> {r['object_b']:20s} {r['raw_function_b']!r:45s} (token={r['shared_token']})")


if __name__ == "__main__":
    main()
