"""Q10/Q11: full semantic normalization audit across all 70 published SOS
objects -- raw vs normalized colors/materials/functions, slash/ampersand
compound detection, and a heuristic near-duplicate ("likely synonym") flag
for function phrases (shared significant words, e.g. "baking food" /
"to bake a cake"). Writes a CSV for full review; prints a summary + examples
to stdout. Does not change any normalization behavior -- audit only.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
import sos_catalog as sc  # noqa: E402

OUT_CSV = "out/normalization_audit.csv"
STOPWORDS = {"a", "an", "the", "to", "of", "for", "in", "on", "with", "and", "or"}


def tokenize(s):
    return {w for w in normalize_words(s) if w not in STOPWORDS and len(w) > 2}


def normalize_words(s):
    return s.lower().replace("/", " ").replace("&", " ").replace(",", " ").split()


def main():
    catalog = sc.load_catalog()
    rows = []
    compound_count = {"material": 0, "color": 0, "function": 0}
    total_values = {"material": 0, "color": 0, "function": 0}

    for obj in sorted(catalog.values(), key=lambda o: o.object_id):
        for field in ("color", "material", "function"):
            for raw, norm in zip(obj.attrs_raw[field], obj.attrs_norm[field]):
                total_values[field] += 1
                is_compound = "/" in raw or "&" in raw
                if is_compound:
                    compound_count[field] += 1
                rows.append({
                    "object_id": obj.object_id, "global_object_id": obj.global_object_id,
                    "field": field, "raw_value": raw, "normalized_value": norm,
                    "is_slash_or_ampersand_compound": is_compound,
                })

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} (object,field,value) rows -> {OUT_CSV}")

    print("\n=== compound (slash/ampersand) value counts ===")
    for field in ("color", "material", "function"):
        print(f"  {field}: {compound_count[field]}/{total_values[field]} values are compounds")

    print("\n=== example compound values ===")
    seen = 0
    for r in rows:
        if r["is_slash_or_ampersand_compound"] and seen < 15:
            print(f"  {r['object_id']:25s} {r['field']:9s} raw={r['raw_value']!r}")
            seen += 1

    # Likely-synonym function phrases: within EACH object's own function
    # list, flag pairs of raw phrases that share a significant word after
    # stopword removal (e.g. "baking food" / "to bake a cake" share "bak..."
    # only after stemming, which this deliberately does NOT do -- flagged
    # here as a token-overlap heuristic, not applied to generation).
    print("\n=== likely-synonym function phrases (token-overlap heuristic, within one object) ===")
    synonym_examples = []
    for obj in catalog.values():
        funcs = obj.attrs_raw["function"]
        for i in range(len(funcs)):
            for j in range(i + 1, len(funcs)):
                t1, t2 = tokenize(funcs[i]), tokenize(funcs[j])
                if t1 & t2:
                    synonym_examples.append((obj.object_id, funcs[i], funcs[j], t1 & t2))
    print(f"{len(synonym_examples)} within-object function pairs share a significant word")
    for oid, f1, f2, shared in synonym_examples[:15]:
        print(f"  {oid:25s} {f1!r} <-> {f2!r}  (shared: {shared})")

    # Cross-object exact-normalized-value collisions for color (the
    # smallest, most semantically atomic field) -- these are legitimate
    # shared facts, not a normalization problem, but worth surfacing.
    print("\n=== distinct normalized color values across the 70-object catalog ===")
    color_counter = Counter()
    for obj in catalog.values():
        for c in obj.attrs_norm["color"]:
            color_counter[c] += 1
    print(f"{len(color_counter)} distinct normalized colors, top 10 by object count:")
    for c, n in color_counter.most_common(10):
        print(f"  {c:20s} {n} objects")


if __name__ == "__main__":
    main()
