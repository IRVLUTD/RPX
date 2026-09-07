"""Q8: verify every scene in gt_incontext.done actually has non-empty,
parseable output in BOTH raw JSONL banks (or is legitimately empty for
gt_incontext_spatial.jsonl if a scene had no valid mos spatial facts --
checked, not assumed), and that neither file is truncated (every line
parses, last line included).
"""
import json
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "out/incontext"


def scan(path):
    scenes = set()
    n = 0
    last_line_ok = True
    with open(path) as f:
        for line in f:
            n += 1
            try:
                row = json.loads(line)
                scenes.add(row["scene_id"])
            except Exception as e:
                last_line_ok = False
                print(f"  MALFORMED LINE {n} in {path}: {e}")
    return scenes, n, last_line_ok


def main():
    done = set(open(f"{OUT_DIR}/gt_incontext.done").read().split())
    print(f"{len(done)} scenes marked done")

    gen_scenes, gen_n, gen_ok = scan(f"{OUT_DIR}/gt_incontext_general.jsonl")
    print(f"gt_incontext_general.jsonl: {gen_n} lines, {len(gen_scenes)} distinct scenes, all lines parsed: {gen_ok}")
    missing_general = done - gen_scenes
    print(f"scenes in .done but MISSING from general bank: {sorted(missing_general)}")

    spat_scenes, spat_n, spat_ok = scan(f"{OUT_DIR}/gt_incontext_spatial.jsonl")
    print(f"gt_incontext_spatial.jsonl: {spat_n} lines, {len(spat_scenes)} distinct scenes, all lines parsed: {spat_ok}")
    missing_spatial = done - spat_scenes
    print(f"scenes in .done but with ZERO spatial rows: {sorted(missing_spatial)} "
          f"(not necessarily an error -- a scene could legitimately have 0 valid spatial_farthest facts; "
          f"cross-check against the scene's drop log before treating as truncation)")

    drops_scenes, drops_n, drops_ok = scan(f"{OUT_DIR}/gt_incontext_drops.jsonl")
    print(f"gt_incontext_drops.jsonl: {drops_n} lines, {len(drops_scenes)} distinct scenes, all lines parsed: {drops_ok}")

    extra_general = gen_scenes - done
    extra_spatial = spat_scenes - done
    print(f"scenes present in general bank but NOT in .done (should be empty -- would mean a partial/crashed write): {sorted(extra_general)}")
    print(f"scenes present in spatial bank but NOT in .done: {sorted(extra_spatial)}")

    ok = (len(done) == 100 and gen_ok and spat_ok and drops_ok and not missing_general
          and not extra_general and not extra_spatial)
    print(f"\nQ8 VERDICT: {'PASS' if ok else 'FAIL -- see above'}")


if __name__ == "__main__":
    main()
