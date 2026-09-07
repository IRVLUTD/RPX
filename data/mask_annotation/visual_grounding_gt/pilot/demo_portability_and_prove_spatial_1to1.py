"""Q14: for 5 rows of every type, reconstruct Image 1 (the reference crop)
using ONLY fields stored in the canonical Parquet rows + a freshly-created,
empty temporary HF cache directory (HF_HOME override) -- proving the
locators are self-sufficient, not dependent on this machine's existing
cache. Reports whether revision is an immutable commit SHA or a mutable
branch name.

Q15: formal 1:1 set-equality proof between the canonical spatial Parquet
and published spatial_farthest, on (scene, phase, frame, reference identity,
answer identity, answer_bbox).
"""
import hashlib
import io
import os
import shutil
import sys
import tempfile

import pyarrow.parquet as pq
from PIL import Image

CANON_DIR = "out/release_candidate_v1"


def portability_demo():
    print("=== Q14: reference-image portability demo ===")
    tmp_cache = tempfile.mkdtemp(prefix="empty_hf_cache_")
    os.environ["HF_HOME"] = tmp_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = tmp_cache
    print(f"empty temp cache: {tmp_cache}")
    from huggingface_hub import hf_hub_download  # imported AFTER env override

    for name in ["incontext_mos_attribute_bbox", "incontext_ego_attribute_bbox", "incontext_mos_spatial_bbox"]:
        df = pq.read_table(f"{CANON_DIR}/{name}.parquet").to_pandas()
        for t in sorted(df.type.unique()):
            sub = df[df.type == t].sample(min(5, len(df[df.type == t])), random_state=11)
            print(f"\n--- {t} ({name}) ---")
            for _, row in sub.iterrows():
                ref = row["reference_image"]
                revision = ref["revision"]
                is_immutable = len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower())
                p = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=ref["shard"], revision=revision,
                                     cache_dir=tmp_cache)
                import tarfile
                with tarfile.open(p) as tf:
                    data = tf.extractfile(ref["member"]).read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                x0, y0, x1, y1 = ref["crop_bbox"]
                crop = img.crop((x0, y0, x1 + 1, y1 + 1))
                buf = io.BytesIO()
                crop.save(buf, format="PNG", compress_level=6)
                got_sha = hashlib.sha256(buf.getvalue()).hexdigest()
                expected_sha = row["reference_crop_sha256"]
                match = got_sha == expected_sha
                print(f"  sample_id={row['sample_id']} shard={ref['shard']} member={ref['member']} "
                      f"revision={revision} ({'IMMUTABLE commit SHA' if is_immutable else 'MUTABLE branch name'}) "
                      f"reconstructed_sha256_match={match}")
    shutil.rmtree(tmp_cache, ignore_errors=True)


def spatial_1to1_proof():
    print("\n=== Q15: canonical spatial Parquet vs published spatial_farthest, formal 1:1 proof ===")
    canon = pq.read_table(f"{CANON_DIR}/incontext_mos_spatial_bbox.parquet").to_pandas()
    pub = pq.read_table("out/vqa_parquet/spatial_bbox.parquet").to_pandas()
    pub = pub[pub.type == "spatial_farthest"]

    import json as jsonlib

    def canon_key(r):
        return (r["scene_id"], int(r["phase"]), r["frame"], r["evidence"]["reference"], r["answer"], tuple(r["answer_bbox"]))

    def pub_key(r):
        ev = jsonlib.loads(r["evidence"]) if isinstance(r["evidence"], str) else r["evidence"]
        return (r["scene_id"], int(r["phase"]), r["frame"], ev["reference"], r["answer"], tuple(r["answer_bbox"]))

    canon_keys = set(canon.apply(canon_key, axis=1))
    pub_keys = set(pub.apply(pub_key, axis=1))
    missing = pub_keys - canon_keys
    extra = canon_keys - pub_keys
    print(f"published spatial_farthest: {len(pub_keys)} distinct keys")
    print(f"canonical in-context spatial: {len(canon_keys)} distinct keys")
    print(f"missing (in published, not in canonical): {len(missing)}")
    print(f"extra (in canonical, not in published): {len(extra)}")
    print(f"1:1 EQUALITY: {'YES' if not missing and not extra and len(canon_keys) == len(pub_keys) else 'NO'}")


if __name__ == "__main__":
    portability_demo()
    spatial_1to1_proof()
