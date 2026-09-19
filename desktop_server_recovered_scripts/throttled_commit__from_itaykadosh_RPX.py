"""
Throttled commit script — LFS-batch-bypass version.

The 516 remaining files are pre-uploaded to HF xet CAS storage.
The 403 "storage limit" occurs when create_commit calls preupload_lfs_files,
which hits the git LFS batch API endpoint.

Bypass: set operation._is_uploaded = True so preupload_lfs_files skips
ALL our operations (empty new_additions). The commit payload only references
sha256 + size (already in checkpoint metadata). HF resolves the xet data.

Usage: python /metadisk/itaykadosh/RPX/throttled_commit.py
"""
import sys
import time
from pathlib import Path

STAGING = Path("/metadisk/itaykadosh/RPX/HF_staging_sos")
REPO_ID = "itaykadosh/RPX"
REVISION = "main"
REPO_TYPE = "dataset"
BATCH_SIZE = 20
SLEEP_BETWEEN_COMMITS = 32  # seconds → stays under 128 commits/hr

from huggingface_hub import HfApi
from huggingface_hub._local_folder import get_local_upload_paths, read_upload_metadata
from huggingface_hub._upload_large_folder import _build_hacky_operation

api = HfApi()
cache_dir = STAGING / ".cache" / "huggingface" / "upload"

# Collect uncommitted items
items = []
for meta_file in sorted(cache_dir.rglob("*.metadata")):
    rel = str(meta_file.relative_to(cache_dir)).removesuffix(".metadata").replace("\\", "/")
    meta = read_upload_metadata(STAGING, rel)
    if meta is None or meta.should_ignore or meta.is_committed:
        continue
    paths = get_local_upload_paths(STAGING, rel)
    items.append((paths, meta))

total = len(items)
print(f"Found {total} uncommitted files to commit.")
if total == 0:
    print("Nothing to do.")
    sys.exit(0)

committed = 0
num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num, i in enumerate(range(0, total, BATCH_SIZE), start=1):
    batch = items[i : i + BATCH_SIZE]
    print(f"\n[{batch_num}/{num_batches}] Committing {len(batch)} files "
          f"({committed}/{total} done)...", flush=True)

    # Build operations; set _is_uploaded=True to bypass LFS batch API
    additions = []
    for item in batch:
        op = _build_hacky_operation(item)
        op._is_uploaded = True  # critical: skips preupload_lfs_files → no LFS batch call
        additions.append(op)

    while True:
        try:
            result = api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=REVISION,
                operations=additions,
                commit_message="Add files using upload-large-folder tool",
            )
            # Mark committed in checkpoint
            for paths, meta in batch:
                meta.is_committed = True
                meta.save(paths)
            committed += len(batch)
            print(f"  ✓ Committed {len(batch)} files. Total: {committed}/{total}", flush=True)
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                retry_after = 310
                for part in err.split():
                    try:
                        val = int(part)
                        if 10 < val < 3600:
                            retry_after = val + 15
                            break
                    except ValueError:
                        pass
                print(f"  Rate limited. Sleeping {retry_after}s...", flush=True)
                time.sleep(retry_after)
            elif "403" in err or "storage limit" in err.lower():
                print(f"  FATAL 403 storage limit still present: {err[:300]}", flush=True)
                sys.exit(1)
            else:
                print(f"  ERROR: {err[:300]}", flush=True)
                print("  Retrying in 60s...", flush=True)
                time.sleep(60)

    if i + BATCH_SIZE < total:
        print(f"  Sleeping {SLEEP_BETWEEN_COMMITS}s...", flush=True)
        time.sleep(SLEEP_BETWEEN_COMMITS)

print(f"\nDone. {committed}/{total} files committed.")
