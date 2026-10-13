#!/usr/bin/env python3
"""Build the packaged T5/T6 quick-start question lists from the Hub acceptance set.

The published ``vqa/metadata/benchmark_subset_v1/acceptance_104.jsonl`` points
its images at older Hub revisions (``main``, a release-candidate commit and the
``ego-preview-v2`` branch), which the toolkit's pinned contract rejects. This
script re-pins every image locator to the toolkit's ``DEFAULT_REVISION`` only
after proving that the whole tar shard it reads from has the same content hash
(Hub LFS SHA-256) at the original and the pinned revision, so every member is
byte-identical. Rows whose shard is missing at either revision are excluded and
listed in ``provenance.json`` (no locator is ever guessed or corrected). The
result is validated with ``load_manifest``.

Output (committed, shipped in the wheel)::

    rpx_benchmark/vqa/quickstart_v1/t5.jsonl        one-image rows
    rpx_benchmark/vqa/quickstart_v1/t6.jsonl        two-image (in-context) rows
    rpx_benchmark/vqa/quickstart_v1/provenance.json source, hashes, checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from rpx_benchmark.hub import DEFAULT_REPO_ID, DEFAULT_REVISION
from rpx_benchmark.vqa.contract import load_manifest

SOURCE = "vqa/metadata/benchmark_subset_v1/acceptance_104.jsonl"
OUT = Path(__file__).resolve().parents[1] / "rpx_benchmark/vqa/quickstart_v1"


def shard_hashes(api: HfApi, revision: str, shards: set[str]) -> dict[str, str]:
    """Content hash of each tar shard at ``revision``; absent shards are omitted."""
    out: dict[str, str] = {}
    paths = sorted(shards)
    for i in range(0, len(paths), 50):
        for info in api.get_paths_info(
            DEFAULT_REPO_ID, paths[i : i + 50], repo_type="dataset", revision=revision
        ):
            out[info.path] = info.lfs.sha256 if info.lfs else info.blob_id
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    source = Path(
        hf_hub_download(DEFAULT_REPO_ID, SOURCE, repo_type="dataset", revision=DEFAULT_REVISION)
    )
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    api = HfApi()
    fields = ("image", "reference_image")
    by_revision: dict[str, set[str]] = {}
    for row in rows:
        for field in fields:
            if row.get(field):
                by_revision.setdefault(row[field]["revision"], set()).add(row[field]["shard"])
    all_shards = set().union(*by_revision.values())
    pinned = shard_hashes(api, DEFAULT_REVISION, all_shards)
    original = {rev: shard_hashes(api, rev, shards) for rev, shards in by_revision.items()}

    kept, excluded, verified = [], [], set()
    for row in rows:
        problems = []
        for field in fields:
            loc = row.get(field)
            if not loc:
                continue
            a, b = original[loc["revision"]].get(loc["shard"]), pinned.get(loc["shard"])
            if a is None or b is None:
                problems.append(
                    f"{field}: {loc['shard']} absent at {loc['revision'] if a is None else DEFAULT_REVISION}"
                )
            elif a != b:
                problems.append(
                    f"{field}: {loc['shard']} differs between {loc['revision']} and {DEFAULT_REVISION}"
                )
        if problems:
            excluded.append({"sample_id": row["sample_id"], "reasons": problems})
            continue
        for field in fields:
            if row.get(field):
                verified.add((row[field]["revision"], row[field]["shard"]))
                row[field] = {**row[field], "revision": DEFAULT_REVISION}
        kept.append(row)
    rows = kept

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for task, in_context in (("t5", False), ("t6", True)):
        selected = [row for row in rows if bool(row.get("reference_image")) == in_context]
        path = args.out / f"{task}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
        samples = load_manifest(path)  # raises on any contract violation
        assert all(sample.is_in_context == in_context for sample in samples)
        counts[task] = len(samples)

    provenance = {
        "source": f"hf://datasets/{DEFAULT_REPO_ID}@{DEFAULT_REVISION}/{SOURCE}",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pinned_revision": DEFAULT_REVISION,
        "rows": counts,
        "shards_verified_identical": len(verified),
        "excluded_rows": excluded,
        "change": "image and reference_image revisions re-pinned; no other field edited; rows with unverifiable shards excluded",
        "builder": "benchmark/scripts/build_vqa_quickstart.py",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
