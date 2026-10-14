#!/usr/bin/env python3
"""Re-pin RPX VQA question lists to the dataset revision the toolkit accepts.

Two uses:

* default: build the packaged T5/T6 quick start from the Hub acceptance set;
* ``--source <hub path> --out-file <jsonl>``: republish any list under
  ``vqa/metadata/benchmark_subset_v1/`` as one validated file (issue #96).


The published ``vqa/metadata/benchmark_subset_v1/acceptance_104.jsonl`` points
its images at older Hub revisions (``main``, a release-candidate commit and the
``ego-preview-v2`` branch), which the toolkit's pinned contract rejects. This
script re-pins every image locator to the toolkit's ``DEFAULT_REVISION`` only
after proving that the whole tar shard it reads from has the same content hash
(Hub LFS SHA-256) at the original and the pinned revision, so every member is
byte-identical. Rows whose shard is missing at either revision are excluded and
listed in ``provenance.json`` (no locator is ever guessed or corrected). The
result is validated with ``load_manifest``.

Some in-context target locators store the MOS phase as a float path segment
(``scenes/scene083/2.0/rgb.tar``), a path that exists at no revision. Those are
rewritten to the real shard (``scenes/scene083/2/rgb.tar``) with the toolkit's
own ``normalize_shard`` before the shard check, so they are verified like any
other locator rather than dropped.

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

from rpx_benchmark.hub import DEFAULT_REPO_ID, DEFAULT_REVISION, VQA_MANIFEST_REVISION
from rpx_benchmark.vqa.contract import load_manifest, normalize_shard

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


def repin(rows: list[dict], api: HfApi) -> tuple[list[dict], list[dict], int, int]:
    """Re-pin every image locator to DEFAULT_REVISION after a whole-shard hash check.

    Returns (kept rows, excluded rows with reasons, shards verified, phase paths fixed).
    """
    fields = ("image", "reference_image")
    fixed = 0
    for row in rows:
        for field in fields:
            loc = row.get(field)
            if loc and normalize_shard(loc["shard"]) != loc["shard"]:
                row[field] = {**loc, "shard": normalize_shard(loc["shard"])}
                fixed += 1
    by_revision: dict[str, set[str]] = {}
    for row in rows:
        for field in fields:
            if row.get(field):
                by_revision.setdefault(row[field]["revision"], set()).add(row[field]["shard"])
    all_shards = set().union(*by_revision.values()) if by_revision else set()
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
    return kept, excluded, len(verified), fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=SOURCE, help="list to re-pin, as a path in the Hub dataset"
    )
    parser.add_argument(
        "--source-revision",
        default=VQA_MANIFEST_REVISION,
        help="Hub revision to read --source from",
    )
    parser.add_argument("--out", type=Path, default=OUT, help="quick-start folder (t5/t6 split)")
    parser.add_argument(
        "--out-file", type=Path, help="write one re-pinned list here instead of the quick start"
    )
    args = parser.parse_args()

    source = Path(
        hf_hub_download(
            DEFAULT_REPO_ID, args.source, repo_type="dataset", revision=args.source_revision
        )
    )
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    loadable = [row for row in rows if row.get("image")]
    rows, excluded, verified, fixed = repin(loadable, HfApi())

    provenance = {
        "source": f"hf://datasets/{DEFAULT_REPO_ID}@{args.source_revision}/{args.source}",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pinned_revision": DEFAULT_REVISION,
        "shards_verified_identical": verified,
        "phase_paths_fixed": fixed,
        "excluded_rows": excluded,
        "change": "image and reference_image revisions re-pinned; float phase path segments "
        "normalized; no other field edited; rows with unverifiable shards excluded",
        "builder": "benchmark/scripts/build_vqa_quickstart.py",
    }
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        samples = load_manifest(args.out_file)  # raises on any contract violation
        total = sum(1 for line in source.read_text().splitlines() if line.strip())
        provenance["rows"] = {
            "t5": sum(not sample.is_in_context for sample in samples),
            "t6": sum(sample.is_in_context for sample in samples),
        }
        provenance["rows_in_source"] = total
        provenance["rows_without_images_skipped"] = total - len(loadable)
        args.out_file.with_suffix(".provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        print(json.dumps({k: v for k, v in provenance.items() if k != "excluded_rows"}, indent=2))
        print(f"excluded rows: {len(excluded)}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for task, in_context in (("t5", False), ("t6", True)):
        selected = [row for row in rows if bool(row.get("reference_image")) == in_context]
        path = args.out / f"{task}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
        samples = load_manifest(path)  # raises on any contract violation
        assert all(sample.is_in_context == in_context for sample in samples)
        counts[task] = len(samples)
    provenance["rows"] = counts
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
