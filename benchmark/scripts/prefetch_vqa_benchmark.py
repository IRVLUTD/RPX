#!/usr/bin/env python3
"""Fetch and verify every image required by a frozen VQA manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images_many


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="inventory missing tar members and emit an audited runnable subset",
    )
    parser.add_argument("--available-manifest", type=Path)
    parser.add_argument("--missing-report", type=Path)
    args = parser.parse_args()

    if args.allow_missing and not (args.available_manifest and args.missing_report):
        parser.error("--allow-missing requires --available-manifest and --missing-report")

    samples = load_manifest(args.manifest)
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(
        f"prefetching {len(samples)} rows; manifest_sha256={digest}",
        flush=True,
    )
    missing_entries: list[dict[str, str]] | None = [] if args.allow_missing else None
    paths = fetch_images_many(
        samples,
        args.image_cache,
        missing_report=missing_entries,
    )
    missing = [sample.sample_id for sample in samples if sample.sample_id not in paths]
    if args.allow_missing:
        assert missing_entries is not None
        missing_ids = set(missing)
        raw_rows = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(raw_rows) != len(samples):
            raise SystemExit("manifest row count changed while creating available subset")
        args.available_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.available_manifest.open("w", encoding="utf-8") as handle:
            for sample, row in zip(samples, raw_rows, strict=True):
                if sample.sample_id not in missing_ids:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        report = {
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": digest,
            "source_rows": len(samples),
            "available_manifest": str(args.available_manifest),
            "available_rows": len(paths),
            "excluded_rows": len(missing),
            "missing_locator_count": len(missing_entries),
            "missing_locators": missing_entries,
        }
        args.missing_report.parent.mkdir(parents=True, exist_ok=True)
        args.missing_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status = (
            "PREFETCH COMPLETE WITH AUDITED EXCLUSIONS"
            if missing
            else "PREFETCH COMPLETE; AUDITED MANIFEST WRITTEN"
        )
        print(
            f"{status}: {len(paths)}/{len(samples)} rows; "
            f"excluded={len(missing)}; report={args.missing_report}",
            flush=True,
        )
        return
    if missing:
        raise SystemExit(f"prefetch coverage failure: {len(missing)} missing")
    print(f"PREFETCH COMPLETE: {len(paths)}/{len(samples)} rows", flush=True)


if __name__ == "__main__":
    main()
