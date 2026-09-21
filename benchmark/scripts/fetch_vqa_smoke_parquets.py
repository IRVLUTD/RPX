#!/usr/bin/env python3
"""Fetch the five current bbox VQA parquets at a frozen Hub revision: the two
normal (single-image) parquets and the three in-context (two-image) ones."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

# vqa/attribute.parquet, vqa/spatial_bbox.parquet, and the three
# incontext_*.parquet files were all published together at this revision.
DATASET_REVISION = "5a1702652bc16fc0a6b2699a78cc85a4582a3b93"

NORMAL_PARQUETS = ("attribute.parquet", "spatial_bbox.parquet")
INCONTEXT_PARQUETS = (
    "incontext_mos_attribute_bbox.parquet",
    "incontext_ego_attribute_bbox.parquet",
    "incontext_mos_spatial_bbox.parquet",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in (*NORMAL_PARQUETS, *INCONTEXT_PARQUETS):
        source = hf_hub_download(
            repo_id="anonymous/RPX",
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=f"vqa/{name}",
        )
        target = args.out / name
        if not target.exists():
            shutil.copy2(source, target)
        print(target)


if __name__ == "__main__":
    main()
