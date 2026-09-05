#!/usr/bin/env python3
"""Fetch smoke RGBs and emit model-neutral inference requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpx_benchmark.vqa.contract import ATTRIBUTE_TYPES, BBOX_TYPES, load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_rgb, image_cache_name
from rpx_benchmark.vqa.prompts import build_prompt
from rpx_benchmark.vqa.roster import get_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()
    model = get_model(args.model)
    samples = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            task = (
                "bbox"
                if sample.question_type in BBOX_TYPES
                else "attribute"
                if sample.question_type in ATTRIBUTE_TYPES
                else "binary"
            )
            if task not in model.capabilities:
                continue
            image_path = args.image_cache / image_cache_name(sample)
            if not args.no_fetch:
                image_path = fetch_rgb(sample, args.image_cache)
            prompt = build_prompt(sample, model.key)
            request = {
                "sample_id": sample.sample_id,
                "image_path": str(image_path.resolve()),
                "prompt": prompt.text,
                "max_new_tokens": prompt.max_new_tokens,
                "output_kind": prompt.output_kind,
                "generation": {"do_sample": False, "num_beams": 1},
            }
            handle.write(json.dumps(request, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
