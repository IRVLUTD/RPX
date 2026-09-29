#!/usr/bin/env python3
"""Machine-readable RPX capability audit for the public GenCeption release."""

from __future__ import annotations

import argparse
import json

from genception_runtime import UPSTREAM_REPOSITORY, UPSTREAM_REVISION


def capability_report() -> dict:
    return {
        "model": "GenCeption",
        "variants": ["1.3b", "14b"],
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "supported": {
            "image_depth": "official depth prompt and decoder; one-frame clip adaptation",
            "video_depth": "official depth prompt and decoder",
            "tracking": "official referring-video-object-segmentation output",
            "vqa_regular": "native referring-expression mask converted to one bbox",
            "vqa_in_context": "reference phase followed by target phase; target mask converted to bbox",
        },
        "unavailable": {
            "camera_pose": (
                "the public pipeline has no pose prompt, camera-pose decoder, or pose demo; "
                "2D/3D keypoint tokens alone do not define an RPX camera transform"
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = capability_report()
    print(json.dumps(report, indent=2) if args.json else report)


if __name__ == "__main__":
    main()
