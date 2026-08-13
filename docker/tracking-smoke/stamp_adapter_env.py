#!/usr/bin/env python3
"""Validate and stamp a model environment after installing an RPX overlay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--checkpoint-repo", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    args = parser.parse_args()

    manifest_path = Path(sys.prefix) / "rpx-environment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model": args.model,
        "source_revision": args.source_revision,
        "checkpoint_repo": args.checkpoint_repo,
        "checkpoint_revision": args.checkpoint_revision,
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"base model environment mismatch: {mismatches}")

    base_revision = manifest.get("rpx_git_sha")
    manifest["environment_base_rpx_git_sha"] = base_revision
    manifest["rpx_git_sha"] = os.environ["RPX_GIT_SHA"]
    manifest["adapter_overlay"] = True

    freeze_path = Path(sys.prefix) / "rpx-pip-freeze.txt"
    freeze_path.write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True
        ),
        encoding="utf-8",
    )
    manifest["pip_freeze"] = str(freeze_path)

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
