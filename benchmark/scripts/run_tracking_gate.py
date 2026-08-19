#!/usr/bin/env python3
"""Run one bounded, real-data RPX D3 gate and prove prediction resume."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from run_tracking import DEFAULT_DATASET_REPO, PINNED_DATASET_REVISION
from tracking_models.cutie_tracker import CUTIE_MODEL_ID, CUTIE_MODEL_REVISION
from tracking_models.dam4sam_tracker import (
    DAM4SAM_MODEL_ID,
    DAM4SAM_MODEL_REVISION,
    DAM4SAM_SOURCE_REVISION,
)
from tracking_models.edgetam_tracker import (
    EDGETAM_MODEL_ID,
    EDGETAM_MODEL_REVISION,
)
from tracking_models.mits_tracker import (
    MITS_MODEL_ID,
    MITS_MODEL_REVISION,
    MITS_SOURCE_REVISION,
)
from tracking_models.ovtr_tracker import (
    OVTR_MODEL_ID,
    OVTR_MODEL_REVISION,
    OVTR_SOURCE_REVISION,
)
from tracking_models.sam2_plus_tracker import (
    SAM2_PLUS_MODEL_ID,
    SAM2_PLUS_MODEL_REVISION,
)
from tracking_models.sam2_tracker import SAM2_MODEL_ID, SAM2_MODEL_REVISION
from tracking_models.sam2long_tracker import (
    SAM2LONG_MODEL_ID,
    SAM2LONG_MODEL_REVISION,
)
from tracking_models.xmem_tracker import (
    XMEM_MODEL_ID,
    XMEM_MODEL_REVISION,
    XMEM_SOURCE_REVISION,
)

GATE_FRAMES = {"smoke": 2, "micro": 8, "acceptance": 25}
MODEL_PROVENANCE = {
    "dam4sam": {
        "source_revision": DAM4SAM_SOURCE_REVISION,
        "checkpoint_repo": DAM4SAM_MODEL_ID,
        "checkpoint_revision": DAM4SAM_MODEL_REVISION,
    },
    "sam2": {
        "source_revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "checkpoint_repo": SAM2_MODEL_ID,
        "checkpoint_revision": SAM2_MODEL_REVISION,
    },
    "edgetam": {
        "source_revision": "7711e012a30a2402c4eaab637bdb00a521302c91",
        "checkpoint_repo": EDGETAM_MODEL_ID,
        "checkpoint_revision": EDGETAM_MODEL_REVISION,
    },
    "cutie": {
        "source_revision": "ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a",
        "checkpoint_repo": CUTIE_MODEL_ID,
        "checkpoint_revision": CUTIE_MODEL_REVISION,
    },
    "sam2long": {
        "source_revision": "7193b77fa0c8827e0520ab281acd2cf394ab898e",
        "checkpoint_repo": SAM2LONG_MODEL_ID,
        "checkpoint_revision": SAM2LONG_MODEL_REVISION,
    },
    "sam2-plus": {
        "source_revision": "09c9ec4686d7170396ed98abcc0150f35e82c6b9",
        "checkpoint_repo": SAM2_PLUS_MODEL_ID,
        "checkpoint_revision": SAM2_PLUS_MODEL_REVISION,
    },
    "mits": {
        "source_revision": MITS_SOURCE_REVISION,
        "checkpoint_repo": MITS_MODEL_ID,
        "checkpoint_revision": MITS_MODEL_REVISION,
    },
    "xmem": {
        "source_revision": XMEM_SOURCE_REVISION,
        "checkpoint_repo": XMEM_MODEL_ID,
        "checkpoint_revision": XMEM_MODEL_REVISION,
    },
    "ovtr": {
        "source_revision": OVTR_SOURCE_REVISION,
        "checkpoint_repo": OVTR_MODEL_ID,
        "checkpoint_revision": OVTR_MODEL_REVISION,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PROVENANCE), required=True)
    parser.add_argument("--gate", choices=sorted(GATE_FRAMES), required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--manifest-path")
    parser.add_argument("--dataset-workers", type=int, default=8)
    return parser.parse_args()


def _validate_image_environment(model: str) -> str:
    revision = os.environ.get("RPX_GIT_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(
            "RPX_GIT_SHA must be the full 40-character commit embedded at image build."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run the gate with Docker --gpus all.")

    manifest_path = Path(sys.prefix) / "rpx-environment.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing model environment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model": model,
        "rpx_git_sha": revision,
        **MODEL_PROVENANCE[model],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"{model} environment provenance mismatch: {mismatches}")
    return revision


def main() -> None:
    args = _parse_args()
    revision = _validate_image_environment(args.model)
    frames = GATE_FRAMES[args.gate]
    output_dir = Path(args.output_root) / revision / args.model / args.gate
    script_dir = Path(__file__).resolve().parent

    common = [
        sys.executable,
        str(script_dir / "run_tracking.py"),
        "--model",
        args.model,
        "--split",
        "easy",
        "--repo",
        args.repo,
        "--revision",
        args.revision,
        "--cache-dir",
        args.cache_dir,
        "--output-dir",
        str(output_dir),
        "--save-predictions",
        "--max-clips",
        "1",
        "--max-frames",
        str(frames),
        "--dataset-workers",
        str(args.dataset_workers),
    ]
    if args.manifest_path:
        common.extend(["--manifest-path", args.manifest_path])

    # The first pass always performs inference, even when this gate was run before.
    subprocess.run(common, check=True)
    # The second pass must reuse every persisted prediction and perform zero forwards.
    subprocess.run([*common, "--resume-predictions"], check=True)
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "validate_tracking_smoke.py"),
            "--output-dir",
            str(output_dir),
            "--model",
            args.model,
            "--expected-clips",
            "1",
            "--expected-frames",
            str(frames),
            "--expected-rpx-revision",
            revision,
            "--require-resume-hit",
        ],
        check=True,
    )
    if args.gate == "acceptance":
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "render_tracking_predictions.py"),
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                args.cache_dir,
            ],
            check=True,
        )
    print(f"RPX {args.model} {args.gate} gate: PASS ({frames} real Easy frames)")


if __name__ == "__main__":
    main()
