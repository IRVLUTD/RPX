#!/usr/bin/env python3
"""Run and validate a small, real-data RPX RCPE gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

GATE_SIZES = {"smoke": 1, "micro": 5, "acceptance": 25}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", choices=tuple(GATE_SIZES), required=True)
    parser.add_argument("--split", default="easy")
    parser.add_argument("--repo", default="IRVLUTD/RPX")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    count = GATE_SIZES[args.gate]
    output = args.output_root / args.revision / args.model / args.gate
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_relative_pose.py")),
        "--model", args.model,
        "--split", args.split,
        "--repo", args.repo,
        "--revision", args.revision,
        "--device", args.device,
        "--output-dir", str(output),
        "--pairs-source", "on_the_fly",
        "--max-samples", str(count),
        "--batch-size", "1",
        "--save-predictions",
        "--skip-flops",
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())

    required = [
        output / "result.json",
        output / "summary.md",
        output / "rcpe_metrics.json",
        output / "predictions.csv",
        output / "pairs_manifest.json",
        output / "cells.parquet",
        output / "phi_jedi.json",
        output / "pose_comprehensive_metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"gate output is incomplete; missing: {missing}")

    with (output / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != count:
        raise SystemExit(f"expected {count} predictions, found {len(rows)}")
    for row in rows:
        for key in ("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22", "tx", "ty", "tz"):
            value = float(row[key])
            if not (-float("inf") < value < float("inf")):
                raise SystemExit(f"non-finite prediction value in {key}")

    metrics = json.loads((output / "rcpe_metrics.json").read_text())
    if int(metrics.get("n_pairs", 0)) != count:
        raise SystemExit(
            f"expected rcpe_metrics n_pairs={count}, got {metrics.get('n_pairs')}"
        )
    manifest = json.loads((output / "pairs_manifest.json").read_text())
    samples = manifest["samples"]
    for sample in samples:
        if sample.get("pair_type") in {"intra_phase", "temporal_chain"}:
            gap = int(sample["frame_idx_b"]) - int(sample["frame_idx"])
            if gap != 5:
                raise SystemExit(f"non-canonical frame gap {gap} in {sample['id']}")
    if args.gate in {"micro", "acceptance"}:
        pair_types = {sample.get("pair_type") for sample in samples}
        phases = {
            int(sample["phase"])
            for sample in samples
            if sample.get("pair_type") == "intra_phase"
        }
        if pair_types != {"intra_phase", "cross_phase", "temporal_chain"}:
            raise SystemExit(f"gate lacks pair-type coverage: {sorted(pair_types)}")
        if phases != {0, 2}:
            raise SystemExit(f"gate lacks phase-0/2 intra coverage: {sorted(phases)}")
    print(f"RPX RCPE {args.model} {args.gate} gate: PASS ({count} real pairs)")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
