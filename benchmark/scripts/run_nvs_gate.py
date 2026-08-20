"""Run and validate reproducible RPX NVS smoke, micro, or acceptance gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GATES = {"smoke": 1, "micro": 5, "acceptance": 25}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", choices=tuple(GATES), required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="easy", choices=("easy", "medium", "hard"))
    args = parser.parse_args()

    count = GATES[args.gate]
    run_script = Path(__file__).with_name("run_nvs.py")
    gate_root = args.output_root / args.model / args.gate
    command = [
        sys.executable,
        str(run_script),
        "--model", args.model,
        "--split", args.split,
        "--device", args.device,
        "--max-samples", str(count),
        "--context-counts", "2",
        "--sample-types", "extrapolation",
        "--sample-order", "scene_round_robin",
        "--extracted-root", str(args.extracted_root),
        "--parquet-path", str(args.parquet_path),
        "--results-root", str(gate_root),
        "--save-predictions",
        "--strict-io",
    ]
    subprocess.run(command, check=True)

    display = "DepthSplat" if args.model == "depthsplat" else args.model
    result_path = gate_root / display / args.split / "result.json"
    frames_root = gate_root / display / args.split / "prediction_frames"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    png_paths = list(frames_root.rglob("*.png"))
    png_count = len(png_paths)
    if result.get("num_samples") != count or png_count != count:
        raise SystemExit(
            f"gate validation failed: samples={result.get('num_samples')} "
            f"PNGs={png_count}, expected={count}"
        )
    if any("__extrapolation__" not in path.stem for path in png_paths):
        raise SystemExit("gate validation failed: a target is not labeled extrapolation")
    protocol = result.get("sampling_protocol", {})
    if protocol.get("context_counts") != [2] or protocol.get("sample_types") != [
        "extrapolation"
    ]:
        raise SystemExit(f"gate validation failed: wrong sampling protocol: {protocol}")
    for metric in ("psnr", "ssim"):
        if metric not in result.get("aggregated", {}):
            raise SystemExit(f"gate validation failed: missing {metric}")
    print(f"RPX {args.model} {args.gate} gate: PASS ({count} real {args.split} samples)")


if __name__ == "__main__":
    main()
