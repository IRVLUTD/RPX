"""Run and validate reproducible RPX NVS smoke, micro, or acceptance gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GATES = {"smoke": 1, "micro": 5, "acceptance": 25}


def _model_metadata(model: str) -> tuple[str, int]:
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from nvs_models import MODEL_DISPLAY_NAMES, MODEL_SPECS  # noqa: PLC0415

    display = MODEL_DISPLAY_NAMES.get(model, model)
    context_count = MODEL_SPECS[model].gate_context_count if model in MODEL_SPECS else 2
    return display, context_count


def _gate_command(args: argparse.Namespace, count: int, context_count: int) -> list[str]:
    run_script = Path(__file__).with_name("run_nvs.py")
    gate_root = args.output_root / args.model / args.gate
    return [
        sys.executable,
        str(run_script),
        "--model", args.model,
        "--split", args.split,
        "--device", args.device,
        "--max-samples", str(count),
        "--context-counts", str(context_count),
        "--sample-types", "extrapolation",
        "--sample-order", "scene_round_robin",
        "--extracted-root", str(args.extracted_root),
        "--parquet-path", str(args.parquet_path),
        "--results-root", str(gate_root),
        "--save-predictions",
        "--strict-io",
    ]


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
    gate_root = args.output_root / args.model / args.gate
    display, context_count = _model_metadata(args.model)
    command = _gate_command(args, count, context_count)
    subprocess.run(command, check=True)

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
    if protocol.get("context_counts") != [context_count] or protocol.get("sample_types") != [
        "extrapolation"
    ]:
        raise SystemExit(f"gate validation failed: wrong sampling protocol: {protocol}")
    for metric in ("psnr", "ssim"):
        if metric not in result.get("aggregated", {}):
            raise SystemExit(f"gate validation failed: missing {metric}")
    print(f"RPX {args.model} {args.gate} gate: PASS ({count} real {args.split} samples)")


if __name__ == "__main__":
    main()
