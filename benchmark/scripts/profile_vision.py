#!/usr/bin/env python3
"""Run any vision task through the same fresh-inference hardware measurement.

Arguments after -- are passed to its existing task runner. Run in that model's
existing environment/image. A fresh output directory is mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpx_benchmark.hardware_profile import ProfileComplete, atomic_json

RUNNERS = {
    "image-depth": ("run_depth.py", "frame"),
    "video-depth": ("run_video_depth.py", "frame"),
    "tracking": ("run_tracking.py", "frame"),
    "rcpe": ("run_relative_pose.py", "pair"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=RUNNERS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--flop-calls", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate native imports/CLI and hook installation; never instantiate a model",
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup_calls < 1 or args.flop_calls < 0:
        parser.error("samples/warmup must be positive; flop-calls must be nonnegative")
    forwarded = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    forbidden = {
        "--output-dir",
        "--model",
        "--split",
        "--save-predictions",
        "--resume-predictions",
        "--upload-to-box",
        "--comprehensive-metrics",
    }
    if any(value.split("=", 1)[0] in forbidden for value in forwarded):
        parser.error(
            "Do not override model/split/output or save/resume/upload predictions in a fresh profile"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script, unit = RUNNERS[args.task]
    if args.task in {"image-depth", "rcpe"} and not any(
        v.split("=", 1)[0] == "--batch-size" for v in forwarded
    ):
        forwarded = ["--batch-size", "1", *forwarded]
    command = [
        str(Path(__file__).with_name(script)),
        "--model",
        args.model,
        "--split",
        args.split,
        "--output-dir",
        str(output / "scratch"),
        *forwarded,
    ]
    config = {
        "task": args.task,
        "model": args.model,
        "split": args.split,
        "unit": unit,
        "samples": args.samples,
        "warmup_calls": args.warmup_calls,
        "flop_calls": args.flop_calls,
        "output_dir": str(output),
        "command": command,
        "dataset_protocol": next(
            (
                value.split("=", 1)[1] if "=" in value else forwarded[index + 1]
                for index, value in enumerate(forwarded)
                if value.split("=", 1)[0] == "--dataset-protocol"
                and ("=" in value or index + 1 < len(forwarded))
            ),
            "mos" if args.task == "tracking" else None,
        ),
        "container_image_id": os.environ.get("RPX_PROFILE_IMAGE_ID"),
        "host_gpu_index": os.environ.get("RPX_PROFILE_HOST_GPU"),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "profile_vision.py": Path(__file__),
                "hardware_profile.py": Path(__file__).parents[1]
                / "rpx_benchmark/hardware_profile.py",
                "hardware_runtime_hooks.py": Path(__file__).with_name("hardware_runtime_hooks.py"),
                "runner.py": Path(__file__).parents[1] / "rpx_benchmark/runner.py",
                "_video_pipeline.py": Path(__file__).parents[1]
                / "rpx_benchmark/tasks/_video_pipeline.py",
                script: Path(command[0]),
            }.items()
        },
    }
    config_path = output / "profile_config.json"
    atomic_json(config_path, config)
    # Do not enable earlier source-injected profiling at the same time as hooks.
    previous_profile_env = os.environ.pop("RPX_HARDWARE_PROFILE_CONFIG", None)
    previous_argv = sys.argv
    sys.argv = command
    from hardware_runtime_hooks import RuntimeHooks

    hooks = RuntimeHooks(config)
    original_parse = argparse.ArgumentParser.parse_args

    class ParsedPreflight(BaseException):
        pass

    def preflight_parse(parser, *parse_args, **parse_kwargs):
        parsed = original_parse(parser, *parse_args, **parse_kwargs)
        atomic_json(
            output / "native_arguments.json", json.loads(json.dumps(vars(parsed), default=str))
        )
        raise ParsedPreflight()

    try:
        hooks.install()
        if args.preflight_only:
            argparse.ArgumentParser.parse_args = preflight_parse
        runpy.run_path(command[0], run_name="__main__")
        hooks.finish()
    except ParsedPreflight:
        atomic_json(
            output / "preflight.json",
            {
                "status": "imports_and_cli_passed",
                "model_inference_tested": False,
                "checkpoint_access_tested": False,
                "config": config,
            },
        )
        print(f"Preflight passed (no inference): {output}")
    except ProfileComplete:
        report = json.loads((output / "hardware_profile.json").read_text())
        print(f"Hardware profile: {report['status']}; {output / 'hardware_profile.json'}")
        if report["status"] != "complete":
            raise SystemExit(
                "Dataset exhausted before the measured sample budget; see measured_units"
            ) from None
    except BaseException as exc:
        atomic_json(output / "profile_failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    else:
        raise SystemExit("Runner completed without satisfying the hardware profiling budget")
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        hooks.close()
        sys.argv = previous_argv
        if previous_profile_env is not None:
            os.environ["RPX_HARDWARE_PROFILE_CONFIG"] = previous_profile_env


if __name__ == "__main__":
    main()
