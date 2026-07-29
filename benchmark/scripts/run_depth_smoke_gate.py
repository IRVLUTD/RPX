"""Run one reproducible Depth smoke gate on an idle CUDA host.

Run this script from the model family's isolated Python environment. It
pins the RPX dataset revision, refuses the three unverified adapters,
checks GPU occupancy, records the environment, invokes the task CLI, and
validates the required result artefacts. It never uploads to Box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

DATASET_REPO = "IRVLUTD/RPX"
DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
BLOCKED_MODELS = {"fe2e", "d4rt", "gem-depth"}
IMAGE_MODELS = {
    "da-v2-large",
    "da3-metric-l",
    "depth-pro",
    "fe2e",
    "hyden",
    "lotus-2",
    "metric3d-v2",
    "moge-2-vit-l",
    "unidepth-v2",
    "zipdepth",
}
VIDEO_MODELS = {
    "chrono-depth",
    "d4rt",
    "da3-video",
    "depth-crafter",
    "gem-depth",
    "monst3r",
    "rolling-depth",
    "vggt-omega",
    "video-da",
    "vigeo",
}


def _default_output_root() -> Path:
    return Path(os.environ.get("RPX_SMOKE_ROOT", "./rpx_smoke_outputs")).expanduser()


def _nvidia_smi() -> str:
    """Find nvidia-smi on normal hosts and sandboxed desktop installs."""
    candidates = [
        shutil.which("nvidia-smi"),
        "/run/host/usr/bin/nvidia-smi",
        "/usr/bin/nvidia-smi",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise SystemExit("No nvidia-smi executable was found; GPU-only smoke cannot start.")


def _run_text(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def _gpu_state(gpu_index: int) -> tuple[list[dict], list[str], str]:
    smi = _nvidia_smi()
    try:
        rows = _run_text(
            [
                smi,
                "--query-gpu=index,name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit("No usable NVIDIA GPU was found; run this gate on a CUDA server.") from exc
    gpus = []
    for row in rows:
        index, name, total, used, driver = [part.strip() for part in row.split(",", 4)]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "driver_version": driver,
            }
        )
    if gpu_index < 0 or gpu_index >= len(gpus):
        raise SystemExit(f"--gpu-index {gpu_index} is invalid for {len(gpus)} visible GPU(s)")
    try:
        processes = _run_text(
            [
                smi,
                "-i",
                str(gpu_index),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Could not query selected-GPU compute processes; refusing to risk overlapping "
            "another team run."
        ) from exc
    processes = [
        line for line in processes if line.strip() and "no running processes" not in line.lower()
    ]
    return gpus, processes, smi


def _git_sha(path: Path, *, allow_image_revision: bool = False) -> str:
    """Resolve a Git SHA, with an explicit immutable-container fallback."""
    try:
        return _run_text(["git", "rev-parse", "HEAD"], cwd=path)
    except subprocess.CalledProcessError:
        revision = os.environ.get("RPX_GIT_SHA", "") if allow_image_revision else ""
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision
        raise


def _code_identity(path: Path) -> tuple[str, bool]:
    sha = _git_sha(path, allow_image_revision=True)
    try:
        status = _run_text(["git", "status", "--porcelain"], cwd=path)
    except subprocess.CalledProcessError:
        # Release containers intentionally omit .git. Their OCI label and
        # RPX_GIT_SHA environment value are injected from a clean source
        # commit by the build helper, so the image is an immutable identity.
        return sha, False
    if not status:
        return sha, False
    digest = hashlib.sha256()
    digest.update(_run_text(["git", "diff", "--binary", "HEAD"], cwd=path).encode())
    for rel in _run_text(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=path
    ).splitlines():
        file_path = path / rel
        digest.update(rel.encode())
        if file_path.is_file():
            digest.update(file_path.read_bytes())
    return f"{sha}-dirty-{digest.hexdigest()[:12]}", True


def _torch_metadata() -> dict:
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        return {"installed": False}
    except Exception as exc:  # CUDA initialization/driver mismatch
        return {"installed": True, "cuda_available": False, "error": str(exc)}


def _classify_failure(exc: BaseException) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "cuda oom" in message:
        return "cuda_oom"
    if (
        "401" in message
        or "403" in message
        or "gated" in message
        or "restricted" in message
        or "not authorized" in message
    ):
        return "access"
    if (
        "404" in message
        or "repository not found" in message
        or "weight download" in message
    ):
        return "weight_download"
    if "no module named" in message or "needs `" in message or "install with" in message:
        return "dependency"
    if "shape" in message or "dimension" in message or "dtype" in message:
        return "shape_dtype"
    if "manifest" in message or "dataset" in message or "decode" in message:
        return "data_manifest"
    if "from_pretrained" in message or "has no attribute" in message:
        return "upstream_api"
    return "code_or_unknown"


def _run_and_tee(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    tail: deque[str] = deque(maxlen=80)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                tail.append(line.rstrip())
            return_code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if return_code != 0:
        raise RuntimeError(f"smoke command exited with status {return_code}\n" + "\n".join(tail))



_VIDEO_CORE_METRICS = frozenset(
    {"absrel", "rmse", "delta1", "delta2", "delta3"}
)
_VIDEO_OPTIONAL_NONFINITE_METRICS = frozenset(
    {"tae", "opw", "tgm", "tcc"}
)


def _validate_metric_values(
    metrics: dict[str, object],
    *,
    task: str,
    context: str,
) -> list[str]:
    """Require finite core metrics while permitting documented video placeholders."""
    if not metrics:
        raise RuntimeError(f"{context} metrics are missing")

    optional_nonfinite: list[str] = []
    for key, value in metrics.items():
        finite = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        if finite:
            continue
        if (
            task == "video"
            and key in _VIDEO_OPTIONAL_NONFINITE_METRICS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            optional_nonfinite.append(key)
            continue
        raise RuntimeError(
            f"{context} metric {key!r} is missing or non-finite"
        )

    if task == "video":
        missing = sorted(_VIDEO_CORE_METRICS - set(metrics))
        if missing:
            raise RuntimeError(
                f"{context} is missing required video depth metrics: {missing}"
            )

    return optional_nonfinite


def _validate_outputs(
    out_dir: Path,
    *,
    expected_samples: int,
    expected_cells: int,
    expect_predictions: bool,
    expect_comprehensive: bool,
    task: str,
) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    required = [out_dir / "result.json", out_dir / "cells.parquet", out_dir / "summary.md"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing required artefacts: {missing}")
    payload = json.loads(required[0].read_text())
    if payload.get("num_samples") != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} samples, result has {payload.get('num_samples')}"
        )
    aggregated = payload.get("aggregated") or {}
    optional_nonfinite_metrics = set(
        _validate_metric_values(
            aggregated,
            task=task,
            context="aggregated",
        )
    )
    cells = pq.read_table(required[1]).to_pylist()
    if len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} cells, found {len(cells)}")
    for row in cells:
        if not row.get("gpu_name") or not row.get("precision"):
            raise RuntimeError("cell row is missing its GPU or precision SystemCard fields")
        metric_values = {
            key.removeprefix("metric:"): value
            for key, value in row.items()
            if key.startswith("metric:")
        }
        optional_nonfinite_metrics.update(
            _validate_metric_values(
                metric_values,
                task=task,
                context="cell",
            )
        )
    validation = {"num_samples": payload["num_samples"], "num_cells": len(cells)}
    if optional_nonfinite_metrics:
        validation["optional_nonfinite_metrics"] = sorted(
            optional_nonfinite_metrics
        )
    predictions = sorted((out_dir / "predictions").rglob("*.npz"))
    if expect_predictions and len(predictions) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} saved predictions, found {len(predictions)}"
        )
    if predictions:
        depth = np.load(predictions[0])["depth"].astype(np.float32)
        if not np.isfinite(depth).all() or float(np.std(depth)) <= 1e-8:
            raise RuntimeError("saved depth prediction is non-finite or degenerate")
        validation["prediction_checked"] = str(predictions[0])
        validation["num_predictions"] = len(predictions)
    if expect_comprehensive:
        comprehensive_path = out_dir / "comprehensive_metrics.json"
        if not comprehensive_path.is_file():
            raise RuntimeError("full image gate did not create comprehensive_metrics.json")
        comprehensive = json.loads(comprehensive_path.read_text())
        if len(comprehensive.get("per_sample") or []) != expected_samples:
            raise RuntimeError("comprehensive metric sample count does not match the split")
        comprehensive_aggregated = comprehensive.get("aggregated") or {}
        if not comprehensive_aggregated or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in comprehensive_aggregated.values()
        ):
            raise RuntimeError("comprehensive aggregated metrics are missing or non-finite")
        validation["comprehensive_metrics"] = str(comprehensive_path)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=["image", "video"])
    parser.add_argument("--gate", required=True, choices=["micro", "acceptance", "easy"])
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--upstream-dir", type=Path, action="append", default=[])
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="use Hugging Face Xet transport (standard HTTP is the safer default)",
    )
    args = parser.parse_args()

    roster = IMAGE_MODELS if args.task == "image" else VIDEO_MODELS
    if args.model not in roster:
        parser.error(f"{args.model!r} is not a canonical {args.task} model")
    if args.model in BLOCKED_MODELS:
        raise SystemExit(
            f"{args.model} is blocked because official weights are unverified; "
            "this launcher never bypasses the safety rail."
        )

    benchmark_dir = Path(__file__).resolve().parent.parent
    repo_root = benchmark_dir.parent
    sha = _git_sha(repo_root, allow_image_revision=True)
    code_id, git_dirty = _code_identity(repo_root)
    host = socket.gethostname()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root.resolve() / code_id / host / args.model / args.gate / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus, gpu_processes, smi = _gpu_state(args.gpu_index)
    if gpu_processes and not args.allow_busy_gpu:
        raise SystemExit(
            "GPU compute processes are already active; refusing to overlap a team run:\n"
            + "\n".join(gpu_processes)
        )

    command = [
        sys.executable,
        str(
            benchmark_dir
            / "scripts"
            / ("run_depth.py" if args.task == "image" else "run_video_depth.py")
        ),
        "--model",
        args.model,
        "--split",
        "easy",
        "--repo",
        DATASET_REPO,
        "--revision",
        DATASET_REVISION,
        "--device",
        "cuda",
        "--output-dir",
        str(out_dir),
    ]
    if args.cache_dir:
        # The CLIs honour HF_HOME; keeping the cache shared avoids duplicate weights/data.
        os.environ["HF_HOME"] = str(args.cache_dir)
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    smi_parent = str(Path(smi).parent)
    os.environ["PATH"] = smi_parent + os.pathsep + os.environ.get("PATH", "")
    torch_meta = _torch_metadata()
    if not torch_meta.get("installed") and "version" not in torch_meta:
        raise SystemExit("PyTorch is not installed in this model environment.")
    if not torch_meta.get("cuda_available"):
        raise SystemExit("PyTorch cannot see CUDA in this environment; CPU fallback is forbidden.")
    if args.task == "image":
        command += [
            "--batch-size",
            "1",
            "--use-official",
            "--save-predictions",
            "--skip-flops",
        ]
        if args.model == "zipdepth":
            # The official ReLU inverse-depth head may emit legitimate zeros.
            # Paper mode preserves them for the pooled disparity fit.
            command += ["--paper-protocol", "--defer-fscore"]
        if args.gate == "micro":
            command += ["--max-samples", "1"]
        elif args.gate == "acceptance":
            command += ["--max-samples", "25"]
        else:
            command += ["--comprehensive-metrics"]
    else:
        if args.gate == "micro":
            command += ["--max-samples", "1", "--frame-budget", "8", "--sampling", "stride"]
        elif args.gate == "acceptance":
            command += ["--max-samples", "1", "--frame-budget", "25", "--sampling", "stride"]
        else:
            command += ["--sampling", "all"]

    upstream_shas = {}
    for path in args.upstream_dir:
        upstream_shas[str(path.resolve())] = _git_sha(path.resolve())
    metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "model": args.model,
        "task": args.task,
        "gate": args.gate,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "rpx_git_sha": sha,
        "code_identity": code_id,
        "git_dirty": git_dirty,
        "upstream_git_shas": upstream_shas,
        "host": host,
        "platform": platform.platform(),
        "python": {"executable": sys.executable, "version": sys.version},
        "torch": torch_meta,
        "gpus": gpus,
        "gpu_processes_before_run": gpu_processes,
        "nvidia_smi": smi,
    }
    metadata_path = out_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (out_dir / "pip-freeze.txt").write_text(
        _run_text([sys.executable, "-m", "pip", "freeze"]) + "\n"
    )

    try:
        _run_and_tee(
            command,
            cwd=benchmark_dir,
            env=os.environ.copy(),
            log_path=out_dir / "run.log",
        )
        if args.gate == "micro":
            expected = 1
            expected_cells = 1
        elif args.gate == "acceptance":
            expected = 25 if args.task == "image" else 1
            expected_cells = 1
        else:
            expected = 24_750 if args.task == "image" else 99
            expected_cells = 99
        metadata["validation"] = _validate_outputs(
            out_dir,
            expected_samples=expected,
            expected_cells=expected_cells,
            expect_predictions=args.task == "image",
            expect_comprehensive=args.task == "image" and args.gate == "easy",
            task=args.task,
        )
        metadata["status"] = "passed"
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["failure_class"] = _classify_failure(exc)
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
