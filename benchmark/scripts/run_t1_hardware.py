#!/usr/bin/env python3
"""Run all T1 paper models serially through the RPX hardware protocol."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PLAN = Path(__file__).parents[1] / "configs/hardware/t1-image-depth.json"
REPORT_COLUMNS = (
    "model",
    "samples",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_frames_per_s",
    "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib",
    "params_m",
    "flops_g_per_frame",
    "gpu_name",
    "image",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "rpx.hardware.t1.v1":
        raise SystemExit(f"Unsupported hardware plan schema in {path}")
    jobs = plan.get("jobs", [])
    expected = {
        "da-v2-large",
        "depth-pro",
        "unidepth-v2",
        "moge-2-vit-l",
        "da3-metric-l",
        "metric3d-v2",
        "lotus-2",
        "hyden",
        "fe2e",
        "zipdepth",
    }
    names = [job.get("model") for job in jobs]
    if len(names) != len(set(names)) or set(names) != expected:
        raise SystemExit("T1 plan must contain each of the ten paper models exactly once")
    for job in jobs:
        image = job.get("image", "")
        if "@sha256:" not in image or not job.get("python", "").startswith("/"):
            raise SystemExit(f"Unpinned image or invalid Python path for {job.get('model')}")
    return plan


def fingerprint(plan_path: Path, plan: dict, job: dict, samples: int, repo: Path) -> str:
    files = (
        plan_path,
        repo / "docker/run_vision_profile.sh",
        repo / "benchmark/scripts/profile_vision.py",
        repo / "benchmark/scripts/hardware_runtime_hooks.py",
        repo / "benchmark/rpx_benchmark/hardware_profile.py",
    )
    payload = {
        "job": job,
        "dataset_revision": plan["dataset_revision"],
        "split": plan["split"],
        "samples": samples,
        "warmup_calls": plan["warmup_calls"],
        "flop_calls": plan["flop_calls"],
        "files": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def completed(state: dict, signature: str, samples: int) -> bool:
    if state.get("status") != "complete" or state.get("fingerprint") != signature:
        return False
    report_path = Path(state.get("profile", ""))
    report = read_json(report_path)
    return (
        report.get("schema_version") == "rpx.hardware.v1"
        and report.get("status") == "complete"
        and report.get("measured_units", 0) >= samples
    )


def gpu_uuid(index: int) -> str:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    devices = {int(row[0].strip()): row[1].strip() for row in csv.reader(io.StringIO(text))}
    if index not in devices:
        raise SystemExit(f"GPU {index} is unavailable; detected indices: {sorted(devices)}")
    return devices[index]


def assert_idle(uuid: str) -> None:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    busy = [row for row in csv.reader(io.StringIO(text)) if row and row[0].strip() == uuid]
    if busy:
        raise RuntimeError(f"Selected GPU has another compute process; nothing was killed: {busy}")


def dataset_repository(cache: Path, revision: str) -> Path:
    relative = Path("datasets--IRVLUTD--RPX")
    candidates = (cache / "hub" / relative, cache / relative)
    for repository in candidates:
        manifest = repository / "snapshots" / revision / "manifests/monocular_depth/easy.json"
        if manifest.is_file():
            return repository
    expected = candidates[0] / "snapshots" / revision / "manifests/monocular_depth/easy.json"
    raise SystemExit(f"Pinned T1 manifest is absent from RPX_HF_CACHE: {expected}")


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def write_host_inventory(root: Path, gpu: int, uuid: str, repo: Path) -> None:
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit",
            "--format=csv,noheader,nounits",
        ],
        "gpu_clock": [
            "nvidia-smi",
            "--query-gpu=index,clocks.current.graphics,clocks.current.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        "cpu": ["lscpu", "--json"],
        "memory": ["free", "-b"],
        "docker": ["docker", "version", "--format", "{{json .}}"],
    }
    inventory = {
        "schema": "rpx.hardware.host.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "selected_gpu_index": gpu,
        "selected_gpu_uuid": uuid,
        "rpx_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "commands": {name: command_output(command) for name, command in commands.items()},
    }
    atomic_json(root / "host_system.json", inventory)


def pull_and_check_images(plan: dict, no_pull: bool) -> None:
    for image in dict.fromkeys(job["image"] for job in plan["jobs"]):
        if not no_pull:
            print(f"PULL {image}", flush=True)
            subprocess.run(["docker", "pull", image], check=True)
        subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)
    for job in plan["jobs"]:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--entrypoint",
                "/bin/sh",
                job["image"],
                "-c",
                'test -x "$1"',
                "sh",
                job["python"],
            ],
            check=True,
        )


def aggregate(plan: dict, root: Path) -> list[dict]:
    rows = []
    for job in plan["jobs"]:
        state = read_json(root / "jobs" / f"{job['model']}.json")
        report = read_json(Path(state.get("profile", "")))
        if state.get("status") != "complete" or report.get("status") != "complete":
            continue
        rows.append(
            {
                "model": job["model"],
                "samples": report.get("measured_units"),
                "latency_mean_ms": report.get("latency_mean_ms"),
                "latency_p50_ms": report.get("latency_p50_ms"),
                "latency_p95_ms": report.get("latency_p95_ms"),
                "latency_p99_ms": report.get("latency_p99_ms"),
                "throughput_frames_per_s": report.get("throughput_units_per_s"),
                "peak_cuda_allocated_mib": report.get("peak_cuda_allocated_mib"),
                "peak_cuda_reserved_mib": report.get("peak_cuda_reserved_mib"),
                "params_m": report.get("params_m"),
                "flops_g_per_frame": report.get("flops_g_per_unit"),
                "gpu_name": report.get("system", {}).get("gpu_name"),
                "image": job["image"],
            }
        )
    atomic_json(
        root / "T1_hardware_metrics.json",
        {
            "schema": "rpx.hardware.t1.results.v1",
            "plan": plan,
            "results": rows,
        },
    )
    csv_path = root / "T1_hardware_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def show_status(plan: dict, root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    print(f"{'MODEL':20} {'STATUS':12} MEASURED")
    for job in plan["jobs"]:
        state = read_json(root / "jobs" / f"{job['model']}.json")
        status = state.get("status", "pending")
        report = read_json(Path(state.get("profile", "")))
        measured = report.get("measured_units", 0)
        print(f"{job['model']:20} {status:12} {measured}")
        if status == "failed":
            print(f"  {state.get('error')}\n  log: {state.get('log')}")
        counts[status] = counts.get(status, 0) + 1
    print(counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    samples = args.samples if args.samples is not None else int(plan["samples"])
    if samples < 1:
        parser.error("samples must be positive")
    root = args.output_root.resolve()
    if args.mode == "status":
        show_status(plan, root)
        return

    cache_value = os.environ.get("RPX_HF_CACHE", "")
    if not cache_value or not Path(cache_value).is_dir():
        raise SystemExit("Set RPX_HF_CACHE to the existing Hugging Face cache directory")
    cache = Path(cache_value).resolve()
    dataset_repo = dataset_repository(cache, plan["dataset_revision"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    lock = (root / ".suite.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another T1 suite is already using this output directory") from None

    uuid = gpu_uuid(args.gpu)
    assert_idle(uuid)
    write_host_inventory(root, args.gpu, uuid, repo)
    pull_and_check_images(plan, args.no_pull)
    atomic_json(root / "plan-used.json", {**plan, "samples": samples})
    failures = 0
    for job in plan["jobs"]:
        state_path = root / "jobs" / f"{job['model']}.json"
        state = read_json(state_path)
        signature = fingerprint(plan_path, plan, job, samples, repo)
        if completed(state, signature, samples):
            print(f"SKIP complete: {job['model']}", flush=True)
            continue
        if state.get("status") in {"failed", "running"} and not args.retry_failed:
            print(f"SKIP {state['status']}: {job['model']} (use --retry-failed)", flush=True)
            failures += 1
            if not args.keep_going:
                break
            continue
        assert_idle(uuid)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        run_name = f"T1-{job['model']}-{stamp}"
        log = root / "logs" / f"{run_name}.log"
        profile = root / "runs" / run_name / "hardware_profile.json"
        state = {
            "model": job["model"],
            "status": "running",
            "fingerprint": signature,
            "gpu_index": args.gpu,
            "gpu_uuid": uuid,
            "image": job["image"],
            "log": str(log),
            "profile": str(profile),
        }
        atomic_json(state_path, state)
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "RPX_PROFILE_ROOT": str(root / "runs"),
                "RPX_PROFILE_RUN_NAME": run_name,
                "RPX_PROFILE_SAMPLES": str(samples),
                "RPX_PROFILE_SPLIT": plan["split"],
                "RPX_PROFILE_WARMUP_CALLS": str(plan["warmup_calls"]),
                "RPX_PROFILE_FLOP_CALLS": str(plan["flop_calls"]),
                "RPX_PROFILE_DATASET_REPO": str(dataset_repo),
            }
        )
        command = [
            "bash",
            str(repo / "docker/run_vision_profile.sh"),
            "image-depth",
            job["model"],
            str(args.gpu),
            job["image"],
            job["python"],
            "--repo",
            plan["dataset_repo"],
            "--revision",
            plan["dataset_revision"],
            "--device",
            "cuda",
        ]
        print(f"START {job['model']} on GPU {args.gpu} | {log}", flush=True)
        try:
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"profile exited {result.returncode}")
            report = read_json(profile)
            if report.get("status") != "complete" or report.get("measured_units", 0) < samples:
                raise RuntimeError("profile did not satisfy the measured sample budget")
            state.update(status="complete", measured_units=report["measured_units"])
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures += 1
        atomic_json(state_path, state)
        print(f"{state['status'].upper()} {job['model']}", flush=True)
        aggregate(plan, root)
        if state["status"] == "failed" and not args.keep_going:
            break

    rows = aggregate(plan, root)
    counts = show_status(plan, root)
    lock.close()
    if failures or len(rows) != len(plan["jobs"]):
        raise SystemExit("T1 suite is incomplete; completed profiles were preserved for resume")
    print(f"T1 COMPLETE: {len(rows)} models × {samples} measured frames")


if __name__ == "__main__":
    main()
