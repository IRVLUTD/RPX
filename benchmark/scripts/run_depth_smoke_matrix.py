"""Run the canonical RPX depth smoke matrix sequentially and resumably.

This is the roster-level companion to ``run_depth_smoke_gate.py``. It never
runs two models at once, never bypasses the three unverified-model safety
rails, and stores state under a code-identity-specific directory so results
from changed code cannot silently satisfy a later gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_depth_smoke_gate as gate
import setup_depth_smoke_env as setup

SCHEMA_VERSION = 1
VALID_GATES = ("micro", "acceptance", "easy")
TERMINAL_FAILURES = {"access", "cuda_oom", "weight_download"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_gates(value: str) -> list[str]:
    gates = _parse_csv(value)
    if not gates:
        raise argparse.ArgumentTypeError("at least one gate is required")
    unknown = [item for item in gates if item not in VALID_GATES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown gate(s) {unknown}; choose from {list(VALID_GATES)}"
        )
    if len(set(gates)) != len(gates):
        raise argparse.ArgumentTypeError("duplicate gates are not allowed")
    if "easy" in gates and len(gates) != 1:
        raise argparse.ArgumentTypeError(
            "the full Easy gate must run alone after acceptance passes"
        )
    if "acceptance" in gates and "micro" in gates:
        gates = ["micro", "acceptance"]
    return gates


def _canonical_models(task: str) -> list[tuple[str, str]]:
    image = [(name, "image") for name in sorted(gate.IMAGE_MODELS)]
    video = [(name, "video") for name in sorted(gate.VIDEO_MODELS)]
    if task == "image":
        return image
    if task == "video":
        return video
    return image + video


def _select_models(task: str, requested: list[str] | None) -> list[tuple[str, str]]:
    available = _canonical_models(task)
    if not requested:
        return available
    by_name = {name: model_task for name, model_task in available}
    all_names = gate.IMAGE_MODELS | gate.VIDEO_MODELS
    unknown = [name for name in requested if name not in all_names]
    if unknown:
        raise ValueError(f"unknown canonical model(s): {unknown}")
    wrong_task = [name for name in requested if name not in by_name]
    if wrong_task:
        raise ValueError(f"model(s) outside --task {task}: {wrong_task}")
    return [(name, by_name[name]) for name in requested]


def _print_roster() -> None:
    print("MODEL\tTASK\tFAMILY\tSTATUS")
    for name, task in _canonical_models("all"):
        family = setup.MODEL_FAMILY.get(name, "-")
        if name in gate.BLOCKED_MODELS:
            status = "blocked-unverified"
        elif name in gate.DEDICATED_RUNTIME_MODELS:
            status = "dedicated-runtime"
        else:
            status = "runnable"
        print(f"{name}\t{task}\t{family}\t{status}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_state(
    path: Path,
    *,
    code_identity: str,
    git_dirty: bool,
    selected: list[tuple[str, str]],
    gates: list[str],
) -> dict[str, Any]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("code_identity") != code_identity:
            raise RuntimeError("matrix state belongs to a different code identity")
        return payload
    now = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "code_identity": code_identity,
        "git_dirty": git_dirty,
        "host": socket.gethostname(),
        "dataset_repo": gate.DATASET_REPO,
        "dataset_revision": gate.DATASET_REVISION,
        "created_utc": now,
        "updated_utc": now,
        "selected_models": [name for name, _task in selected],
        "requested_gates": gates,
        "setups": {},
        "models": {},
    }


def _gate_state(state: dict[str, Any], model: str, task: str, gate_name: str) -> dict:
    model_state = state["models"].setdefault(
        model,
        {
            "task": task,
            "family": setup.MODEL_FAMILY.get(model),
            "blocked": model in gate.BLOCKED_MODELS,
            "gates": {},
        },
    )
    return model_state["gates"].setdefault(
        gate_name,
        {"status": "pending", "attempts": 0},
    )


def _latest_metadata(
    output_root: Path,
    code_identity: str,
    host: str,
    model: str,
    gate_name: str,
) -> Path | None:
    parent = output_root / code_identity / host / model / gate_name
    candidates = sorted(parent.glob("*/run_metadata.json"))
    return candidates[-1] if candidates else None


def _environment_ready(env_root: Path, model: str) -> bool:
    family = setup.MODEL_FAMILY[model]
    env_dir = env_root / family
    return (env_dir / "bin" / "python").is_file() and (env_dir / "rpx-environment.json").is_file()


def _check_storage(env_root: Path, minimum_free_gb: float) -> None:
    probe = env_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1024**3
    if free_gb < minimum_free_gb:
        raise SystemExit(
            f"environment storage has only {free_gb:.1f} GiB free; "
            f"--setup-missing requires at least {minimum_free_gb:.1f} GiB"
        )
    if free_gb < 150:
        print(
            f"WARNING: {free_gb:.1f} GiB free. One local family is safe, but the "
            "complete 17-model server cache should have at least 150 GiB.",
            flush=True,
        )


def _run_setup(
    *,
    repo_root: Path,
    model: str,
    env_root: Path,
    setup_python: str,
    log_path: Path,
) -> int:
    command = [
        setup_python,
        str(repo_root / "benchmark/scripts/setup_depth_smoke_env.py"),
        "--model",
        model,
        "--env-root",
        str(env_root),
        "--python",
        setup_python,
    ]
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        print(f"  setup failed; log: {log_path}", flush=True)
    return return_code


def _run_gate(
    *,
    repo_root: Path,
    env_root: Path,
    output_root: Path,
    cache_dir: Path | None,
    gpu_index: int,
    model: str,
    task: str,
    gate_name: str,
) -> int:
    family = setup.MODEL_FAMILY[model]
    python = env_root / family / "bin" / "python"
    command = [
        str(python),
        str(repo_root / "benchmark/scripts/run_depth_smoke_gate.py"),
        "--task",
        task,
        "--model",
        model,
        "--gate",
        gate_name,
        "--output-root",
        str(output_root),
        "--gpu-index",
        str(gpu_index),
    ]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]
    source = env_root / "sources" / family
    if family in setup.UPSTREAMS and source.is_dir():
        command += ["--upstream-dir", str(source)]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=repo_root, check=False).returncode


def _record_metadata(gate_state: dict, metadata_path: Path | None) -> None:
    if metadata_path is None:
        gate_state.update(
            {
                "status": "failed",
                "failure_class": "code_or_unknown",
                "error": "gate created no run_metadata.json",
            }
        )
        return
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    gate_state.update(
        {
            "status": payload.get("status", "failed"),
            "failure_class": payload.get("failure_class"),
            "error": payload.get("error"),
            "metadata_path": str(metadata_path),
            "validation": payload.get("validation"),
            "finished_utc": payload.get("finished_utc"),
        }
    )


def _prerequisite(gate_name: str) -> str | None:
    if gate_name == "acceptance":
        return "micro"
    if gate_name == "easy":
        return "acceptance"
    return None


def _skip_reason(
    item: dict[str, Any],
    *,
    rerun_passed: bool,
    retry_failed: bool,
) -> str | None:
    status = item.get("status")
    attempts = int(item.get("attempts", 0))
    failure = item.get("failure_class")
    if status == "passed" and not rerun_passed:
        return "already passed; resume skip"
    if failure in TERMINAL_FAILURES and attempts >= 1:
        return f"terminal {failure}; one-attempt ceiling reached"
    if attempts >= 2 and status != "passed":
        return "two-attempt code/dependency ceiling reached"
    if status in {"failed", "interrupted"} and not retry_failed:
        return f"prior {status}/{failure}; use --retry-failed after a fix"
    return None


def _setup_skip_reason(item: dict[str, Any], *, retry_failed: bool) -> str | None:
    """Enforce the same bounded-resume policy for heavyweight setup work."""
    status = item.get("status")
    attempts = int(item.get("attempts", 0))
    if attempts >= 2:
        return "two-attempt environment setup ceiling reached"
    if attempts >= 1 and status in {"failed", "interrupted", "passed"} and not retry_failed:
        return f"prior setup {status}; use --retry-failed after correcting the recipe"
    return None


def main() -> None:  # noqa: C901 - the CLI is intentionally linear and auditable
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", choices=["image", "video", "all"], default="all")
    parser.add_argument(
        "--models",
        help="comma-separated canonical names; default is the selected task roster",
    )
    parser.add_argument(
        "--gates",
        type=_parse_gates,
        default=_parse_gates("micro,acceptance"),
        help="micro,acceptance (default), one gate, or easy by itself",
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path(os.environ.get("RPX_ENV_ROOT", "./rpx_depth_envs")),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=gate._default_output_root())
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--setup-python", default="python3.11")
    parser.add_argument("--setup-missing", action="store_true")
    parser.add_argument("--rerun-passed", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        _print_roster()
        return
    try:
        selected = _select_models(
            args.task,
            _parse_csv(args.models) if args.models else None,
        )
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = Path(__file__).resolve().parents[2]
    env_root = args.env_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    code_identity, git_dirty = gate._code_identity(repo_root)
    host = socket.gethostname()
    state_path = output_root / "matrix" / code_identity / host / "matrix.json"
    state = _load_state(
        state_path,
        code_identity=code_identity,
        git_dirty=git_dirty,
        selected=selected,
        gates=args.gates,
    )
    state["selected_models"] = [name for name, _task in selected]
    state["requested_gates"] = args.gates
    state.setdefault("setups", {})
    state["updated_utc"] = _utc_now()
    _atomic_write_json(state_path, state)

    if args.setup_missing:
        _check_storage(env_root, args.min_free_gb)
        env_root.mkdir(parents=True, exist_ok=True)

    attempted_setup_families: set[str] = set()
    unexpected = False
    for model, task in selected:
        print(f"\n=== {model} ({task}) ===", flush=True)
        if model in gate.BLOCKED_MODELS:
            for gate_name in args.gates:
                item = _gate_state(state, model, task, gate_name)
                item.update(
                    {
                        "status": "blocked",
                        "failure_class": "unverified_weights",
                        "updated_utc": _utc_now(),
                    }
                )
            state["updated_utc"] = _utc_now()
            _atomic_write_json(state_path, state)
            print("  blocked: official weights are not verified", flush=True)
            continue
        if model in gate.DEDICATED_RUNTIME_MODELS:
            for gate_name in args.gates:
                item = _gate_state(state, model, task, gate_name)
                item.update(
                    {
                        "status": "blocked",
                        "failure_class": "dedicated_runtime_required",
                        "updated_utc": _utc_now(),
                    }
                )
            state["updated_utc"] = _utc_now()
            _atomic_write_json(state_path, state)
            print(
                "  use the model's dedicated Docker overlay for this smoke gate",
                flush=True,
            )
            continue

        family = setup.MODEL_FAMILY[model]
        if not _environment_ready(env_root, model):
            setup_state = state["setups"].setdefault(
                family,
                {"status": "pending", "attempts": 0},
            )
            if args.setup_missing and family not in attempted_setup_families:
                setup_skip = _setup_skip_reason(
                    setup_state,
                    retry_failed=args.retry_failed,
                )
                if setup_skip:
                    print(f"  setup: {setup_skip}", flush=True)
                else:
                    attempted_setup_families.add(family)
                    setup_log = state_path.parent / "setup" / f"{family}.log"
                    setup_state["attempts"] = int(setup_state.get("attempts", 0)) + 1
                    setup_state["status"] = "running"
                    setup_state["started_utc"] = _utc_now()
                    setup_state["updated_utc"] = setup_state["started_utc"]
                    setup_state["log"] = str(setup_log)
                    state["updated_utc"] = _utc_now()
                    _atomic_write_json(state_path, state)
                    return_code = _run_setup(
                        repo_root=repo_root,
                        model=model,
                        env_root=env_root,
                        setup_python=args.setup_python,
                        log_path=setup_log,
                    )
                    ready = return_code == 0 and _environment_ready(env_root, model)
                    setup_state["status"] = "passed" if ready else "failed"
                    setup_state["failure_class"] = None if ready else "dependency"
                    setup_state["return_code"] = return_code
                    setup_state["finished_utc"] = _utc_now()
                    setup_state["updated_utc"] = setup_state["finished_utc"]
                    state["updated_utc"] = _utc_now()
                    _atomic_write_json(state_path, state)
                    if ready:
                        print(f"  environment ready: {env_root / family}", flush=True)
            if not _environment_ready(env_root, model):
                for gate_name in args.gates:
                    item = _gate_state(state, model, task, gate_name)
                    item.update(
                        {
                            "status": "missing_environment",
                            "failure_class": "dependency",
                            "setup_log": str(state_path.parent / "setup" / f"{family}.log"),
                            "setup_attempts": int(setup_state.get("attempts", 0)),
                            "updated_utc": _utc_now(),
                        }
                    )
                state["updated_utc"] = _utc_now()
                _atomic_write_json(state_path, state)
                unexpected = True
                print("  skipped: environment is not ready", flush=True)
                continue

        for gate_name in args.gates:
            item = _gate_state(state, model, task, gate_name)
            skip_reason = _skip_reason(
                item,
                rerun_passed=args.rerun_passed,
                retry_failed=args.retry_failed,
            )
            if skip_reason:
                print(f"  {gate_name}: {skip_reason}", flush=True)
                if item.get("status") != "passed":
                    unexpected = True
                continue

            required = _prerequisite(gate_name)
            if required:
                required_state = _gate_state(state, model, task, required)
                if required_state.get("status") != "passed":
                    item.update(
                        {
                            "status": "prerequisite_failed",
                            "failure_class": "gate_order",
                            "requires": required,
                            "updated_utc": _utc_now(),
                        }
                    )
                    state["updated_utc"] = _utc_now()
                    _atomic_write_json(state_path, state)
                    unexpected = True
                    print(f"  {gate_name}: skipped until {required} passes", flush=True)
                    continue

            item["attempts"] = int(item.get("attempts", 0)) + 1
            item["status"] = "running"
            item["started_utc"] = _utc_now()
            item["updated_utc"] = item["started_utc"]
            state["updated_utc"] = _utc_now()
            _atomic_write_json(state_path, state)
            return_code = _run_gate(
                repo_root=repo_root,
                env_root=env_root,
                output_root=output_root,
                cache_dir=cache_dir,
                gpu_index=args.gpu_index,
                model=model,
                task=task,
                gate_name=gate_name,
            )
            metadata = _latest_metadata(
                output_root,
                code_identity,
                host,
                model,
                gate_name,
            )
            _record_metadata(item, metadata)
            item["return_code"] = return_code
            item["updated_utc"] = _utc_now()
            state["updated_utc"] = _utc_now()
            _atomic_write_json(state_path, state)
            print(
                f"  {gate_name}: {item['status']}"
                + (f" ({item.get('failure_class')})" if item.get("failure_class") else ""),
                flush=True,
            )
            if item["status"] != "passed":
                unexpected = True

    print(f"\nMatrix state: {state_path}")
    if unexpected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
