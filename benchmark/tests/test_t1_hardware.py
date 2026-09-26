from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).parents[1]
REPO = BENCHMARK.parent


def load_runner():
    path = BENCHMARK / "scripts/run_t1_hardware.py"
    spec = importlib.util.spec_from_file_location("run_t1_hardware", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_t1_plan_is_complete_pinned_and_uses_thousand_frames():
    runner = load_runner()
    plan = runner.load_plan(BENCHMARK / "configs/hardware/t1-image-depth.json")
    assert plan["samples"] == 1000
    assert plan["warmup_calls"] == 1
    assert len(plan["jobs"]) == 10
    assert len({job["image"] for job in plan["jobs"]}) == 3
    assert all("@sha256:" in job["image"] for job in plan["jobs"])
    assert "depthlm" not in {job["model"] for job in plan["jobs"]}


def test_dataset_repository_accepts_both_hugging_face_cache_layouts(tmp_path):
    runner = load_runner()
    revision = "revision"
    direct = tmp_path / "datasets--IRVLUTD--RPX"
    manifest = direct / "snapshots" / revision / "manifests/monocular_depth/easy.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    assert runner.dataset_repository(tmp_path, revision) == direct

    manifest.unlink()
    standard = tmp_path / "hub/datasets--IRVLUTD--RPX"
    manifest = standard / "snapshots" / revision / "manifests/monocular_depth/easy.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    assert runner.dataset_repository(tmp_path, revision) == standard


def test_completed_requires_matching_fingerprint_and_budget(tmp_path):
    runner = load_runner()
    report = tmp_path / "hardware_profile.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "rpx.hardware.v1",
                "status": "complete",
                "measured_units": 1000,
            }
        )
    )
    state = {"status": "complete", "fingerprint": "same", "profile": str(report)}
    assert runner.completed(state, "same", 1000)
    assert not runner.completed(state, "different", 1000)
    assert not runner.completed(state, "same", 1001)


def test_busy_selected_gpu_stops_without_terminating_processes(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: "GPU-selected, 42, python\nGPU-other, 9, python\n",
    )
    with pytest.raises(RuntimeError, match="nothing was killed"):
        runner.assert_idle("GPU-selected")
    runner.assert_idle("GPU-free")


def test_aggregate_exports_only_completed_profiles(tmp_path):
    runner = load_runner()
    plan = {
        "jobs": [
            {"model": "done", "image": "repo@sha256:1"},
            {"model": "pending", "image": "repo@sha256:2"},
        ]
    }
    (tmp_path / "jobs").mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "status": "complete",
                "measured_units": 1000,
                "latency_mean_ms": 5.0,
                "latency_p50_ms": 4.9,
                "latency_p95_ms": 5.5,
                "latency_p99_ms": 6.0,
                "throughput_units_per_s": 200.0,
                "peak_cuda_allocated_mib": 100.0,
                "peak_cuda_reserved_mib": 120.0,
                "params_m": 10.0,
                "flops_g_per_unit": 20.0,
                "system": {"gpu_name": "test gpu"},
            }
        )
    )
    (tmp_path / "jobs/done.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "profile": str(profile),
            }
        )
    )
    rows = runner.aggregate(plan, tmp_path)
    assert [row["model"] for row in rows] == ["done"]
    assert rows[0]["latency_p95_ms"] == 5.5
    assert (tmp_path / "T1_hardware_metrics.csv").is_file()


def test_container_wrapper_mounts_profiler_and_uses_selected_gpu(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "$1" = image ]; then echo sha256:fake; else printf "%s\\n" "$@"; fi\n'
    )
    docker.chmod(0o755)
    cache = tmp_path / "cache"
    (cache / "hub/datasets--IRVLUTD--RPX").mkdir(parents=True)
    environment = {
        **os.environ,
        "PATH": str(binaries) + ":" + os.environ["PATH"],
        "RPX_HF_CACHE": str(cache),
        "RPX_PROFILE_ROOT": str(tmp_path / "output"),
        "RPX_PROFILE_RUN_NAME": "test-run",
    }
    result = subprocess.run(
        [
            "bash",
            str(REPO / "docker/run_vision_profile.sh"),
            "image-depth",
            "depth-pro",
            "2",
            "fake",
            "/opt/python",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "device=2" in result.stdout
    assert "RPX_PROFILE_SAMPLES=1000" not in result.stdout
    assert "scripts/profile_vision.py" in result.stdout
    assert "rpx_benchmark/hardware_profile.py" in result.stdout
