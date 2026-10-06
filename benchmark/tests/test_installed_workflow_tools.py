"""User workflows: six evaluators, raw-data analysis and honest timing reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.examples.benchmark_tasks import create_smoke, run_task
from rpx_benchmark.examples.profile_callable import measure
from rpx_benchmark.examples.summarize_metrics import calculate, read_records
from rpx_benchmark.exceptions import ConfigError, MetricError


@pytest.mark.parametrize("task", ["T1", "T2", "T3", "T4", "T5", "T6"])
def test_six_installed_task_examples_score_and_write_reports(task, tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("An offline integration fixture attempted a network download")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_download)
    manifest, fn = create_smoke(tmp_path / "inputs", task)
    calls = []

    def predict(*args):
        calls.append(args)
        return fn(*args)

    out = tmp_path / "results"
    report = run_task(
        task,
        manifest,
        predict,
        out,
        split="easy",
        image_cache=tmp_path / "inputs/images",
        model_name="synthetic",
        model_revision="demo-v1",
    )
    assert report["num_samples"] == 1
    assert len(calls) == 1
    if task in {"T5", "T6"}:
        assert report["metrics"]["bbox_accuracy_at_0_5"] == 1.0
        assert len(calls[0][0]) == (2 if task == "T6" else 1)
        assert json.loads((out / "result.json").read_text()) == report
        assert len((out / "predictions.jsonl").read_text().splitlines()) == 1
    else:
        assert Path(report["paths"]["json"]).is_file()
        assert Path(report["paths"]["cells"]).is_file()
        if task == "T1":
            assert report["aggregated"]["absrel"] == pytest.approx(0.05)
        elif task == "T3":
            assert report["aggregated"]["mota"] == 1.0
        elif task == "T4":
            assert report["aggregated"]["rotation_error_deg"] == 0.0
            assert report["aggregated"]["translation_angular_deg"] == 0.0


def test_t6_refuses_one_image_manifest(tmp_path):
    manifest, fn = create_smoke(tmp_path / "inputs", "T5")
    with pytest.raises(ConfigError, match="wrong one-image/two-image"):
        run_task("T6", manifest, fn, tmp_path / "out")


def raw_rows():
    rng = np.random.default_rng(19)
    return [
        {
            "model": model,
            "task": "T1",
            "scene_id": f"scene{s:03d}",
            "phase": phase,
            "absrel": float(rng.uniform(0.02, 0.15) + offset),
            "delta1": float(rng.uniform(0.7, 0.95) - offset),
        }
        for model, offset in [("a", 0.0), ("b", 0.2)]
        for s in range(12)
        for phase in range(3)
    ]


def test_raw_summaries_keep_models_separate_and_report_missing_scenes():
    rows = raw_rows()
    rows = [
        r
        for r in rows
        if not (r["model"] == "b" and r["scene_id"] == "scene000" and r["phase"] == 2)
    ]
    result = calculate(rows, ["absrel", "delta1"])
    a, b = result["results"]
    assert a["model"] == "a" and b["model"] == "b"
    assert (a["n_eff"], b["n_eff"], b["n_dropped"]) == (12, 11, 1)
    assert a["phi"] is not None and b["phi"] is not None
    assert a["jedi"]["jedi"] > b["jedi"]["jedi"]
    for summary in (a, b):
        assert summary["j_min"] == min(summary["jedi"]["per_phase"].values())
    assert result["metric_specs"]["absrel"]["direction"] == "lower"


@pytest.mark.parametrize("value", [None, True, "NaN", "Infinity", "missing"])
def test_raw_summary_rejects_missing_nonfinite_or_boolean_metrics(value):
    rows = raw_rows()
    rows[0]["absrel"] = value
    with pytest.raises(MetricError, match="absrel"):
        calculate(rows, ["absrel", "delta1"])


def test_raw_summary_requires_the_full_phase_protocol():
    rows = [row for row in raw_rows() if row["phase"] != 2]
    with pytest.raises(MetricError, match="all three phases"):
        calculate(rows, ["absrel"])


@pytest.mark.parametrize("suffix", ["json", "jsonl", "csv"])
def test_raw_records_formats_produce_equal_analysis(suffix, tmp_path):
    import csv

    rows = raw_rows()
    path = tmp_path / f"raw.{suffix}"
    if suffix == "csv":
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif suffix == "jsonl":
        path.write_text("\n".join(json.dumps(r) for r in rows))
    else:
        path.write_text(json.dumps({"per_sample": rows}))
    assert calculate(read_records(path), ["absrel"]) == calculate(rows, ["absrel"])


def test_profiler_excludes_warmup_and_records_scope():
    calls = []

    def predict(rgb):
        calls.append(rgb.shape)
        return rgb.mean(axis=2)

    result = measure(
        predict,
        {"rgb": np.zeros((8, 10, 3), np.uint8)},
        warmup=2,
        repeats=5,
        precision="fp32",
        unit="frame",
    )
    assert len(calls) == 7
    assert len(result["latency_samples_ms"]) == 5
    assert result["measured_calls"] == 5 and result["warmup_calls"] == 2
    assert result["latency"]["p99_ms"] >= result["latency"]["p50_ms"] >= 0.0
    assert result["system_card"]["gpu_count"] == 0
    assert result["memory"]["cuda_peak_allocated_mb_by_visible_device"] == {}
    assert result["inputs"]["rgb"] == {"shape": [8, 10, 3], "dtype": "uint8"}
    json.dumps(result, allow_nan=False)


def test_cuda_profiler_refuses_unavailable_gpu(monkeypatch):
    from types import SimpleNamespace

    import rpx_benchmark.examples.profile_callable as module

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(ConfigError, match="no CUDA GPU"):
        measure(lambda **inputs: None, {"rgb": np.zeros((2, 2, 3))}, device="cuda", repeats=1)


def test_cuda_profiler_synchronizes_every_visible_gpu_on_each_call(monkeypatch):
    from types import SimpleNamespace

    import rpx_benchmark.examples.profile_callable as module

    events = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        synchronize=lambda index: events.append(f"sync-{index}"),
        reset_peak_memory_stats=lambda index: events.append(f"reset-{index}"),
        max_memory_allocated=lambda index: (index + 1) * 1024**2,
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        module.SystemCard, "auto_detect", lambda **kwargs: module.SystemCard(gpu_count=2)
    )
    result = module.measure(
        lambda **inputs: events.append("predict"),
        {"rgb": np.zeros((2, 2, 3))},
        device="cuda",
        warmup=1,
        repeats=2,
    )
    call = ["sync-0", "sync-1", "predict", "sync-0", "sync-1"]
    assert events == call + ["reset-0", "reset-1"] + call + call
    assert result["memory"]["cuda_peak_allocated_mb_by_visible_device"] == {"0": 1.0, "1": 2.0}
    assert len(result["latency_samples_ms"]) == 2


def test_documented_analysis_and_latency_examples():
    import re

    import rpx_benchmark

    root = Path(rpx_benchmark.__file__).parent / "guides"
    for guide in ["analysis", "profiling"]:
        source = (root / guide / "README.md").read_text()
        code = re.findall(r"```python\n(.*?)```", source, re.DOTALL)[0]
        exec(compile(code, f"{guide}/README.md", "exec"), {})
