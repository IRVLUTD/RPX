"""M3 tests: runner hook dispatch + profiler backends.

Covers:

- Depth / segmentation TaskSpecs carry the deployment-readiness hooks
  the runner now dispatches through.
- The runner calls a registered temporal-stability hook and skips it
  for tasks without one.
- `LatencyProfiler` percentiles behave correctly on trivial inputs.
- `MemoryProfiler` degrades gracefully: every backend returns either
  a float or ``None`` without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rpx_benchmark.api import (
    TaskType,
)
from rpx_benchmark.profiler import LatencyProfiler, MemoryProfiler
from rpx_benchmark.tasks.registry import get_task_spec

# --------------------------------------------------------------------------- #
# Hook registration invariants
# --------------------------------------------------------------------------- #


def test_depth_taskspec_has_ts_hook() -> None:
    spec = get_task_spec(TaskType.MONOCULAR_DEPTH)
    assert callable(spec.temporal_stability_fn)
    assert spec.geometric_coherence_fn is None


def test_segmentation_taskspec_has_both_hooks() -> None:
    spec = get_task_spec(TaskType.OBJECT_SEGMENTATION)
    assert callable(spec.temporal_stability_fn)
    assert callable(spec.geometric_coherence_fn)


def test_tracking_taskspec_has_no_deployment_hooks() -> None:
    """Tracking hasn't landed temporal hooks yet; the runner should skip them."""
    spec = get_task_spec(TaskType.OBJECT_TRACKING)
    assert spec.temporal_stability_fn is None
    assert spec.geometric_coherence_fn is None


# --------------------------------------------------------------------------- #
# Runner → hook dispatch
# --------------------------------------------------------------------------- #


def _write_depth_dataset(root: Path, n: int = 3) -> Path:
    """Tiny monocular-depth manifest backed by flat PNGs."""
    root.mkdir(parents=True, exist_ok=True)
    rgb_dir = root / "scenes" / "scene_000" / "0" / "rgb"
    depth_dir = root / "scenes" / "scene_000" / "0" / "depth"
    pose_dir = root / "scenes" / "scene_000" / "0" / "pose"
    for d in (rgb_dir, depth_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        Image.fromarray(np.full((16, 16, 3), 128, np.uint8)).save(rgb_dir / f"{i}.png")
        Image.fromarray(np.full((16, 16), 2000, np.uint16)).save(depth_dir / f"{i}.png")
        np.savez(
            pose_dir / f"{i}.npz",
            position=np.array([float(i), 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        samples.append(
            {
                "id": f"scene_000_clutter_{i}",
                "rgb": f"scenes/scene_000/0/rgb/{i}.png",
                "depth": f"scenes/scene_000/0/depth/{i}.png",
                "pose": f"scenes/scene_000/0/pose/{i}.npz",
                "phase": "clutter",
                "difficulty": "easy",
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "monocular_depth",
                "root": str(root),
                "samples": samples,
            }
        )
    )
    return manifest


def test_runner_dispatches_temporal_hook_via_taskspec(tmp_path: Path, monkeypatch) -> None:
    """The runner must invoke ``spec.temporal_stability_fn`` — not hardcoded branches."""
    from rpx_benchmark.loader import RPXDataset
    from rpx_benchmark.runner import BenchmarkRunner

    calls = {"n": 0}
    spec = get_task_spec(TaskType.MONOCULAR_DEPTH)
    original_hook = spec.temporal_stability_fn

    def counting_hook(preds, samples, poses):
        calls["n"] += 1
        return original_hook(preds, samples, poses)

    monkeypatch.setattr(spec, "temporal_stability_fn", counting_hook)

    manifest = _write_depth_dataset(tmp_path)
    ds = RPXDataset.from_manifest(manifest, batch_size=1)

    def trivial(rgb):
        return np.full(rgb.shape[:2], 2.0, dtype=np.float32)

    import rpx_benchmark as rpx

    model = rpx.make_numpy_depth_model(trivial)
    runner = BenchmarkRunner(model=model, dataset=ds)
    _, report = runner.run_with_report(
        primary_metric="absrel",
        model_name="unit",
    )
    assert calls["n"] == 1, "runner did not dispatch through the TaskSpec hook"
    # Runner now populates latency percentiles into the report directly
    # via the injected LatencyProfiler; the median field is a float.
    assert isinstance(report.latency_ms_per_sample, float)


# --------------------------------------------------------------------------- #
# Profiler backends
# --------------------------------------------------------------------------- #


def test_latency_percentiles_trivial() -> None:
    lp = LatencyProfiler(warmup=0)
    for s in (0.010, 0.020, 0.030, 0.040, 0.050):
        lp.add_sample_seconds(s)
    pcs = lp.percentiles()
    assert pcs["p50_ms"] == pytest.approx(30.0)
    assert pcs["mean_ms"] == pytest.approx(30.0)
    # p95 on [10,20,30,40,50] ≈ 48
    assert pcs["p95_ms"] > pcs["p50_ms"]


def test_latency_percentiles_trim_warmup() -> None:
    lp = LatencyProfiler(warmup=2)
    for s in (1.0, 1.0, 0.010, 0.020):  # two warmup outliers trimmed
        lp.add_sample_seconds(s)
    pcs = lp.percentiles()
    assert pcs["p50_ms"] == pytest.approx(15.0)  # mean of 10 and 20 ms


def test_latency_percentiles_empty_returns_none() -> None:
    lp = LatencyProfiler()
    pcs = lp.percentiles()
    assert all(v is None for v in pcs.values())


def test_memory_profiler_peaks_degrade_gracefully() -> None:
    mp = MemoryProfiler().reset()
    peaks = mp.peaks()
    assert set(peaks.keys()) == {"cpu_mb", "cuda_mb", "mps_mb"}
    for v in peaks.values():
        assert v is None or isinstance(v, float)


def test_memory_profiler_cpu_sample_is_nonzero() -> None:
    """If psutil or resource is available, CPU RSS should produce a float."""
    mp = MemoryProfiler(sample_cuda=False, sample_mps=False).reset()
    peaks = mp.peaks()
    # Tests run inside the same process that imported numpy + pydantic;
    # at least a few MB must be resident.
    if peaks["cpu_mb"] is not None:
        assert peaks["cpu_mb"] > 0.0


def test_efficiency_metadata_new_fields_round_trip() -> None:
    from rpx_benchmark.profiler import EfficiencyMetadata

    em = EfficiencyMetadata(
        params_m=10.0,
        flops_g=5.0,
        latency_p50_ms=12.3,
        latency_p95_ms=45.6,
        peak_cuda_mb=1024.0,
    )
    row = em.to_table_row()
    assert row["latency_p50_ms"] == 12.3
    assert row["peak_cuda_mb"] == 1024.0
    assert row["peak_mps_mb"] is None
