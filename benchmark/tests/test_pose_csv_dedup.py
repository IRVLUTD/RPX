"""Dedup contract for ``BatchedRelativePoseBenchmarkModel.maybe_save``.

The benchmark runner's deployment-readiness path dispatches each pair
twice (once for warm-up timing, once for the measured pass). A naive
append-on-every-call CSV produces duplicate rows for the same
``(scene, phase, frame_a, frame_b)`` key. This test pins the in-process
dedup behaviour so that regression doesn't sneak back in.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rpx_benchmark.adapters import BatchedRelativePoseBenchmarkModel


def _fake_sample(scene: str, phase: int, frame_a: str, frame_b: str):
    return SimpleNamespace(
        id=f"{scene}__{phase}__{frame_a}__{frame_b}",
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        metadata={
            "scene_id": scene,
            "phase_idx": phase,
            "frame": frame_a,
            "frame_b": frame_b,
        },
    )


def _read(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", newline="") as f:
        return list(csv.reader(f))


def test_maybe_save_writes_one_row_per_unique_pair(tmp_path: Path):
    """Two pairs, each saved twice (mimics warmup + measure passes) →
    exactly two data rows in the CSV (plus the header)."""

    def adapter(_pairs):  # noqa: ARG001
        return [{"rotation": np.eye(3), "translation": np.zeros(3)}]

    model = BatchedRelativePoseBenchmarkModel(
        adapter,
        name="dedup",
        save_dir=tmp_path,
    )

    out = {"rotation": np.eye(3), "translation": np.array([0.1, 0.2, 0.3])}
    pair_a = _fake_sample("scene_x", 0, "00000", "00005")
    pair_b = _fake_sample("scene_x", 0, "00001", "00006")

    # Call order mirrors the runner: each pair is dispatched twice.
    model.maybe_save(pair_a, out)
    model.maybe_save(pair_b, out)
    model.maybe_save(pair_a, out)
    model.maybe_save(pair_b, out)

    rows = _read(tmp_path / "predictions.csv")
    # 1 header + 2 unique data rows.
    assert len(rows) == 3, f"expected header + 2 rows, got {len(rows)}: {rows}"
    assert rows[0][:4] == ["scene_id", "phase", "frame_a", "frame_b"]
    assert rows[1][:4] == ["scene_x", "0", "00000", "00005"]
    assert rows[2][:4] == ["scene_x", "0", "00001", "00006"]


def test_maybe_save_dedup_is_per_instance(tmp_path: Path):
    """A *new* model instance does not inherit the seen set — important
    for resuming runs: a re-run after restart must be free to re-write
    the file (the existing on-disk header is detected separately)."""
    model_1 = BatchedRelativePoseBenchmarkModel(
        lambda _: [{"rotation": np.eye(3), "translation": np.zeros(3)}],
        name="m1",
        save_dir=tmp_path,
    )
    out = {"rotation": np.eye(3), "translation": np.zeros(3)}
    model_1.maybe_save(_fake_sample("s", 0, "00000", "00005"), out)

    rows_after_first = _read(tmp_path / "predictions.csv")
    assert len(rows_after_first) == 2  # header + 1

    # Different instance → fresh seen-set → row appends again (the in-process
    # dedup intentionally does NOT block cross-process resumes).
    model_2 = BatchedRelativePoseBenchmarkModel(
        lambda _: [{"rotation": np.eye(3), "translation": np.zeros(3)}],
        name="m2",
        save_dir=tmp_path,
    )
    model_2.maybe_save(_fake_sample("s", 0, "00000", "00005"), out)

    rows_after_second = _read(tmp_path / "predictions.csv")
    assert len(rows_after_second) == 3  # header + 2
