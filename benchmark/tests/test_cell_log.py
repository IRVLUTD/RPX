"""Tests for the per-(scene, phase) cell-log artifact.

Locks the contracts of :mod:`rpx_benchmark.cell_log`:

- per-sample rows group into one cell per (scene, phase),
- metric values are averaged across frames in the cell,
- non-numeric metadata (id, difficulty) is carried through,
- write/read round-trips for both Parquet and JSONL,
- missing metric values become None (round-tripped as null),
- ``metric:`` prefix namespacing is preserved end-to-end.
"""

from __future__ import annotations

import pytest

from rpx_benchmark.cell_log import (
    FIXED_COLUMNS,
    METRIC_PREFIX,
    cells_from_per_sample,
    merge_cells,
    metric_col,
    metric_keys,
    read_cells,
    write_cells,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _per_sample(n_scenes: int = 3, frames_per_phase: int = 4):
    rows = []
    for s in range(n_scenes):
        for p in ["clu", "int", "cln"]:
            for f in range(frames_per_phase):
                rows.append(
                    {
                        "id": f"{s}_{p}_{f}",
                        "scene": f"scene_{s:03d}",
                        "phase": p,
                        "difficulty": "easy",
                        "absrel": 0.10 + (0.05 if p == "int" else 0.0),
                        "rmse": 0.50,
                        "delta1": 0.90,
                        "latency_ms": 42.0,
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


class TestCellsFromPerSample:
    def test_one_row_per_scene_phase(self):
        rows = _per_sample(n_scenes=5, frames_per_phase=10)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        assert len(cells) == 5 * 3

    def test_metric_namespacing(self):
        rows = _per_sample(n_scenes=2, frames_per_phase=3)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        first = cells[0]
        # Every non-fixed key must be ``metric:*``.
        for k in first:
            if k in FIXED_COLUMNS:
                continue
            assert k.startswith(METRIC_PREFIX), f"unprefixed key: {k}"
        assert metric_col("absrel") in first
        assert metric_col("rmse") in first
        assert metric_col("delta1") in first

    def test_metric_values_are_means(self):
        rows = []
        # Two frames per cell with values 0.10 and 0.30 → mean 0.20.
        for v in [0.10, 0.30]:
            rows.append(
                {"id": "x", "scene": "s0", "phase": "clu", "absrel": v, "rmse": 0.5}
            )
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        assert len(cells) == 1
        assert cells[0]["metric:absrel"] == pytest.approx(0.20)
        assert cells[0]["metric:rmse"] == pytest.approx(0.50)
        assert cells[0]["n_samples"] == 2

    def test_n_samples_counted(self):
        rows = _per_sample(n_scenes=2, frames_per_phase=7)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        assert all(c["n_samples"] == 7 for c in cells)

    def test_latency_carried(self):
        rows = _per_sample(n_scenes=2, frames_per_phase=3)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        assert all(c["latency_ms_mean"] == pytest.approx(42.0) for c in cells)

    def test_difficulty_carried(self):
        rows = _per_sample(n_scenes=2, frames_per_phase=3)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        assert all(c["difficulty"] == "easy" for c in cells)

    def test_provenance_stamped(self):
        rows = _per_sample(n_scenes=1, frames_per_phase=1)
        cells = cells_from_per_sample(
            rows, model_name="DA3", task="monocular_depth", git_sha="abc123"
        )
        assert all(c["model_name"] == "DA3" for c in cells)
        assert all(c["task"] == "monocular_depth" for c in cells)
        assert all(c["git_sha"] == "abc123" for c in cells)
        assert all(c["timestamp_utc"] for c in cells)
        assert all(c["toolkit_version"] for c in cells)

    def test_skips_rows_without_scene_or_phase(self):
        rows = [
            {"id": "x", "absrel": 0.1},  # no scene/phase
            {"id": "y", "scene": "s", "phase": "clu", "absrel": 0.2},
        ]
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        assert len(cells) == 1

    def test_explicit_metric_keys(self):
        rows = _per_sample(n_scenes=1, frames_per_phase=2)
        cells = cells_from_per_sample(
            rows, model_name="m", task="t", metric_keys=["absrel"]
        )
        # Only the requested metric column should be in output.
        first = cells[0]
        assert "metric:absrel" in first
        assert "metric:rmse" not in first
        assert "metric:delta1" not in first

    def test_metric_keys_helper(self):
        rows = _per_sample(n_scenes=1, frames_per_phase=1)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        keys = metric_keys(cells[0])
        assert "absrel" in keys
        assert "rmse" in keys
        assert "delta1" in keys

    def test_empty_input(self):
        cells = cells_from_per_sample([], model_name="m", task="t")
        assert cells == []


# --------------------------------------------------------------------------- #
# JSONL round-trip
# --------------------------------------------------------------------------- #


class TestJsonlRoundTrip:
    def test_round_trip(self, tmp_path):
        rows = _per_sample(n_scenes=3, frames_per_phase=2)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        path = tmp_path / "cells.jsonl"
        result = write_cells(cells, path, format="jsonl")
        assert result.format == "jsonl"
        assert result.n_rows == len(cells)
        assert result.path == path

        loaded = read_cells(path)
        assert len(loaded) == len(cells)
        # Same metric values survive the round trip.
        for orig, back in zip(cells, loaded, strict=True):
            assert orig["metric:absrel"] == back["metric:absrel"]
            assert orig["scene_id"] == back["scene_id"]

    def test_auto_format_jsonl_extension(self, tmp_path):
        rows = _per_sample(n_scenes=1, frames_per_phase=1)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        path = tmp_path / "x.jsonl"
        r = write_cells(cells, path, format="auto")
        assert r.format == "jsonl"


# --------------------------------------------------------------------------- #
# Parquet round-trip (only runs when pyarrow is installed)
# --------------------------------------------------------------------------- #


class TestParquetRoundTrip:
    @pytest.fixture(autouse=True)
    def _require_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_round_trip(self, tmp_path):
        rows = _per_sample(n_scenes=3, frames_per_phase=4)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        path = tmp_path / "cells.parquet"
        r = write_cells(cells, path, format="parquet")
        assert r.format == "parquet"
        assert r.n_rows == len(cells)

        loaded = read_cells(path)
        assert len(loaded) == len(cells)
        for orig, back in zip(cells, loaded, strict=True):
            assert orig["metric:absrel"] == back["metric:absrel"]
            assert orig["phase"] == back["phase"]

    def test_auto_format_parquet_extension(self, tmp_path):
        rows = _per_sample(n_scenes=1, frames_per_phase=1)
        cells = cells_from_per_sample(rows, model_name="m", task="t")
        path = tmp_path / "x.parquet"
        r = write_cells(cells, path, format="auto")
        assert r.format == "parquet"

    def test_empty_cells_writes_schema_only(self, tmp_path):
        path = tmp_path / "empty.parquet"
        r = write_cells([], path, format="parquet")
        assert r.n_rows == 0
        loaded = read_cells(path)
        assert loaded == []


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


class TestMerge:
    def test_merge_jsonl(self, tmp_path):
        a = cells_from_per_sample(
            _per_sample(n_scenes=2, frames_per_phase=2), model_name="m1", task="t"
        )
        b = cells_from_per_sample(
            _per_sample(n_scenes=3, frames_per_phase=2), model_name="m2", task="t"
        )
        pa = tmp_path / "a.jsonl"
        pb = tmp_path / "b.jsonl"
        write_cells(a, pa, format="jsonl")
        write_cells(b, pb, format="jsonl")

        merged = merge_cells([pa, pb])
        assert len(merged) == len(a) + len(b)
        models = {c["model_name"] for c in merged}
        assert models == {"m1", "m2"}
