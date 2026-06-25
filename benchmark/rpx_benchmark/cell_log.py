"""Per-(scene, phase) metric log — the canonical experiment artifact.

The runner produces a flat list of per-sample metric rows. This module
groups them into per-(scene, phase) cells and writes them to a typed
log file. The log is the *single source of truth* for downstream
analysis: Φ, JEDI, ESD-stratified summaries, bootstrap CIs, sensitivity
sweeps, and the paper's per-task results tables all read from it.

Why a separate log?
-------------------

- **JEDI bounds must be frozen across the full model zoo** before
  individual model JEDI values are comparable. The cell log lets us
  collect every model first, then freeze bounds once, then recompute
  JEDI without re-running models.
- **Sensitivity to scoring choices** (α, ε, Wilks vs Pillai, ESD
  weighting tier, etc.) reduces to a re-read of the same cells.
- **Bootstrap CIs on Φ** resample scenes from the cell log; no need to
  re-run any model.
- **Bug fixes** in metric calculators can be partially recovered by
  re-running on cached predictions; the cell log records what the
  numbers *were* at run time so the diff is interpretable.

Storage format
--------------

Parquet by default (typed columns, fast pandas/DuckDB reads). When
``pyarrow`` is unavailable, falls back to JSON Lines — same schema,
slightly larger files.

Schema
------

One row per (model, task, scene, phase, frame_budget) cell:

  model_name        str       e.g. "DA3 Metric-L"
  task              str       TaskType.value, e.g. "monocular_depth"
  scene_id          str       canonical scene identifier
  phase             str       "clu" | "int" | "cln"
  difficulty        str|null  "easy" | "medium" | "hard" (when known)
  n_samples         int       number of frames averaged into this cell
  frame_budget      int       0 = full clip / per-frame task; >0 = the
                              subsampled clip size for the Video Depth temporal-
                              resolution ablation (typical values 25,
                              75, 150). Defaulting to 0 keeps the column
                              meaningful for Image Depth (always 0) while
                              giving Video Depth a single column to slice on.
  latency_ms_mean   float|n   mean per-sample latency
  toolkit_version   str       rpx_benchmark.__version__
  git_sha           str|null  caller-supplied (e.g. adapter version)
  timestamp_utc     str       ISO-8601, when the row was written
  metric:<name>     float|n   one column per metric key (prefixed
                              ``metric:`` so the metric columns are
                              namespaced and never collide with the
                              fixed-schema columns above)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .exceptions import DatasetError
from .logging_utils import get_logger

log = get_logger(__name__)


# Columns that live in the fixed schema. Metric columns are prefixed
# ``metric:`` so they're always namespaced apart.
FIXED_COLUMNS: tuple[str, ...] = (
    "model_name",
    "task",
    "scene_id",
    "phase",
    "difficulty",
    "n_samples",
    "frame_budget",
    "latency_ms_mean",
    "toolkit_version",
    "git_sha",
    "timestamp_utc",
)

METRIC_PREFIX = "metric:"


def metric_col(name: str) -> str:
    """Namespaced column name for a metric: ``metric:absrel``, etc."""
    return f"{METRIC_PREFIX}{name}"


def metric_keys(row: Mapping[str, Any]) -> List[str]:
    """Return the metric keys present in a cell row (without prefix)."""
    return [k[len(METRIC_PREFIX) :] for k in row if k.startswith(METRIC_PREFIX)]


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


def _toolkit_version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 — best-effort, never fail the log write
        return "unknown"


def cells_from_per_sample(
    per_sample_metrics: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    task: str,
    git_sha: Optional[str] = None,
    metric_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Group per-sample metric rows into per-(scene, phase) cells.

    Each output cell carries the *mean* of every metric across the frames
    that fell in that (scene, phase) bucket. Non-numeric fields (id,
    difficulty) are taken from the first row in the bucket; difficulty
    is assumed stable per (scene, phase).

    Parameters
    ----------
    per_sample_metrics
        Same list the runner builds: each dict has metric keys + ``id``
        / ``phase`` / ``scene`` / ``difficulty`` / ``latency_ms``.
    model_name, task
        Stamped onto every emitted row.
    git_sha
        Optional adapter version stamp.
    metric_keys
        If given, only these metric columns are emitted. Otherwise all
        numeric keys that aren't fixed-schema metadata are kept.

    Returns
    -------
    list of dict
        One row per cell, ready to feed :func:`write_cells`.
    """
    metadata_keys = {"id", "phase", "difficulty", "scene", "latency_ms"}

    # Auto-detect metric keys from the first row if not given.
    if metric_keys is None and per_sample_metrics:
        first = per_sample_metrics[0]
        metric_keys = [
            k
            for k, v in first.items()
            if k not in metadata_keys and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
    metric_keys = list(metric_keys or [])

    # Bucket by (scene, phase).
    buckets: Dict[tuple[Any, Any], List[Mapping[str, Any]]] = {}
    for row in per_sample_metrics:
        scene = row.get("scene")
        phase = row.get("phase")
        if scene is None or phase is None:
            continue
        buckets.setdefault((scene, phase), []).append(row)

    timestamp = datetime.now(timezone.utc).isoformat()
    version = _toolkit_version()

    cells: List[Dict[str, Any]] = []
    for (scene, phase), rows in buckets.items():
        # frame_budget is the first-seen value for the cell (per-sample
        # rows for one (scene, phase) should always carry the same
        # budget — they come from one VideoDepthDataset run). Defaults to 0
        # for per-frame tasks that don't subsample.
        budget = next(
            (int(r["frame_budget"]) for r in rows if isinstance(r.get("frame_budget"), int)),
            0,
        )
        cell: Dict[str, Any] = {
            "model_name": model_name,
            "task": task,
            "scene_id": str(scene),
            "phase": str(phase.value if hasattr(phase, "value") else phase),
            "difficulty": _first_str(rows, "difficulty"),
            "n_samples": len(rows),
            "frame_budget": budget,
            "latency_ms_mean": _mean_or_none(rows, "latency_ms"),
            "toolkit_version": version,
            "git_sha": git_sha,
            "timestamp_utc": timestamp,
        }
        for k in metric_keys:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            cell[metric_col(k)] = float(np.mean(vals)) if vals else None
        cells.append(cell)

    return cells


def _first_str(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[str]:
    """First non-null string-like value for ``key`` across rows."""
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        return str(v.value if hasattr(v, "value") else v)
    return None


def _mean_or_none(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def cell_from_metrics(
    metrics: Mapping[str, Any],
    *,
    model_name: str,
    task: str,
    scene_id: str,
    phase: Any,
    n_samples: int,
    difficulty: Any = None,
    latency_ms_mean: Optional[float] = None,
    git_sha: Optional[str] = None,
    metric_keys: Optional[Sequence[str]] = None,
    timestamp_utc: Optional[str] = None,
    frame_budget: int = 0,
) -> Dict[str, Any]:
    """Build one canonical (scene, phase) cell from *pre-aggregated* metrics.

    Companion to :func:`cells_from_per_sample`.  Use this for
    clip/sequence-level tasks (Video Depth, tracking) where one model forward
    already yields a single cell's worth of numbers — there are no
    per-sample rows to group.  The emitted row has the **identical
    schema** (:data:`FIXED_COLUMNS` + ``metric:`` columns), so per-frame
    and per-clip cells merge into one ``cells.parquet`` and feed the same
    cross-phase / phase×difficulty Φ analysis.

    Parameters
    ----------
    metrics
        Already-aggregated metric name → scalar (e.g. the output of
        ``evaluate_video_depth_clip``). ``NaN`` values are preserved (the metric
        was attempted but undefined — e.g. OPW with no flow backend);
        the downstream MANOVA drops them.
    model_name, task, scene_id, phase, n_samples
        Fixed-schema identity of the cell. ``phase`` / ``difficulty`` may
        be enums (``.value`` is taken) or plain strings.
    difficulty
        The scene's ESD tier (easy/medium/hard) — required for
        cross-split Φ; ``None`` when unknown.
    """
    if metric_keys is None:
        metric_keys = [
            k for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    def _label(v: Any) -> Optional[str]:
        return None if v is None else str(v.value if hasattr(v, "value") else v)

    cell: Dict[str, Any] = {
        "model_name": model_name,
        "task": task,
        "scene_id": str(scene_id),
        "phase": _label(phase),
        "difficulty": _label(difficulty),
        "n_samples": int(n_samples),
        "frame_budget": int(frame_budget),
        "latency_ms_mean": latency_ms_mean,
        "toolkit_version": _toolkit_version(),
        "git_sha": git_sha,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
    }
    for k in metric_keys:
        v = metrics.get(k)
        cell[metric_col(k)] = (
            float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        )
    return cell


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


@dataclass
class WriteResult:
    """Outcome of a :func:`write_cells` call."""

    path: Path
    format: str  # "parquet" | "jsonl"
    n_rows: int
    metric_columns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "n_rows": self.n_rows,
            "metric_columns": list(self.metric_columns),
        }


def _try_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def write_cells(
    cells: Sequence[Mapping[str, Any]],
    path: os.PathLike,
    *,
    format: str = "auto",
) -> WriteResult:
    """Write cells to disk.

    Parameters
    ----------
    cells
        Output of :func:`cells_from_per_sample`.
    path
        File path. If ``format='auto'`` and the extension is ``.parquet``,
        Parquet is preferred; otherwise JSON Lines. Missing extension
        defaults to Parquet when ``pyarrow`` is available, JSONL otherwise.
    format
        Override: ``"parquet"`` | ``"jsonl"`` | ``"auto"``.

    Returns
    -------
    WriteResult
        Path, chosen format, and the metric-column names that were written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        if out.suffix.lower() == ".jsonl":
            format = "jsonl"
        elif out.suffix.lower() in (".parquet", ".pq"):
            format = "parquet"
        elif _try_parquet():
            format = "parquet"
            out = out.with_suffix(".parquet")
        else:
            format = "jsonl"
            out = out.with_suffix(".jsonl")

    if format == "parquet":
        if not _try_parquet():
            raise DatasetError(
                "Parquet write requested but pyarrow is not installed.",
                hint="pip install 'rpx-benchmark[hub]' (or pyarrow directly).",
            )
        _write_parquet(cells, out)
    elif format == "jsonl":
        _write_jsonl(cells, out)
    else:
        raise DatasetError(
            f"unknown cell-log format {format!r}",
            hint="Use 'parquet', 'jsonl', or 'auto'.",
        )

    metric_columns = sorted({k for row in cells for k in row if k.startswith(METRIC_PREFIX)})
    return WriteResult(
        path=out,
        format=format,
        n_rows=len(cells),
        metric_columns=metric_columns,
    )


def _write_parquet(cells: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not cells:
        # Write an empty file with just the fixed schema so readers don't
        # crash on a missing column. Metric columns are unknown until we
        # see data.
        schema = pa.schema(
            [
                pa.field("model_name", pa.string()),
                pa.field("task", pa.string()),
                pa.field("scene_id", pa.string()),
                pa.field("phase", pa.string()),
                pa.field("difficulty", pa.string()),
                pa.field("n_samples", pa.int64()),
                pa.field("frame_budget", pa.int64()),
                pa.field("latency_ms_mean", pa.float64()),
                pa.field("toolkit_version", pa.string()),
                pa.field("git_sha", pa.string()),
                pa.field("timestamp_utc", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_arrays([[] for _ in schema], schema=schema), path)
        return

    # Union of every column key across rows so missing values become null.
    all_keys: List[str] = list(FIXED_COLUMNS)
    seen = set(all_keys)
    for row in cells:
        for k in row:
            if k not in seen and k.startswith(METRIC_PREFIX):
                all_keys.append(k)
                seen.add(k)

    columns: Dict[str, List[Any]] = {k: [] for k in all_keys}
    for row in cells:
        for k in all_keys:
            columns[k].append(row.get(k))

    table = pa.table(columns)
    pq.write_table(table, path)


def _write_jsonl(cells: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for row in cells:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def read_cells(path: os.PathLike) -> List[Dict[str, Any]]:
    """Read cells back into a list of dicts.

    Format detected from extension: ``.parquet`` / ``.pq`` → Parquet,
    anything else → JSON Lines.
    """
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"cell log not found: {p}")

    if p.suffix.lower() in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise DatasetError(
                f"can't read {p} — pyarrow not installed.",
                hint="pip install 'rpx-benchmark[hub]'",
            ) from e
        table = pq.read_table(p)
        return table.to_pylist()

    rows: List[Dict[str, Any]] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_cells(paths: Iterable[os.PathLike]) -> List[Dict[str, Any]]:
    """Concatenate multiple cell logs into one list.

    Per-log column sets may differ (different tasks have different
    metric columns); the union schema is preserved on output.
    """
    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(read_cells(p))
    return merged


__all__ = [
    "FIXED_COLUMNS",
    "METRIC_PREFIX",
    "WriteResult",
    "cells_from_per_sample",
    "merge_cells",
    "metric_col",
    "metric_keys",
    "read_cells",
    "write_cells",
]
