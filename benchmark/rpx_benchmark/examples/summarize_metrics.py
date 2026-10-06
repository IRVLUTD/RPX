"""Calculate Φ and JEDI from CSV, JSON or JSONL raw metric records.

Records need scene/scene_id, phase, and explicitly selected metric columns.
Optional model and task columns separate independent runs. Three canonical
phases are required in each run; incomplete scenes are reported and dropped
by the existing complete-case analysis. Aggregate paper tables cannot recover
the paired observations needed for Φ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..exceptions import MetricError
from ..metrics.specs import get_spec
from ..phi_jedi_summary import CANONICAL_PHASE_ORDER, summarize_phi_jedi


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a list of records; runner JSON may wrap them in ``per_sample``."""
    records: Any
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text())
        if isinstance(records, dict):
            records = records.get("per_sample")
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(r, dict) for r in records)
    ):
        raise MetricError("Input must contain a nonempty list of raw metric records")
    return records


def calculate(records: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
    """Validate raw values and summarize each (model, task) independently."""
    if not metric_keys or len(set(metric_keys)) != len(metric_keys):
        raise MetricError("Choose at least one metric, with no duplicates")
    specs = {key: asdict(get_spec(key)) for key in metric_keys}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    aliases = {"0": "clutter", "1": "interaction", "2": "clean"}
    for index, record in enumerate(records, 1):
        row = dict(record)
        scene = row.get("scene") or row.get("scene_id")
        if scene is None or not str(scene).strip():
            raise MetricError(f"Row {index}: scene/scene_id is required")
        phase = aliases.get(str(row.get("phase")), str(row.get("phase")).lower())
        if phase not in CANONICAL_PHASE_ORDER:
            raise MetricError(f"Row {index}: phase must be clutter, interaction, clean or 0/1/2")
        row.update(scene=str(scene), phase=phase)
        for key in metric_keys:
            raw = row.get(key)
            if isinstance(raw, bool):
                raise MetricError(f"Row {index}: {key} cannot be a boolean")
            if raw is None:
                raise MetricError(f"Row {index}: {key} must be a finite number")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise MetricError(f"Row {index}: {key} must be a finite number") from exc
            if not math.isfinite(value):
                raise MetricError(f"Row {index}: {key} must be a finite number")
            row[key] = value
        group = (str(row.get("model") or "custom"), str(row.get("task") or "unspecified"))
        groups.setdefault(group, []).append(row)
    results = []
    for (model, task), rows in groups.items():
        phases = {row["phase"] for row in rows}
        if phases != set(CANONICAL_PHASE_ORDER):
            raise MetricError(
                f"{model}/{task}: all three phases are required; found {sorted(phases)}"
            )
        summary = summarize_phi_jedi(rows, metric_keys=metric_keys).to_dict()
        jedi = summary["jedi"]
        j_min = min(jedi["per_phase"].values()) if jedi else None
        results.append(
            {"model": model, "task": task, "input_rows": len(rows), "j_min": j_min, **summary}
        )
    return {
        "protocol": "three_phase_complete_case",
        "metric_specs": specs,
        "phase_order": list(CANONICAL_PHASE_ORDER),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate(read_records(args.input), args.metrics)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
