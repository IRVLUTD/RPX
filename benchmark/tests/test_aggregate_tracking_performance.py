from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_tracking_performance import _summary  # noqa: E402


def test_summary_collects_accuracy_and_hardware(tmp_path: Path) -> None:
    model = tmp_path / "sam2"
    model.mkdir()
    means = {
        "hota": 0.6,
        "deta": 0.7,
        "assa": 0.5,
        "idf1": 0.55,
        "mota": 0.45,
        "idsw": 2.0,
    }
    (model / "paper_analysis.json").write_text(
        json.dumps({"model": "sam2", "metric_means": {"overall": means}}),
        encoding="utf-8",
    )
    cells = pd.DataFrame(
        {
            "n_frames": [250] * 300,
            "metric:idsw": [2.0] * 300,
            "latency_ms": [20.0] * 300,
            "throughput_fps": [50.0] * 300,
            "peak_gpu_memory_allocated_mb": [4096.0] * 300,
            "peak_gpu_memory_reserved_mb": [4608.0] * 300,
            "clip_wall_time_s": [5.0] * 300,
            "parameter_count": [224_000_000] * 300,
        }
    )
    cells.to_parquet(model / "combined_cells.parquet", index=False)
    easy = model / "easy"
    easy.mkdir()
    (easy / "result.json").write_text(
        json.dumps({"hardware": {"gpu_name": "test-gpu"}}), encoding="utf-8"
    )

    result = _summary(model)

    assert result["cells"] == 300
    assert result["frames"] == 75_000
    assert result["hota"] == 0.6
    assert result["idsw_total"] == 600
    assert result["latency_ms_median"] == 20.0
    assert result["propagation_fps_median"] == 50.0
    assert result["peak_gpu_memory_allocated_mb"] == 4096.0
    assert result["parameter_count"] == 224_000_000
    assert result["hardware"]["gpu_name"] == "test-gpu"
