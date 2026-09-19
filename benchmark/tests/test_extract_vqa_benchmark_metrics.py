from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_vqa_benchmark_metrics.py"


def _module() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_scene_csv_contract_contains_repeated_measures_columns() -> None:
    fields = set(_module()["SCENE_FIELDS"])
    assert {
        "model",
        "scene",
        "kind",
        "phase",
        "capture",
        "bbox_accuracy_at_0_5",
        "bbox_mean_giou",
        "bbox_center_in_gt_accuracy",
    } <= fields


def test_scene_rows_flatten_nested_metrics() -> None:
    metrics = {
        "bbox_validity_rate": 0.9,
        "bbox_accuracy_at_0_25": 0.8,
        "bbox_accuracy_at_0_5": 0.7,
        "bbox_accuracy_at_0_75": 0.6,
        "bbox_mean_accuracy_50_95": 0.5,
        "bbox_mean_giou": 0.4,
        "bbox_mean_giou_valid": 0.3,
        "bbox_center_in_gt_accuracy": 0.2,
        "bbox_mean_iou": 0.1,
        "parse_rate": 0.95,
    }
    results = [
        {
            "model": "internvl3.5-1b",
            "metrics_by_scene": [
                {
                    "scene": "scene001",
                    "kind": "ego",
                    "phase": None,
                    "capture": "ego",
                    "rows": 12,
                    "predictions": 12,
                    "inference_failures": 0,
                    "metrics": metrics,
                }
            ],
        }
    ]

    rows = _module()["scene_rows"](results)

    assert rows == [
        {
            "model": "internvl3.5-1b",
            "scene": "scene001",
            "kind": "ego",
            "phase": "",
            "capture": "ego",
            "rows": 12,
            "predictions": 12,
            "inference_failures": 0,
            **metrics,
        }
    ]
