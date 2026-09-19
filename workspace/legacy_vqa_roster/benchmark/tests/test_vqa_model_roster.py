"""Contract tests for the shared RPX VQA model roster."""

from __future__ import annotations

import json
from pathlib import Path

from rpx_benchmark.adapters.vqa_scaffold import (
    ALL_VQA_SUBTASKS,
    VQA_MODEL_CARDS,
    VQA_SUBTASK_CONTRACTS,
    VQASubtask,
    models_for_subtask,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docker" / "vqa-smoke" / "model-matrix.json"


def test_common_roster_has_ten_ordered_model_variants():
    assert list(VQA_MODEL_CARDS) == [
        "paligemma-2-3b",
        "paligemma-2-10b",
        "gemma-4-4b",
        "gemma-4-12b",
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "internvl-2.5-8b",
        "llava-onevision-7b",
        "phi-3.5-vision-4.2b",
        "idefics3-8b",
    ]


def test_every_model_runs_every_subtask_with_multi_image_support():
    for key, card in VQA_MODEL_CARDS.items():
        assert card.subtasks == ALL_VQA_SUBTASKS, key
        assert card.supports_multi_image, key
    for subtask in VQASubtask:
        assert models_for_subtask(subtask) == tuple(VQA_MODEL_CARDS)


def test_subtask_output_and_primary_metric_contracts():
    expected = {
        VQASubtask.DIRECTIONAL_BINARY: ("binary_text", "accuracy"),
        VQASubtask.SPATIAL_BBOX: ("bbox", "iou_at_0.5"),
        VQASubtask.ATTRIBUTE_COMPOSITION: ("free_text", "exact_match_accuracy"),
    }
    assert set(VQA_SUBTASK_CONTRACTS) == ALL_VQA_SUBTASKS
    for subtask, pair in expected.items():
        contract = VQA_SUBTASK_CONTRACTS[subtask]
        assert (contract.output_kind, contract.primary_metric) == pair


def test_bbox_protocol_is_native_only_where_handoff_says_so():
    assert {key for key, card in VQA_MODEL_CARDS.items() if card.bbox_output != "prompted"} == {
        "paligemma-2-3b",
        "paligemma-2-10b",
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
    }


def test_docker_matrix_matches_python_roster_exactly():
    payload = json.loads(MATRIX.read_text())
    rows = payload["models"]
    assert [row["order"] for row in rows] == list(range(1, 11))
    assert [row["id"] for row in rows] == list(VQA_MODEL_CARDS)
    assert payload["subtasks"] == [subtask.value for subtask in VQASubtask]
    for row in rows:
        card = VQA_MODEL_CARDS[row["id"]]
        assert row["size"] == card.size
        assert row["dependency_group"] == card.dependency_group
        assert row["bbox_output"] == card.bbox_output


def test_roster_has_seven_dependency_environments():
    groups = {card.dependency_group for card in VQA_MODEL_CARDS.values()}
    assert groups == {
        "paligemma-2",
        "gemma-4",
        "qwen2.5-vl",
        "internvl-2.5",
        "llava-onevision",
        "phi-3.5-vision",
        "idefics3",
    }
