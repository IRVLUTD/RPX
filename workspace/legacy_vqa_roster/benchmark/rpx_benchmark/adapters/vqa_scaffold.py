"""Canonical model and I/O contracts for the RPX VQA benchmark.

The roster is intentionally independent of model imports.  Smoke tooling can
therefore enumerate every required model before any of the mutually
incompatible, GPU-heavy dependencies are installed.

The source of truth is the common-roster table supplied with the VQA handoff
(``image2.png``).  All ten rows are evaluated on the same three benchmark
subtasks; model size is part of the key because it identifies a distinct
checkpoint and paper row.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Literal


class VQASubtask(str, Enum):
    """The three agreed VQA benchmark outputs from the handoff."""

    DIRECTIONAL_BINARY = "directional_binary"
    SPATIAL_BBOX = "spatial_bbox"
    ATTRIBUTE_COMPOSITION = "attribute_composition"


@dataclass(frozen=True)
class VQASubtaskContract:
    input_kind: str
    output_kind: Literal["binary_text", "bbox", "free_text"]
    primary_metric: str
    diagnostic_metrics: tuple[str, ...]


VQA_SUBTASK_CONTRACTS: dict[VQASubtask, VQASubtaskContract] = {
    VQASubtask.DIRECTIONAL_BINARY: VQASubtaskContract(
        input_kind="scene_image+question",
        output_kind="binary_text",
        primary_metric="accuracy",
        diagnostic_metrics=("yes_rate", "unparseable_rate"),
    ),
    VQASubtask.SPATIAL_BBOX: VQASubtaskContract(
        input_kind="scene_image+referring_expression",
        output_kind="bbox",
        primary_metric="iou_at_0.5",
        diagnostic_metrics=("mean_iou", "centroid_hit_rate", "unparseable_rate"),
    ),
    VQASubtask.ATTRIBUTE_COMPOSITION: VQASubtaskContract(
        input_kind="scene_image+question",
        output_kind="free_text",
        primary_metric="exact_match_accuracy",
        diagnostic_metrics=("embedding_similarity_at_0.70", "unparseable_rate"),
    ),
}


@dataclass(frozen=True)
class VQAModelCard:
    """Static metadata used by adapter, smoke and Docker tooling."""

    name: str
    size: str
    family: str
    dependency_group: str
    bbox_output: Literal["native_loc", "native_box", "prompted"]
    subtasks: FrozenSet[VQASubtask]
    supports_multi_image: bool = True


ALL_VQA_SUBTASKS: FrozenSet[VQASubtask] = frozenset(VQASubtask)


# Order matches the supplied common-roster table and is stable for cumulative
# image construction.  Checkpoint revisions are deliberately not guessed here;
# they belong in the per-family Docker implementation after upstream
# verification.
VQA_MODEL_CARDS: dict[str, VQAModelCard] = {
    "paligemma-2-3b": VQAModelCard(
        "PaliGemma 2 3B", "3B", "paligemma-2", "paligemma-2", "native_loc", ALL_VQA_SUBTASKS
    ),
    "paligemma-2-10b": VQAModelCard(
        "PaliGemma 2 10B", "10B", "paligemma-2", "paligemma-2", "native_loc", ALL_VQA_SUBTASKS
    ),
    "gemma-4-4b": VQAModelCard(
        "Gemma 4 4B", "4B", "gemma-4", "gemma-4", "prompted", ALL_VQA_SUBTASKS
    ),
    "gemma-4-12b": VQAModelCard(
        "Gemma 4 12B", "12B", "gemma-4", "gemma-4", "prompted", ALL_VQA_SUBTASKS
    ),
    "qwen2.5-vl-3b": VQAModelCard(
        "Qwen2.5-VL 3B", "3B", "qwen2.5-vl", "qwen2.5-vl", "native_box", ALL_VQA_SUBTASKS
    ),
    "qwen2.5-vl-7b": VQAModelCard(
        "Qwen2.5-VL 7B", "7B", "qwen2.5-vl", "qwen2.5-vl", "native_box", ALL_VQA_SUBTASKS
    ),
    "internvl-2.5-8b": VQAModelCard(
        "InternVL 2.5 8B", "8B", "internvl-2.5", "internvl-2.5", "prompted", ALL_VQA_SUBTASKS
    ),
    "llava-onevision-7b": VQAModelCard(
        "LLaVA-OneVision 7B",
        "7B",
        "llava-onevision",
        "llava-onevision",
        "prompted",
        ALL_VQA_SUBTASKS,
    ),
    "phi-3.5-vision-4.2b": VQAModelCard(
        "Phi-3.5-Vision 4.2B",
        "4.2B",
        "phi-3.5-vision",
        "phi-3.5-vision",
        "prompted",
        ALL_VQA_SUBTASKS,
    ),
    "idefics3-8b": VQAModelCard(
        "Idefics3 8B", "8B", "idefics3", "idefics3", "prompted", ALL_VQA_SUBTASKS
    ),
}


def models_for_subtask(subtask: VQASubtask) -> tuple[str, ...]:
    """Return canonical model keys in paper/Docker order."""

    return tuple(key for key, card in VQA_MODEL_CARDS.items() if subtask in card.subtasks)


__all__ = [
    "ALL_VQA_SUBTASKS",
    "VQA_MODEL_CARDS",
    "VQA_SUBTASK_CONTRACTS",
    "VQAModelCard",
    "VQASubtask",
    "VQASubtaskContract",
    "models_for_subtask",
]
