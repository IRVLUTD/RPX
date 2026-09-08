"""Task prompts before each model's official chat/image template is applied."""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ConfigError
from .contract import BBOX_TYPES, BINARY_TYPES, VQASample


@dataclass(frozen=True)
class PromptSpec:
    text: str
    max_new_tokens: int
    output_kind: str


def display_question(question: str) -> str:
    """Hide catalog encoding without changing the stored ground truth label."""
    return " ".join(question.replace("_", " ").split())


def _bbox_instruction(question: str) -> str:
    """Stage-one instruction shared by every JSON-capable VLM.

    Separating semantic question answering from target-image grounding keeps a
    model from trying to solve both jobs in one short structured response.  The
    adapter never receives the ground-truth label: stage two is conditioned only
    on the label predicted here.
    """
    return (
        f"{question}\n"
        "Identify the one visible object that answers the question. For a relational "
        "question, answer with the result object, not the reference object named in "
        "the question. Return only its shortest common object name, with no sentence, "
        "JSON, coordinates, Markdown, or explanation."
    )


def build_prompt(sample: VQASample, model_key: str) -> PromptSpec:
    question = display_question(sample.question)
    if sample.question_type in BINARY_TYPES:
        if model_key.startswith("paligemma2-"):
            return PromptSpec(f"answer en {question}\n", 8, "binary")
        return PromptSpec(
            f"{question}\nAnswer using exactly one lowercase word: yes or no.", 4, "binary"
        )
    if sample.question_type in BBOX_TYPES:
        if model_key.startswith("paligemma2-"):
            # The runner answers the spatial question first, then grounds its
            # predicted label with PaliGemma's native detect prefix. At no point
            # does the inference path receive the ground-truth object label.
            return PromptSpec(f"answer en {question}\n", 24, "paligemma_two_stage")
        return PromptSpec(
            _bbox_instruction(question),
            32,
            "two_stage_bbox_json_normalized_1000",
        )
    raise ConfigError(
        f"unsupported VQA question type: {sample.question_type}",
        hint="use a type in the frozen VQA task contract",
    )
