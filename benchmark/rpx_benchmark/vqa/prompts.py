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
    """Direct, model-neutral bbox instruction for every JSON-capable VLM.

    The scored path is deliberately one model call: the model sees the original
    question and every required image while choosing and localizing the answer.
    Splitting this into answer-then-ground calls compounds two independent model
    errors and no longer measures the dataset's end-to-end task.
    """
    return (
        f"{question}\n"
        "Identify and localize the one visible object that answers the question. "
        "For a relational question, localize the result object, not the reference "
        "object named or shown in the question. The answer object is in the target "
        "image (Image 2 for a two-image input; otherwise the only image). Return only "
        'JSON: {"label":"object name","bbox":[x_min,y_min,x_max,y_max]}. '
        "Normalize every bbox coordinate from 0 to 1000 relative to the target "
        "image width and height. Use XYXY corner order and enclose the entire object. "
        "No Markdown or explanation."
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
            96,
            "bbox_json_normalized_1000",
        )
    raise ConfigError(
        f"unsupported VQA question type: {sample.question_type}",
        hint="use a type in the frozen VQA task contract",
    )
