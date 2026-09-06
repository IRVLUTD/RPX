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
            f'{question}\nReturn only JSON: {{"label":"object name","bbox":'
            f"[x_min,y_min,x_max,y_max]}}. Use integer coordinates in the original "
            f"{sample.img_w} by {sample.img_h} image. The bbox must enclose the object "
            "that answers the question. No Markdown or explanation.",
            64,
            "bbox_json",
        )
    raise ConfigError(
        f"unsupported VQA question type: {sample.question_type}",
        hint="use a type in the frozen VQA task contract",
    )
