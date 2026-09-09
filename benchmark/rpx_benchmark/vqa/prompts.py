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


def build_semantic_diagnostic_prompt(sample: VQASample, model_key: str) -> PromptSpec:
    """Ask for the answer label only, without leaking the ground truth.

    This is deliberately diagnostic rather than a scored benchmark protocol:
    it separates target selection/reasoning from coordinate prediction.
    """
    question = display_question(sample.question)
    instruction = (
        f"{question}\nIdentify the one object that answers the question. "
        "For a relational question, name the result object, not the reference "
        "object named or shown in the question. Return only the shortest common "
        "noun phrase naming that object: 1 to 5 words, with no sentence, article, "
        "coordinates, Markdown, punctuation, or explanation."
    )
    if model_key.startswith("paligemma2-"):
        instruction = f"answer en {instruction}\n"
    return PromptSpec(instruction, 32, "diagnostic_semantic_label")


def build_oracle_localization_prompt(sample: VQASample, model_key: str) -> PromptSpec:
    """Localize the known GT label in the target image for diagnosis only."""
    return build_label_localization_prompt(
        display_question(sample.answer), model_key, oracle=True
    )


def build_label_localization_prompt(
    label: str, model_key: str, *, oracle: bool = False
) -> PromptSpec:
    """Ground a label/referring phrase without canonicalizing its wording.

    The predicted-label diagnostic deliberately passes the model's own answer
    phrase through unchanged.  Correctness is determined spatially against the
    target identity, so synonyms and harmless modifiers cannot create a false
    semantic failure merely because their strings differ from the catalog.
    """
    label = display_question(label)
    output_kind = (
        "diagnostic_oracle_bbox" if oracle else "diagnostic_predicted_label_bbox"
    )
    if model_key.startswith("paligemma2-"):
        return PromptSpec(f"detect {label}\n", 64, output_kind)
    return PromptSpec(
        f'Locate the visible object named "{label}" in the target image. '
        'Return only JSON: {"label":"object name","bbox":'
        "[x_min,y_min,x_max,y_max]}. Normalize every bbox coordinate from 0 "
        "to 1000 relative to the target-image width and height. Use XYXY corner "
        "order and enclose the entire object. No Markdown or explanation.",
        96,
        output_kind,
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
            # PaliGemma is prefix-trained rather than conversational, so retain
            # its official VQA prefix while asking it to solve and localize in
            # one generation. No predicted-label follow-up is allowed here.
            return PromptSpec(
                f"answer en {_bbox_instruction(question)}\n",
                96,
                "bbox_json_normalized_1000",
            )
        return PromptSpec(
            _bbox_instruction(question),
            96,
            "bbox_json_normalized_1000",
        )
    raise ConfigError(
        f"unsupported VQA question type: {sample.question_type}",
        hint="use a type in the frozen VQA task contract",
    )
