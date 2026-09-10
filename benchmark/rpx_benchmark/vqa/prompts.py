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


def native_referring_expression(sample: VQASample) -> str:
    """Rewrite an RPX question into a native grounding expression.

    Florence/PaliGemma grounding heads are trained on referring expressions,
    not interrogative VQA syntax.  This transformation is deterministic and
    uses only the public question/type -- never the answer or GT bbox -- so the
    scored path remains one model generation.
    """
    question = display_question(sample.question).strip().rstrip("?")
    lower = question.lower()
    if sample.question_type == "spatial_lr_extreme":
        if "left" in lower:
            return "the leftmost object"
        if "right" in lower:
            return "the rightmost object"
    if (
        sample.question_type == "depth_closest"
        and "closest" in lower
        and "camera" in lower
    ):
        return "the object closest to the camera"
    if sample.question_type == "spatial_farthest":
        marker = "farthest from the "
        if marker in lower:
            return "the object farthest from " + question[lower.index(marker) + len(marker):]
    if sample.question_type == "inctx_spatial_farthest":
        return "the object in image 2 farthest from the object shown in image 1"

    # Normal attribute questions all begin with one of these stable forms.
    for prefix in ("which object is ", "what is the object "):
        if lower.startswith(prefix):
            description = question[len(prefix):]
            if sample.question_type == "attr_single_color":
                return "the " + description + " object"
            return "the object " + description

    # In-context questions already encode the complete reference relation;
    # remove only the redundant request for output formatting and turn the
    # interrogative into a grammatical relative clause.
    suffix = "? What is its bounding box in Image 2"
    if question.endswith(suffix):
        question = question[: -len(suffix)]
    prefix = "Which object in Image 2 "
    if question.startswith(prefix):
        relation = question[len(prefix):]
        if relation.startswith("has "):
            relation = "that " + relation
        elif relation.startswith("is "):
            relation = "that " + relation
        return "the object in image 2 " + relation
    if question.lower().startswith("which object "):
        return "the object " + question[len("Which object "):]
    return question


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
    if model_key.startswith("florence2-"):
        # Florence task tokens are a closed native interface.  Extra formatting
        # prose is treated as VQA question text and can be echoed by the model.
        return PromptSpec(question, 32, "diagnostic_semantic_label")
    if model_key.startswith("paligemma2-"):
        # Follow the checkpoint's documented `answer {lang} {question}` form
        # exactly.  Enforce short-answer formatting in the diagnostic parser.
        return PromptSpec(
            f"answer en {question}\n", 32, "diagnostic_semantic_label"
        )
    instruction = (
        f"{question}\nIdentify the one object that answers the question. "
        "For a relational question, name the result object, not the reference "
        "object named or shown in the question. Return only the shortest common "
        "noun phrase naming that object: 1 to 5 words, with no sentence, article, "
        "coordinates, Markdown, punctuation, or explanation."
    )
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
    if model_key.startswith("florence2-"):
        # Florence's official phrase-grounding interface is TASK + phrase.
        # Adding JSON instructions makes every noun in those instructions a
        # grounding candidate and was the cause of the invalid smoke results.
        return PromptSpec(label, 128, output_kind)
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
        if model_key.startswith("florence2-"):
            # One scored native call: a deterministic, GT-free rewrite of the
            # question is supplied to caption-to-phrase grounding. Only the
            # returned bbox is evaluated; no answer-label call is involved.
            return PromptSpec(
                native_referring_expression(sample),
                128,
                "bbox_native_question_grounding",
            )
        if model_key.startswith("paligemma2-"):
            # PaliGemma's detector accepts a referring phrase rather than an
            # interrogative. Rewrite question syntax without using its answer,
            # then perform selection/localization in one generation.
            expression = native_referring_expression(sample)
            return PromptSpec(
                f"detect {expression}\n", 64, "bbox_native_question_grounding"
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
