"""Benchmark contracts for RPX's single-image VQA tasks."""

from .contract import (
    BBOX_TYPES,
    BINARY_TYPES,
    TASK_TYPES,
    VQASample,
    load_manifest,
)
from .metrics import score_predictions
from .outputs import ParsedOutput, parse_output
from .prompts import PromptSpec, build_prompt
from .roster import MODELS, ModelSpec

__all__ = [
    "BBOX_TYPES",
    "BINARY_TYPES",
    "MODELS",
    "TASK_TYPES",
    "ModelSpec",
    "ParsedOutput",
    "PromptSpec",
    "VQASample",
    "build_prompt",
    "load_manifest",
    "parse_output",
    "score_predictions",
]
