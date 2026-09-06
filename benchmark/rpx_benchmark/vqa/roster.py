"""Frozen single-image model roster supplied for the RPX VQA benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ConfigError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    size_b: float
    vram_gb: int
    bbox_mode: str
    capabilities: frozenset[str]


ALL_TASKS = frozenset({"bbox"})

MODELS = (
    ModelSpec("paligemma2-3b", "PaliGemma 2 3B", 3.0, 7, "native_loc", ALL_TASKS),
    ModelSpec("qwen2.5-vl-3b", "Qwen2.5-VL 3B", 3.0, 7, "native", ALL_TASKS),
    ModelSpec("gemma3-4b", "Gemma 3 4B", 4.0, 10, "prompted", ALL_TASKS),
    ModelSpec("phi-3.5-vision-4b", "Phi-3.5-Vision 4.2B", 4.2, 10, "prompted", ALL_TASKS),
    ModelSpec("llava-onevision-7b", "LLaVA-OneVision 7B", 7.0, 16, "prompted", ALL_TASKS),
    ModelSpec("qwen2.5-vl-7b", "Qwen2.5-VL 7B", 7.0, 15, "native", ALL_TASKS),
    ModelSpec("idefics3-8b", "Idefics3 8B", 8.0, 18, "prompted", ALL_TASKS),
    ModelSpec("internvl2.5-8b", "InternVL 2.5 8B", 8.0, 17, "prompted", ALL_TASKS),
    ModelSpec("paligemma2-10b", "PaliGemma 2 10B", 10.0, 21, "native_loc", ALL_TASKS),
    ModelSpec("gemma3-12b", "Gemma 3 12B", 12.0, 28, "prompted", ALL_TASKS),
)


def get_model(key: str) -> ModelSpec:
    try:
        return next(model for model in MODELS if model.key == key)
    except StopIteration as exc:
        raise ConfigError(
            f"unknown VQA model {key!r}", hint="choose a key from the frozen model roster"
        ) from exc
