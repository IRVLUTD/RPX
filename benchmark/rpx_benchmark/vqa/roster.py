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


ALL_TASKS = frozenset({"binary", "attribute", "bbox"})
BBOX_ONLY = frozenset({"bbox"})

MODELS = (
    ModelSpec("grounding-dino", "GroundingDINO", 0.3, 2, "native", BBOX_ONLY),
    ModelSpec("florence-2-large", "Florence-2-large", 0.8, 3, "native", BBOX_ONLY),
    ModelSpec("paligemma2-3b", "PaliGemma 2 3B", 3.0, 7, "native_loc", ALL_TASKS),
    ModelSpec("qwen2.5-vl-3b", "Qwen2.5-VL 3B", 3.0, 7, "native", ALL_TASKS),
    ModelSpec("molmo-2-4b", "Molmo 2 4B", 4.0, 9, "native_point", ALL_TASKS),
    ModelSpec("qwen2.5-vl-7b", "Qwen2.5-VL 7B", 7.0, 15, "native", ALL_TASKS),
    ModelSpec("internvl2.5-8b", "InternVL 2.5 8B", 8.0, 17, "prompted", ALL_TASKS),
    ModelSpec("paligemma2-10b", "PaliGemma 2 10B", 10.0, 21, "native_loc", ALL_TASKS),
    ModelSpec("robopoint-13b", "RoboPoint 13B", 13.0, 27, "native_point", BBOX_ONLY),
    ModelSpec("cogvlm2-19b", "CogVLM2 19B", 19.0, 38, "native", ALL_TASKS),
)


def get_model(key: str) -> ModelSpec:
    try:
        return next(model for model in MODELS if model.key == key)
    except StopIteration as exc:
        raise ConfigError(
            f"unknown VQA model {key!r}", hint="choose a key from the frozen model roster"
        ) from exc
