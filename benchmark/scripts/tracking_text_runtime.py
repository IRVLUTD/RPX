"""Runtime loader for the pinned text-initialized tracking vocabulary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TEXT_VOCAB_REPO = "IRVLUTD/RPX"
TEXT_VOCAB_REVISION = "91921dcb5d328da3e0750c6b5e3f7109cb8b2bb1"
TEXT_VOCAB_PATH = (
    "tracking/metadata/text_initialization_v1/scene_condition_vocab.parquet"
)
TEXT_VOCAB_SHA256 = (
    "49db73a4a4711aef5148d87a206d5f520b726f40b7b421c692d721e01873afe1"
)


@dataclass(frozen=True)
class TextPrompt:
    mask_index: int
    prompt_text: str
    source_catalog_id: str
    object_id: str


class TrackingTextVocabulary:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if digest != TEXT_VOCAB_SHA256:
            raise RuntimeError(
                f"Tracking text vocabulary SHA-256 mismatch: {digest}; "
                f"expected {TEXT_VOCAB_SHA256}."
            )
        frame = pd.read_parquet(self.path)
        required = {
            "scene_id",
            "kind",
            "phase",
            "mask_index",
            "prompt_text",
            "source_catalog_id",
            "object_id",
        }
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Tracking text vocabulary columns missing: {sorted(missing)}")
        self._rows: dict[tuple[str, str, int | None], tuple[TextPrompt, ...]] = {}
        for key, group in frame.groupby(["scene_id", "kind", "phase"], dropna=False):
            scene, kind, phase_value = key
            phase = None if pd.isna(phase_value) else int(phase_value)
            prompts = tuple(
                TextPrompt(
                    mask_index=int(row.mask_index),
                    prompt_text=str(row.prompt_text),
                    source_catalog_id=str(row.source_catalog_id),
                    object_id=str(row.object_id),
                )
                for row in group.sort_values("mask_index").itertuples(index=False)
            )
            texts = [prompt.prompt_text for prompt in prompts]
            if len(texts) != len(set(texts)):
                raise RuntimeError(f"Duplicate prompt text in vocabulary cell {key!r}")
            self._rows[(str(scene), str(kind), phase)] = prompts

    def prompts_for(
        self, scene_id: str, kind: str, phase: int | None
    ) -> tuple[TextPrompt, ...]:
        key = (scene_id, kind, phase if kind == "mos" else None)
        prompts = self._rows.get(key)
        if not prompts:
            raise RuntimeError(f"No text-initialization vocabulary for {key!r}")
        return prompts
