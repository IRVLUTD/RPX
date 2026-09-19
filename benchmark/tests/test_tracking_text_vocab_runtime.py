from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import tracking_text_runtime  # noqa: E402
from tracking_text_runtime import TrackingTextVocabulary  # noqa: E402


def test_text_vocab_loads_mos_and_ego(tmp_path, monkeypatch) -> None:
    path = tmp_path / "vocab.parquet"
    pd.DataFrame(
        [
            {
                "scene_id": "scene001",
                "kind": "mos",
                "phase": 2,
                "mask_index": 4,
                "prompt_text": "black hammer",
                "source_catalog_id": "63.2",
                "object_id": "hammer.2",
            },
            {
                "scene_id": "scene001",
                "kind": "ego",
                "phase": None,
                "mask_index": 1,
                "prompt_text": "tan lion",
                "source_catalog_id": "82.2",
                "object_id": "lion.2",
            },
        ]
    ).to_parquet(path, index=False)
    monkeypatch.setattr(
        tracking_text_runtime,
        "TEXT_VOCAB_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    vocabulary = TrackingTextVocabulary(path)
    assert vocabulary.prompts_for("scene001", "mos", 2)[0].prompt_text == "black hammer"
    assert vocabulary.prompts_for("scene001", "ego", 0)[0].prompt_text == "tan lion"


def test_text_vocab_rejects_unpinned_bytes(tmp_path) -> None:
    path = tmp_path / "vocab.parquet"
    path.write_bytes(b"not the released parquet")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        TrackingTextVocabulary(path)
