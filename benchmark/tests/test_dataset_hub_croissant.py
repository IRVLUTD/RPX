"""Tests for ``rpx_benchmark.dataset_hub.croissant``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.croissant import (
    _LAYOUT_NOTE,
    CroissantPatch,
    stage_croissant,
)
from rpx_benchmark.exceptions import ConfigError, DatasetError


@pytest.fixture
def fake_src(tmp_path: Path) -> Path:
    """Minimal Croissant JSON we can patch in tests."""
    src = tmp_path / "rpx_croissant.json"
    src.write_text(
        json.dumps(
            {
                "@type": "sc:Dataset",
                "name": "RPX",
                "description": "An RGB-D benchmark.",
                "url": "https://anonymous.4open.science/r/RPX",
                "version": "0.0.0",
                "citeAs": "TODO bibtex",
            }
        ),
        encoding="utf-8",
    )
    return src


def test_stage_croissant_writes_patched_json(tmp_path: Path, fake_src: Path):
    out = stage_croissant(
        tmp_path / "stage", src=fake_src, patch=CroissantPatch(repo_id="acme/RPX")
    )
    assert out == tmp_path / "stage" / "rpx_croissant.json"
    payload = json.loads(out.read_text("utf-8"))
    assert payload["url"] == "https://huggingface.co/datasets/acme/RPX"
    assert payload["version"] == "1.0.0"


def test_stage_croissant_appends_layout_note(tmp_path: Path, fake_src: Path):
    out = stage_croissant(tmp_path / "stage", src=fake_src)
    payload = json.loads(out.read_text("utf-8"))
    assert _LAYOUT_NOTE.strip() in payload["description"]


def test_stage_croissant_skip_layout_note(tmp_path: Path, fake_src: Path):
    out = stage_croissant(
        tmp_path / "stage",
        src=fake_src,
        patch=CroissantPatch(annotate_layout=False),
    )
    payload = json.loads(out.read_text("utf-8"))
    assert _LAYOUT_NOTE.strip() not in payload["description"]


def test_stage_croissant_replaces_citation_when_provided(tmp_path: Path, fake_src: Path):
    out = stage_croissant(
        tmp_path / "stage",
        src=fake_src,
        patch=CroissantPatch(cite_as="@misc{rpx2026, ...}"),
    )
    payload = json.loads(out.read_text("utf-8"))
    assert payload["citeAs"] == "@misc{rpx2026, ...}"


def test_stage_croissant_keeps_existing_citation_when_not_overridden(
    tmp_path: Path,
    fake_src: Path,
):
    out = stage_croissant(tmp_path / "stage", src=fake_src, patch=CroissantPatch())
    payload = json.loads(out.read_text("utf-8"))
    assert payload["citeAs"] == "TODO bibtex"


def test_stage_croissant_idempotent_when_re_appended(tmp_path: Path, fake_src: Path):
    """Re-staging should not duplicate the layout note."""
    stage_croissant(tmp_path / "stage", src=fake_src)
    out = stage_croissant(tmp_path / "stage", src=fake_src, overwrite=True)
    payload = json.loads(out.read_text("utf-8"))
    # Note appears once — not twice.
    assert payload["description"].count("Note on on-disk shape:") == 1


def test_stage_croissant_missing_src_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="does not exist"):
        stage_croissant(tmp_path / "stage", src=tmp_path / "nope.json")


def test_stage_croissant_refuses_overwrite(tmp_path: Path, fake_src: Path):
    stage_croissant(tmp_path / "stage", src=fake_src)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        stage_croissant(tmp_path / "stage", src=fake_src)


def test_default_src_resolves_under_paper_submission_dir():
    from rpx_benchmark.dataset_hub.croissant import _default_croissant_src

    default = _default_croissant_src()
    assert default.name == "rpx_croissant.json"
    assert default.parent.name == "croissant"
    assert "paper-submission" in default.parts
