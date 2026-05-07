"""Tests for the objects_meta/ layer (questionnaire dedup).

Covers:

* mock generator emits ``<sos_scene>/questionnaire.txt`` and
  cross-references SOS object names from MOS ``mask_to_object.json``;
* ``_parse_questionnaire`` correctly extracts Q→[answer] pairs;
* ``pack_objects_meta`` writes one ``objects_meta/<obj>/questionnaire.json``
  per SOS scene plus a ``_index.json`` listing all object IDs;
* ``download_for_task`` adds ``objects_meta/*/questionnaire.json`` to
  the allow_patterns when the recipe references ``questionnaire``;
* recipes that don't reference questionnaire don't pull objects_meta.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pa = pytest.importorskip("pyarrow")

from rpx_benchmark.dataset_hub import downloader
from rpx_benchmark.dataset_hub.downloader import download_for_task
from rpx_benchmark.dataset_hub.manifest import build_frame_manifest
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import (
    PackPlan,
    _parse_questionnaire,
    pack_capture_tree,
    pack_objects_meta,
)
from rpx_benchmark.dataset_hub.recipes import SceneType
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.exceptions import DatasetError

# --------------------------------------------------------------------- #
# Mock + parser
# --------------------------------------------------------------------- #


def test_mock_writes_sos_questionnaire(tmp_path: Path):
    out = generate_mock(
        tmp_path / "ds",
        MockSpec(
            multi_object_scenes=0,
            single_object_scenes=2,
            phases_per_multi=0,
            frames_per_phase=1,
        ),
    )
    qs = sorted(out.glob("sos/*/questionnaire.txt"))
    assert len(qs) == 2
    text = qs[0].read_text(encoding="utf-8")
    assert "What is the name of the object" in text


def test_mock_mos_mask_to_object_uses_sos_names(tmp_path: Path):
    out = generate_mock(
        tmp_path / "ds",
        MockSpec(
            multi_object_scenes=2,
            single_object_scenes=3,
            phases_per_multi=1,
            frames_per_phase=1,
        ),
    )
    sos_names = {p.name for p in (out / "sos").iterdir()}
    sample = json.loads(
        (out / "mos/scene1/0/sam2/mask_to_object.json").read_text("utf-8"),
    )
    referenced = set(sample.values())
    assert referenced.issubset(sos_names)


def test_parse_questionnaire_extracts_questions_and_answers():
    text = """1. What is the name of the object in these images?
tape, holder, support

2. What is the category of the object in these images?
stationery, adhesive
"""
    parsed = _parse_questionnaire(text)
    assert len(parsed) == 2
    assert parsed["What is the name of the object in these images?"] == [
        "tape",
        "holder",
        "support",
    ]
    assert parsed["What is the category of the object in these images?"] == [
        "stationery",
        "adhesive",
    ]


def test_parse_questionnaire_ignores_comments():
    text = """# this is a comment
1. What is the name?
foo
"""
    parsed = _parse_questionnaire(text)
    assert parsed == {"What is the name?": ["foo"]}


# --------------------------------------------------------------------- #
# pack_objects_meta
# --------------------------------------------------------------------- #


@pytest.fixture
def packed_with_meta(tmp_path: Path):
    src = generate_mock(
        tmp_path / "src",
        MockSpec(
            multi_object_scenes=2,
            single_object_scenes=4,
            phases_per_multi=1,
            frames_per_phase=2,
        ),
    )
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack_capture_tree(PackPlan(src_root=src, staging_root=staging), scan)
    artefacts = pack_objects_meta(
        PackPlan(src_root=src, staging_root=staging),
        scan,
    )
    return src, staging, scan, artefacts


def test_pack_objects_meta_writes_one_per_sos_scene(packed_with_meta):
    src, staging, scan, artefacts = packed_with_meta
    sos_count = sum(1 for s in scan.scenes if s.scene_type is SceneType.SINGLE_OBJECT)
    assert len(artefacts) == sos_count
    # Files actually exist on disk.
    for a in artefacts:
        assert (staging / a.repo_path).is_file()


def test_objects_meta_index_lists_all_object_ids(packed_with_meta):
    src, staging, scan, artefacts = packed_with_meta
    index = json.loads(
        (staging / "objects_meta" / "_index.json").read_text("utf-8"),
    )
    expected = sorted(s.scene_id for s in scan.scenes if s.scene_type is SceneType.SINGLE_OBJECT)
    assert index["object_ids"] == expected


def test_each_questionnaire_json_has_object_id_and_questions(packed_with_meta):
    src, staging, scan, artefacts = packed_with_meta
    sample = artefacts[0]
    payload = json.loads((staging / sample.repo_path).read_text("utf-8"))
    assert payload["object_id"] == sample.object_id
    assert "questions" in payload
    assert len(payload["questions"]) == 5  # FewSOL has 5 standard questions


def test_pack_objects_meta_refuses_overwrite(tmp_path: Path):
    src = generate_mock(
        tmp_path / "src",
        MockSpec(
            multi_object_scenes=0,
            single_object_scenes=1,
            phases_per_multi=0,
            frames_per_phase=1,
        ),
    )
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    plan = PackPlan(src_root=src, staging_root=staging)
    pack_objects_meta(plan, scan)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        pack_objects_meta(plan, scan)


# --------------------------------------------------------------------- #
# Downloader integration
# --------------------------------------------------------------------- #


@pytest.fixture
def staging_with_meta(tmp_path: Path) -> Path:
    src = generate_mock(
        tmp_path / "src",
        MockSpec(
            multi_object_scenes=3,
            single_object_scenes=2,
            phases_per_multi=3,
            frames_per_phase=2,
        ),
    )
    out = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack = pack_capture_tree(PackPlan(src_root=src, staging_root=out), scan)
    pack_objects_meta(PackPlan(src_root=src, staging_root=out), scan)

    multi_ids = [s.scene_id for s in scan.scenes if s.scene_type is SceneType.MULTI_OBJECT]
    splits = {multi_ids[0]: "easy", multi_ids[1]: "medium", multi_ids[2]: "hard"}
    build_frame_manifest(scan, pack, out, splits=splits)
    return out


def _patch_hf(monkeypatch, staging: Path) -> MagicMock:
    fake = MagicMock()
    fake.hf_hub_download = lambda repo_id, repo_type, filename, **_: str(
        staging / filename,
    )
    fake.snapshot_download = lambda **_: str(staging)
    monkeypatch.setattr(downloader, "_hub", lambda: fake)
    return fake


def test_vqa_recipe_pulls_objects_meta(staging_with_meta, monkeypatch):
    _patch_hf(monkeypatch, staging_with_meta)
    res = download_for_task(task="vqa", split="easy", repo_id="acme/RPX")
    assert any("objects_meta/*/questionnaire.json" in p for p in res.allow_patterns)


def test_object_templates_recipe_pulls_no_objects_meta(staging_with_meta, monkeypatch):
    """object_templates is RGB+masks only, no questionnaire."""
    _patch_hf(monkeypatch, staging_with_meta)
    res = download_for_task(task="object_templates", split=None, repo_id="acme/RPX")
    assert not any("objects_meta/" in p for p in res.allow_patterns)


def test_segmentation_recipe_does_not_pull_objects_meta(staging_with_meta, monkeypatch):
    """Segmentation = rgb + masks only; questionnaire isn't part of it."""
    _patch_hf(monkeypatch, staging_with_meta)
    res = download_for_task(task="segmentation", split="easy", repo_id="acme/RPX")
    assert not any("objects_meta/" in p for p in res.allow_patterns)
