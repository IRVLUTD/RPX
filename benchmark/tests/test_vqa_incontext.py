"""Tests for the two-image (in-context) VQA extension: contract, shard-path
normalization, hub_rgb cache-key/URL logic, canonicalization, deterministic
sampling, and the PaliGemma side-by-side composite. Network-dependent
fetch/verification is exercised separately (see hub_rgb.fetch_reference_crop
docstring); these tests stay offline and deterministic like the rest of the
suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

from rpx_benchmark.exceptions import ManifestError
from rpx_benchmark.vqa.canonical import canon_color, canon_function, semantic_key
from rpx_benchmark.vqa.contract import (
    IN_CONTEXT_REVISION,
    VQASample,
    normalize_shard,
)
from rpx_benchmark.vqa.hub_rgb import hub_url, image_cache_name, reference_cache_name
from rpx_benchmark.vqa.outputs import parse_output
from rpx_benchmark.vqa.sampling import DeterministicSelector, is_centered

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vqa_models.vllm_backend import VLLMVQARunner  # noqa: E402


def incontext_row(**overrides) -> dict:
    value = {
        "scene_id": "scene001",
        "kind": "mos",
        "phase": 0.0,
        "frame": "00000",
        "img_w": 640,
        "img_h": 480,
        "type": "inctx_attr_single_color",
        "question": (
            "Which object in Image 2 has the same color as the object shown in "
            "Image 1? What is its bounding box in Image 2?"
        ),
        "answer": "lion",
        "answer_bbox": [308, 226, 400, 377],
        "schema_version": "incontext_v1",
        "sample_id": "edd102d6fa0c1bd9458619f7",
        "target_image": {
            "member": "rgb/00000.webp",
            "repo_id": "IRVLUTD/RPX",
            "revision": IN_CONTEXT_REVISION,
            "shard": "scenes/scene001/0.0/rgb.tar",  # the known float-phase defect
        },
        "reference_image": {
            "crop_bbox": [318, 222, 434, 274],
            "member": "rgb/00060.webp",
            "repo_id": "IRVLUTD/RPX",
            "revision": IN_CONTEXT_REVISION,
            "shard": "objects/marker.5/0/rgb.tar",
        },
        "reference_crop_bbox": [318, 222, 434, 274],
        "reference_crop_sha256": "174f17cb5d644439f3ea95825c06a44cd7c82ce04f785d43b04301d63ab9968f",
        "source_sample_id": "99abfd5ef7f47a1aaa22cdd2",
    }
    value.update(overrides)
    return value


# ── contract ────────────────────────────────────────────────────────────────


def test_incontext_sample_parses_and_validates():
    sample = VQASample.from_dict(incontext_row())
    assert sample.is_in_context
    assert sample.reference_image is not None
    assert sample.images == (sample.reference_image, sample.image)
    assert sample.image["revision"] == IN_CONTEXT_REVISION


def test_incontext_answer_bbox_checked_against_target_not_reference_dims():
    # img_w/img_h are the TARGET's (Image 2) dimensions -- the answer bbox
    # must validate against those even though the reference (Image 1) frame
    # has entirely different pixel dimensions.
    row = incontext_row(answer_bbox=[0, 0, 639, 479])
    sample = VQASample.from_dict(row)
    assert sample.answer_bbox == (0, 0, 639, 479)
    bad = incontext_row(answer_bbox=[0, 0, 640, 480])  # one past the inclusive bound
    with pytest.raises(ManifestError, match="inclusive xyxy"):
        VQASample.from_dict(bad)


def test_incontext_missing_reference_image_rejected():
    row = incontext_row()
    del row["reference_image"]
    with pytest.raises(ManifestError, match="reference_image"):
        VQASample.from_dict(row)


def test_incontext_mutable_revision_rejected():
    row = incontext_row()
    row["target_image"] = {**row["target_image"], "revision": "main"}
    with pytest.raises(ManifestError, match="immutable revision"):
        VQASample.from_dict(row)


def test_incontext_locator_not_recomputed_from_scene_kind_phase_frame():
    # image_locator() would produce scenes/scene001/0/rgb.tar (no float bug,
    # revision "main") -- the contract must NOT recompute/replace the stored
    # in-context locator with that, per item 2 of the task.
    sample = VQASample.from_dict(incontext_row())
    assert sample.image["revision"] == IN_CONTEXT_REVISION
    assert sample.image["shard"] == "scenes/scene001/0.0/rgb.tar"


def test_incontext_manifest_round_trips_through_write_and_load(tmp_path):
    from rpx_benchmark.vqa.contract import load_manifest, write_manifest

    sample = VQASample.from_dict(incontext_row())
    path = tmp_path / "manifest.jsonl"
    write_manifest([sample], path)
    (reloaded,) = load_manifest(path)
    assert reloaded.image == sample.image
    assert reloaded.reference_image == sample.reference_image
    assert reloaded.reference_crop_bbox == sample.reference_crop_bbox
    assert reloaded.reference_crop_sha256 == sample.reference_crop_sha256


def test_normal_sample_backward_compatible_after_incontext_extension():
    row = {
        "scene_id": "scene001",
        "kind": "mos",
        "phase": 0,
        "frame": "00000",
        "img_w": 640,
        "img_h": 480,
        "type": "attr_odd_one_out",
        "question": "Which object is NOT made of metal?",
        "answer": "air_duster_can",
        "answer_bbox": [100, 120, 200, 220],
    }
    sample = VQASample.from_dict(row)
    assert not sample.is_in_context
    assert sample.images == (sample.image,)
    assert sample.reference_image is None
    assert sample.image == {
        "repo_id": "IRVLUTD/RPX",
        "revision": "main",
        "shard": "scenes/scene001/0/rgb.tar",
        "member": "rgb/00000.webp",
    }


# ── shard-path normalization (the discovered float-phase bug) ──────────────


def test_normalize_shard_fixes_known_float_phase_defect():
    assert normalize_shard("scenes/scene001/0.0/rgb.tar") == "scenes/scene001/0/rgb.tar"
    assert normalize_shard("scenes/scene001/1.0/rgb.tar") == "scenes/scene001/1/rgb.tar"
    assert normalize_shard("scenes/scene001/2.0/rgb.tar") == "scenes/scene001/2/rgb.tar"


def test_normalize_shard_is_a_no_op_on_already_correct_paths():
    assert normalize_shard("scenes/scene001/0/rgb.tar") == "scenes/scene001/0/rgb.tar"
    assert normalize_shard("objects/marker.5/0/rgb.tar") == "objects/marker.5/0/rgb.tar"


def test_normalize_shard_does_not_touch_unrelated_numbers():
    # Only a bare "N.0" path segment is a phase artifact; anything else
    # (object ids like "marker.5", version dirs like "v1.0") must survive.
    assert normalize_shard("objects/marker.5/0/rgb.tar") == "objects/marker.5/0/rgb.tar"


def test_hub_url_applies_shard_normalization():
    sample = VQASample.from_dict(incontext_row())
    url = hub_url(sample.image)
    assert "/scenes/scene001/0/rgb.tar" in url
    assert "0.0" not in url
    assert url.startswith(f"https://huggingface.co/datasets/IRVLUTD/RPX/resolve/{IN_CONTEXT_REVISION}/")


# ── hub_rgb cache keys ───────────────────────────────────────────────────────


def test_image_and_reference_cache_names_are_distinct_and_stable():
    sample = VQASample.from_dict(incontext_row())
    a = image_cache_name(sample)
    b = reference_cache_name(sample)
    assert a != b
    assert a == image_cache_name(VQASample.from_dict(incontext_row()))
    assert b == reference_cache_name(VQASample.from_dict(incontext_row()))


def test_reference_cache_name_changes_with_crop_bbox():
    sample_a = VQASample.from_dict(incontext_row())
    sample_b = VQASample.from_dict(incontext_row(reference_crop_bbox=[0, 0, 10, 10]))
    assert reference_cache_name(sample_a) != reference_cache_name(sample_b)


# ── bbox validation against Image 2 dims (processor-resize/out-of-bounds regression) ──


def test_incontext_bbox_out_of_bounds_rejected_like_normal_samples():
    sample = VQASample.from_dict(incontext_row())
    # A bbox outside the frozen normalized range must be rejected for Image 2.
    parsed = parse_output(
        sample,
        '{"label":"lion","bbox":[300,200,9999,9999]}',
        "qwen2.5-vl-3b",
    )
    assert not parsed.valid
    assert "outside normalized 0-1000" in parsed.error


def test_incontext_valid_bbox_within_target_dims_accepted():
    sample = VQASample.from_dict(incontext_row())
    parsed = parse_output(sample, '{"label":"lion","bbox":[308,226,400,377]}', "qwen2.5-vl-3b")
    assert parsed.valid
    assert parsed.bbox == pytest.approx((196.812, 108.254, 255.6, 180.583))


# ── canonicalization ─────────────────────────────────────────────────────────


def test_canon_function_aliases():
    assert canon_function("to drink") == "drink"
    assert canon_function("drink") == "drink"
    assert canon_function("to eat") == "eat"
    assert canon_function("to snack") == "snack"
    assert canon_function("clean") == "clean"


def test_canon_color_gray_grey():
    assert canon_color("gray") == "grey"
    assert canon_color("grey") == "grey"
    assert canon_color("red") == "red"


def test_semantic_key_collapses_function_alias_duplicate():
    base = dict(
        scene_id="scene001",
        kind="mos",
        phase=0,
        frame="00000",
        target_source_catalog_id="1.1",
        reference_source_catalog_id="2.2",
    )
    row_a = {**base, "type": "inctx_attr_single_function", "attribute_value": "drink"}
    row_b = {**base, "type": "inctx_attr_single_function", "attribute_value": "to drink"}
    assert semantic_key(row_a) == semantic_key(row_b)
    row_c = {**base, "type": "inctx_attr_single_function", "attribute_value": "clean"}
    assert semantic_key(row_a) != semantic_key(row_c)


def test_semantic_key_differs_across_frames():
    row_a = {
        "scene_id": "scene001", "kind": "mos", "phase": 0, "frame": "00000",
        "type": "inctx_attr_single_function", "attribute_value": "drink",
        "target_source_catalog_id": "1.1", "reference_source_catalog_id": "2.2",
    }
    row_b = {**row_a, "frame": "00001"}
    assert semantic_key(row_a) != semantic_key(row_b)


# ── deterministic sampling ───────────────────────────────────────────────────


def test_is_centered_uses_precomputed_flag_when_present():
    assert is_centered({"answer_is_centered": True}) is True
    assert is_centered({"answer_is_centered": False}) is False


def test_is_centered_computes_from_bbox_when_flag_absent():
    centered_row = {"answer_bbox": [280, 200, 360, 280], "img_w": 640, "img_h": 480}
    assert is_centered(centered_row) is True
    corner_row = {"answer_bbox": [0, 0, 10, 10], "img_w": 640, "img_h": 480}
    assert is_centered(corner_row) is False


def test_deterministic_selector_is_reproducible_and_respects_quota():
    def make_rows(n):
        return [
            {"scene_id": f"s{i}", "kind": "mos", "phase": 0, "frame": f"{i:05d}", "type": "t",
             "question": f"q{i}", "sample_id": f"id{i}", "answer": "a"}
            for i in range(n)
        ]

    cells = {"cell_a": make_rows(5), "cell_b": make_rows(5)}
    first = DeterministicSelector(seed=20260908).select(cells, lambda _c: 2)
    second = DeterministicSelector(seed=20260908).select(cells, lambda _c: 2)
    assert first == second
    assert all(len(v) == 2 for v in first.values())


def test_deterministic_selector_prefers_new_scene_frame():
    rows = [
        {"scene_id": "shared", "kind": "mos", "phase": 0, "frame": "00000", "type": "t",
         "question": "q0", "sample_id": "id0", "answer": "a"},
        {"scene_id": "shared", "kind": "mos", "phase": 0, "frame": "00000", "type": "t",
         "question": "q1", "sample_id": "id1", "answer": "a"},
        {"scene_id": "unique", "kind": "mos", "phase": 0, "frame": "00001", "type": "t",
         "question": "q2", "sample_id": "id2", "answer": "a"},
    ]
    selector = DeterministicSelector(seed=1)
    selector.used_scene_frame.add(("shared", "00000"))
    picked = selector.select_cell(rows, quota=1)
    assert picked[0]["scene_id"] == "unique"


# ── PaliGemma side-by-side composite (pure PIL logic, no vLLM/CUDA needed) ──


def test_paligemma_side_by_side_composite_shape():
    left = Image.new("RGB", (100, 50), (255, 0, 0))
    right = Image.new("RGB", (80, 80), (0, 255, 0))
    composite = VLLMVQARunner._side_by_side([left, right])
    assert composite.height == 80
    assert composite.width > left.width + right.width - 40  # scaled + gap, not naive sum


def test_paligemma_empty_stage1_is_retained_as_model_output(monkeypatch):
    runner = object.__new__(VLLMVQARunner)
    runner._last_adapter_metadata = {}
    monkeypatch.setattr(runner, "_generate_paligemma", lambda *args, **kwargs: "")
    image = Image.new("RGB", (32, 32))
    assert runner._predict_paligemma([image], "answer en Which object?\n", 24) == ""
    assert runner.prediction_metadata() == {
        "adapter": "paligemma_two_stage",
        "stage1_raw_output": "",
        "stage1_label": None,
        "stage2_raw_output": None,
        "stage2_skipped_reason": "empty_stage1_label",
    }


def test_paligemma_both_stages_are_retained(monkeypatch):
    runner = object.__new__(VLLMVQARunner)
    runner._last_adapter_metadata = {}
    outputs = iter(["shoe", "<loc0001><loc0002><loc0003><loc0004> shoe"])
    monkeypatch.setattr(runner, "_generate_paligemma", lambda *args, **kwargs: next(outputs))
    image = Image.new("RGB", (32, 32))
    raw = runner._predict_paligemma([image], "answer en Which object?\n", 24)
    assert raw.startswith("<loc0001>")
    assert runner.prediction_metadata()["stage1_label"] == "shoe"
    assert runner.prediction_metadata()["stage2_raw_output"] == raw
