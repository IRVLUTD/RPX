from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from rpx_benchmark.exceptions import ManifestError
from rpx_benchmark.vqa.contract import VQASample, image_locator, load_manifest, stable_sample_id
from rpx_benchmark.vqa.hub_rgb import image_cache_name
from rpx_benchmark.vqa.metrics import bbox_iou, score_predictions
from rpx_benchmark.vqa.outputs import ParsedOutput, normalize_label, parse_output
from rpx_benchmark.vqa.prompts import build_prompt
from rpx_benchmark.vqa.roster import MODELS, get_model


def row(question_type: str = "spatial_lr_binary") -> dict:
    value = {
        "scene_id": "scene001",
        "kind": "mos",
        "phase": 0,
        "frame": "00000",
        "img_w": 640,
        "img_h": 480,
        "type": question_type,
        "question": "Is the alarm_clock to the left of the doll?",
        "answer": "yes",
    }
    if question_type in {"spatial_lr_extreme", "depth_closest", "spatial_farthest"}:
        value.update(question="Which object is furthest to the left?", answer="alarm_clock")
        value["answer_bbox"] = [100, 120, 200, 220]
    if question_type == "attr_composition":
        value.update(
            question="What is the object made of metal that is used for dusting?",
            answer="air_duster_can",
        )
        value["answer_bbox"] = [100, 120, 200, 220]
    return value


def test_stable_id_and_portable_locator() -> None:
    value = row()
    assert stable_sample_id(value) == stable_sample_id(dict(reversed(list(value.items()))))
    assert image_locator(value) == {
        "repo_id": "IRVLUTD/RPX",
        "revision": "main",
        "shard": "scenes/scene001/0/rgb.tar",
        "member": "rgb/00000.webp",
    }
    ego = {**value, "kind": "ego", "phase": None}
    assert image_locator(ego)["revision"] == "ego-preview-v2"
    sample = VQASample.from_dict(value)
    changed_question = VQASample.from_dict({**value, "question": "A different question?"})
    assert sample.sample_id != changed_question.sample_id
    assert image_cache_name(sample) == image_cache_name(changed_question)


def test_contract_rejects_wrong_bbox_convention() -> None:
    value = row("depth_closest")
    value["answer_bbox"] = [0, 0, 640, 480]
    with pytest.raises(ManifestError, match="inclusive xyxy"):
        VQASample.from_dict(value)


def test_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    sample = VQASample.from_dict(row()).to_dict()
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(sample) + "\n" + json.dumps(sample) + "\n")
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(path)


def test_prompts_are_task_specific_and_single_image() -> None:
    binary = build_prompt(VQASample.from_dict(row()), "qwen2.5-vl-3b")
    assert "yes or no" in binary.text
    assert "alarm clock" in binary.text and "_" not in binary.text
    attribute = build_prompt(VQASample.from_dict(row("attr_composition")), "gemma3-12b")
    assert '"bbox"' in attribute.text
    bbox = build_prompt(VQASample.from_dict(row("depth_closest")), "qwen2.5-vl-3b")
    assert '"bbox"' in bbox.text and "640 by 480" in bbox.text
    paligemma = build_prompt(VQASample.from_dict(row("depth_closest")), "paligemma2-3b")
    assert paligemma.text.startswith("answer en ")
    assert paligemma.output_kind == "paligemma_two_stage"
    pali_binary = build_prompt(VQASample.from_dict(row()), "paligemma2-3b")
    assert pali_binary.text.startswith("answer en ")


def test_strict_output_parsers() -> None:
    binary = VQASample.from_dict(row())
    assert parse_output(binary, " YES. ", "qwen2.5-vl-3b").label == "yes"
    assert not parse_output(binary, "yes because it is", "qwen2.5-vl-3b").valid
    attribute = VQASample.from_dict(row("attr_composition"))
    assert normalize_label("Air_Duster_Can.") == "air duster can"
    parsed_attr = parse_output(attribute, '{"label":"air_duster_can","bbox":[100,120,200,220]}', "gemma3-12b")
    assert parsed_attr.label == "air duster can" and parsed_attr.valid
    bbox = VQASample.from_dict(row("depth_closest"))
    parsed_bbox = parse_output(
        bbox,
        '```json\n{"label":"alarm_clock","bbox":[100,120,200,220]}\n```',
        "qwen2.5-vl-3b",
    )
    assert parsed_bbox.valid and parsed_bbox.bbox == (100.0, 120.0, 200.0, 220.0)
    parsed_loc = parse_output(
        bbox, "<loc0256><loc0160><loc0469><loc0320> alarm_clock<eos>", "paligemma2-3b"
    )
    assert parsed_loc.valid
    assert parsed_loc.bbox == pytest.approx((100, 120, 200, 219.84375))
    assert parsed_loc.label == "alarm clock"


def test_metrics_oracle() -> None:
    samples = [
        VQASample.from_dict(row()),
        VQASample.from_dict(row("attr_composition")),
        VQASample.from_dict(row("depth_closest")),
    ]
    parsed = [
        ParsedOutput(True, label="yes"),
        ParsedOutput(True, label="air duster can", bbox=(100, 120, 200, 220)),
        ParsedOutput(True, label="alarm clock", bbox=(100, 120, 200, 220)),
    ]
    result = score_predictions(zip(samples, parsed, strict=True))
    assert result["parse_rate"] == 1.0
    assert result["binary_accuracy"] == 1.0
    assert result["attribute_exact_match"] == 0.0
    assert result["bbox_accuracy_at_0_5"] == 1.0
    assert bbox_iou((100, 120, 200, 220), (100, 120, 200, 220)) == 1.0


def test_roster_matches_rpx_draft() -> None:
    assert [model.display_name for model in MODELS] == [
        "PaliGemma 2 3B",
        "Qwen2.5-VL 3B",
        "Gemma 3 4B",
        "Phi-3.5-Vision 4.2B",
        "LLaVA-OneVision 7B",
        "Qwen2.5-VL 7B",
        "Idefics3 8B",
        "InternVL 2.5 8B",
        "PaliGemma 2 10B",
        "Gemma 3 12B",
    ]
    assert get_model("gemma3-12b").capabilities == {"bbox"}


@pytest.mark.skip(reason="bbox smoke manifest is generated from the pinned Hub revision")
def test_committed_smoke_manifest_has_required_balance() -> None:
    manifest = Path(__file__).parents[1] / "data" / "vqa_smoke" / "v1" / "manifest.jsonl"
    samples = load_manifest(manifest)
    counts = Counter(
        f"{sample.question_type}:{sample.answer}"
        if sample.question_type in {"spatial_lr_binary", "spatial_ud_binary"}
        else f"{sample.question_type}:{sample.kind}"
        if sample.question_type == "attr_composition"
        else sample.question_type
        for sample in samples
    )
    assert counts == {
        "spatial_lr_binary:yes": 4,
        "spatial_lr_binary:no": 4,
        "spatial_ud_binary:yes": 4,
        "spatial_ud_binary:no": 4,
        "spatial_lr_extreme": 4,
        "depth_closest": 4,
        "spatial_farthest": 4,
        "attr_composition:mos": 4,
        "attr_composition:ego": 4,
    }
    assert {sample.phase for sample in samples if sample.kind == "mos"} == {0, 1, 2}
    assert len({json.dumps(sample.image, sort_keys=True) for sample in samples}) == 5


def test_docker_matrix_matches_python_roster() -> None:
    path = Path(__file__).parents[2] / "docker" / "vqa-smoke" / "model-matrix.json"
    matrix = json.loads(path.read_text())
    assert matrix["single_image_only"] is True
    assert [model["key"] for model in matrix["models"]] == [model.key for model in MODELS]
    assert [set(model["tasks"]) for model in matrix["models"]] == [
        set(model.capabilities) for model in MODELS
    ]
