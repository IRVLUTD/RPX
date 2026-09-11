from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpx_benchmark.exceptions import ManifestError
from rpx_benchmark.vqa import hub_rgb
from rpx_benchmark.vqa.contract import VQASample, image_locator, load_manifest, stable_sample_id
from rpx_benchmark.vqa.hub_rgb import image_cache_name
from rpx_benchmark.vqa.metrics import bbox_iou, label_token_f1, score_predictions
from rpx_benchmark.vqa.outputs import (
    ParsedOutput,
    normalize_label,
    parse_output,
)
from rpx_benchmark.vqa.prompts import (
    build_label_localization_prompt,
    build_oracle_localization_prompt,
    build_prompt,
    build_semantic_diagnostic_prompt,
    native_referring_expression,
)
from rpx_benchmark.vqa.roster import MODELS, get_model

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_vqa_acceptance_gallery import _native_candidates  # noqa: E402
from run_vqa_diagnostic import diagnostic_bbox  # noqa: E402
from run_vllm_vqa import validate_gpu_environment  # noqa: E402
from vqa_models.backend_registry import backend_name  # noqa: E402
from vqa_models.florence_backend import FlorenceVQARunner, ImageGeometry  # noqa: E402
from vqa_models.internvl_backend import InternVLVQARunner, _dynamic_preprocess  # noqa: E402
from vqa_models.paligemma_backend import PaliGemmaVQARunner  # noqa: E402
from vqa_models.vllm_backend import (  # noqa: E402
    CHECKPOINTS as VLLM_CHECKPOINTS,
)
from vqa_models.vllm_backend import (
    VLLMCheckpoint,
    VLLMVQARunner,
)


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


def test_bulk_prefetch_can_inventory_missing_members_without_retrying_rows(
    tmp_path: Path, monkeypatch
) -> None:
    sample = VQASample.from_dict(row("depth_closest"))

    class EmptyArchive:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(hub_rgb, "HTTPRangeReader", lambda _url: object())
    monkeypatch.setattr(hub_rgb.tarfile, "open", lambda **_kwargs: EmptyArchive())
    missing: list[dict[str, str]] = []
    paths = hub_rgb.fetch_images_many([sample], tmp_path, missing_report=missing)
    assert paths == {}
    assert missing == [
        {
            "sample_id": sample.sample_id,
            "role": "rgb",
            "url": hub_rgb.hub_url(sample.image),
            "member": "rgb/00000.webp",
        }
    ]


def test_prompts_are_task_specific_and_single_image() -> None:
    binary = build_prompt(VQASample.from_dict(row()), "qwen2.5-vl-3b")
    assert "yes or no" in binary.text
    assert "alarm clock" in binary.text and "_" not in binary.text
    attribute = build_prompt(VQASample.from_dict(row("attr_composition")), "gemma4-12b")
    assert "Identify and localize" in attribute.text
    assert "target image" in attribute.text
    assert attribute.output_kind == "bbox_json_normalized_1000"
    bbox = build_prompt(VQASample.from_dict(row("depth_closest")), "qwen2.5-vl-3b")
    assert "Identify and localize" in bbox.text
    assert bbox.text == attribute.text.replace(
        "What is the object made of metal that is used for dusting?",
        "Which object is furthest to the left?",
    )
    assert bbox.output_kind == "bbox_json_normalized_1000"
    pali_bbox = build_prompt(VQASample.from_dict(row("depth_closest")), "paligemma2-3b")
    assert pali_bbox.text == "detect the object furthest to the left\n"
    assert pali_bbox.output_kind == "bbox_native_question_grounding"


def test_native_referring_expression_preserves_incontext_relation_without_gt() -> None:
    sample = SimpleNamespace(
        question_type="inctx_attr_single_material",
        question=(
            "Which object in Image 2 is made of the same material as the object "
            "shown in Image 1? What is its bounding box in Image 2?"
        ),
    )
    assert native_referring_expression(sample) == (
        "the object in image 2 that is made of the same material as the object shown in Image 1"
    )
    pali_binary = build_prompt(VQASample.from_dict(row()), "paligemma2-3b")
    assert pali_binary.text.startswith("answer en ")


def test_json_bbox_instruction_is_identical_across_models() -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    keys = (
        "gemma4-12b",
        "internvl2.5-8b",
        "idefics3-8b",
        "qwen2.5-vl-7b",
        "qwen3-vl-8b",
        "qwen3-vl-2b",
        "llava-onevision-7b",
        "phi-3.5-vision-4b",
    )
    prompts = [build_prompt(sample, key) for key in keys]
    assert len({prompt.text for prompt in prompts}) == 1
    assert {prompt.output_kind for prompt in prompts} == {"bbox_json_normalized_1000"}


@pytest.mark.parametrize("model_key", ["internvl3.5-1b", "internvl3.5-14b"])
def test_internvl_requests_native_grounding_bbox(model_key: str) -> None:
    prompt = build_prompt(VQASample.from_dict(row("depth_closest")), model_key)
    assert prompt.output_kind == "bbox_native_internvl_grounding"
    assert prompt.text == (
        "Please provide the bounding box coordinate of the region this sentence "
        "describes: the object furthest to the left"
    )


@pytest.mark.parametrize("model_key", ["florence2-base", "florence2-large"])
def test_florence_uses_direct_scored_question_grounding(model_key: str) -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    prompt = build_prompt(sample, model_key)
    assert prompt.text == "the object furthest to the left"
    assert prompt.output_kind == "bbox_native_question_grounding"
    model = get_model(model_key)
    assert model.capabilities == {"bbox"}
    assert model.diagnostic_capabilities == {"semantic_label", "phrase_grounding"}


@pytest.mark.parametrize("model_key", ["paligemma2-3b", "paligemma2-10b"])
def test_paligemma_uses_direct_scored_question_grounding(model_key: str) -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    prompt = build_prompt(sample, model_key)
    assert prompt.text == "detect the object furthest to the left\n"
    assert prompt.output_kind == "bbox_native_question_grounding"
    model = get_model(model_key)
    assert model.capabilities == {"bbox"}
    assert model.diagnostic_capabilities == {"semantic_label", "object_detection"}


@pytest.mark.parametrize("model_key", ["deepseek-vl2-tiny", "deepseek-vl2"])
def test_deepseek_uses_native_one_stage_grounding(model_key: str) -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    prompt = build_prompt(sample, model_key)
    assert prompt.text == "<|ref|>the object furthest to the left<|/ref|>."
    assert prompt.output_kind == "bbox_native_question_grounding"
    assert get_model(model_key).capabilities == {"bbox"}


@pytest.mark.parametrize("model_key", ["deepseek-vl2-tiny", "deepseek-vl2"])
def test_deepseek_incontext_uses_native_grounding_mode(model_key: str) -> None:
    sample = SimpleNamespace(
        question_type="inctx_attr_single_material",
        question=(
            "Which object in Image 2 is made of the same material as the object "
            "shown in Image 1? What is its bounding box in Image 2?"
        ),
        is_in_context=True,
    )
    prompt = build_prompt(sample, model_key)
    assert prompt.text.startswith("<|grounding|>")
    assert "first image is a crop showing the reference object" in prompt.text
    assert "second image" in prompt.text
    assert "<|ref|>" not in prompt.text
    assert prompt.output_kind == "bbox_native_question_grounding"


def test_diagnostic_prompts_separate_semantics_from_oracle_localization() -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    semantic = build_semantic_diagnostic_prompt(sample, "gemma4-12b")
    assert sample.answer not in semantic.text
    assert semantic.output_kind == "diagnostic_semantic_label"
    oracle = build_oracle_localization_prompt(sample, "gemma4-12b")
    assert "alarm clock" in oracle.text
    assert "target image" in oracle.text
    assert oracle.output_kind == "diagnostic_oracle_bbox"
    pali_oracle = build_oracle_localization_prompt(sample, "paligemma2-10b")
    assert pali_oracle.text == "detect alarm clock\n"
    pali_semantic = build_semantic_diagnostic_prompt(sample, "paligemma2-10b")
    assert pali_semantic.text == "answer en Which object is furthest to the left?\n"
    florence_semantic = build_semantic_diagnostic_prompt(sample, "florence2-large")
    assert florence_semantic.text == "Which object is furthest to the left?"
    predicted = build_label_localization_prompt("red alarm clock", "gemma4-12b")
    assert '"red alarm clock"' in predicted.text
    assert predicted.output_kind == "diagnostic_predicted_label_bbox"
    pali_predicted = build_label_localization_prompt("red alarm clock", "paligemma2-10b")
    assert pali_predicted.text == "detect red alarm clock\n"
    florence_oracle = build_oracle_localization_prompt(sample, "florence2-large")
    assert florence_oracle.text == "alarm clock"
    assert "JSON" not in florence_oracle.text
    florence_predicted = build_label_localization_prompt("red alarm clock", "florence2-base")
    assert florence_predicted.text == "red alarm clock"


def test_direct_bbox_adapter_is_one_scored_call(monkeypatch) -> None:
    runner = object.__new__(VLLMVQARunner)
    runner._last_adapter_metadata = {}
    calls = []
    monkeypatch.setattr(
        runner,
        "_chat_generate",
        lambda paths, prompt, tokens: calls.append((paths, prompt, tokens)) or '{"bbox":[1,2,3,4]}',
    )
    paths = [Path("reference.png"), Path("target.png")]
    assert runner._predict_json_direct(paths, "question", 96) == '{"bbox":[1,2,3,4]}'
    assert calls == [(paths, "question", 96)]
    assert runner.prediction_metadata() == {
        "adapter": "direct_bbox_json",
        "single_scored_model_call": True,
    }


def test_gallery_exposes_rejected_direct_native_candidates() -> None:
    raw = json.dumps(
        {
            "error": "expected exactly one target-region candidate",
            "candidates": [
                {"label": "shoe", "bbox": [10, 20, 30, 40]},
                {"label": "boot", "bbox": [50, 60, 70, 80]},
            ],
        }
    )
    prediction = {"adapter_metadata": {"adapter": "direct_native_question_grounding"}}
    assert _native_candidates(raw, prediction) == [
        {"label": "shoe", "bbox": [10, 20, 30, 40]},
        {"label": "boot", "bbox": [50, 60, 70, 80]},
    ]


def test_oracle_diagnostic_uses_only_target_image(monkeypatch, tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(reference)
    Image.new("RGB", (8, 8)).save(target)
    runner = object.__new__(VLLMVQARunner)
    runner.checkpoint = VLLMCheckpoint("test/model", "revision")
    runner._last_adapter_metadata = {}
    calls = []
    monkeypatch.setattr(
        runner,
        "_chat_generate",
        lambda paths, prompt, tokens: (
            calls.append((paths, prompt, tokens)) or '{"label":"alarm clock","bbox":[1,2,3,4]}'
        ),
    )
    runner.predict(
        [reference, target],
        "locate alarm clock",
        96,
        "diagnostic_oracle_bbox",
    )
    assert calls == [([target], "locate alarm clock", 96)]
    assert runner.prediction_metadata()["ground_truth_label_disclosed"] is True

    runner.predict(
        [reference, target],
        "locate predicted phrase",
        96,
        "diagnostic_predicted_label_bbox",
    )
    assert calls[-1] == ([target], "locate predicted phrase", 96)
    assert runner.prediction_metadata()["ground_truth_label_disclosed"] is False


def test_strict_output_parsers() -> None:
    binary = VQASample.from_dict(row())
    assert parse_output(binary, " YES. ", "qwen2.5-vl-3b").label == "yes"
    assert not parse_output(binary, "yes because it is", "qwen2.5-vl-3b").valid
    attribute = VQASample.from_dict(row("attr_composition"))
    assert normalize_label("Air_Duster_Can.") == "air duster can"
    parsed_attr = parse_output(
        attribute, '{"label":"air_duster_can","bbox":[100,120,200,220]}', "gemma4-12b"
    )
    assert parsed_attr.label == "air duster can" and parsed_attr.valid
    assert parsed_attr.bbox == pytest.approx((63.9, 57.48, 127.8, 105.38))
    bbox = VQASample.from_dict(row("depth_closest"))
    parsed_bbox = parse_output(
        bbox,
        '```json\n{"label":"alarm_clock","bbox":[100,120,200,220]}\n```',
        "qwen2.5-vl-3b",
    )
    assert parsed_bbox.valid
    assert parsed_bbox.bbox == pytest.approx((63.9, 57.48, 127.8, 105.38))
    unit_bbox = parse_output(
        bbox,
        '{"label":"alarm_clock","bbox":[0.1,0.2,0.3,0.4]}',
        "internvl2.5-8b",
    )
    assert unit_bbox.valid
    assert unit_bbox.coordinate_format == "normalized_0_1"
    assert unit_bbox.bbox == pytest.approx((63.9, 95.8, 191.7, 191.6))
    full_unit_bbox = parse_output(
        bbox,
        '{"label":"scene","bbox":[0,0,1,1]}',
        "internvl2.5-8b",
    )
    assert full_unit_bbox.coordinate_format == "normalized_0_1"
    assert full_unit_bbox.bbox == pytest.approx((0, 0, 639, 479))
    nested_bbox = parse_output(
        bbox,
        '{"label":"alarm_clock","bbox":[[100,120,200,220]]}',
        "internvl2.5-8b",
    )
    assert nested_bbox.valid
    assert nested_bbox.coordinate_format == "normalized_0_1000"
    parsed_loc = parse_output(
        bbox, "<loc0256><loc0160><loc0469><loc0320> alarm_clock<eos>", "paligemma2-3b"
    )
    assert parsed_loc.valid
    assert parsed_loc.bbox == pytest.approx((100, 120, 200, 219.84375))
    assert parsed_loc.label == "alarm clock"
    assert parsed_loc.coordinate_format == "paligemma_loc_1024"
    pali_json = parse_output(
        bbox,
        '{"label":"alarm clock","bbox":[156,250,313,500]}',
        "paligemma2-3b",
    )
    assert pali_json.valid
    assert pali_json.coordinate_format == "normalized_0_1000"


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
    assert result["primary_metric"] == "bbox_accuracy_at_0_5"
    assert result["bbox_scoring_uses_generated_label"] is False
    assert result["parse_rate"] == 1.0
    assert result["binary_accuracy"] == 1.0
    assert result["attribute_exact_match"] == 0.0
    assert result["bbox_accuracy_at_0_5"] == 1.0
    assert result["bbox_accuracy_at_0_25"] == 1.0
    assert result["bbox_accuracy_at_0_75"] == 1.0
    assert result["bbox_mean_accuracy_50_95"] == 1.0
    assert result["bbox_label_normalized_exact_match"] == 1.0
    assert result["bbox_label_token_f1"] == 1.0
    assert result["bbox_joint_label_exact_and_iou_at_0_5"] == 1.0
    assert bbox_iou((100, 120, 200, 220), (100, 120, 200, 220)) == 1.0


def test_label_token_f1_is_lexical_and_missing_labels_score_zero() -> None:
    assert label_token_f1("coffee can", "caned coffee") == pytest.approx(0.5)
    assert label_token_f1(None, "coffee can") == 0.0

    sample = VQASample.from_dict(row("depth_closest"))
    result = score_predictions([(sample, ParsedOutput(False, error="bad JSON"))])
    assert result["bbox_label_normalized_exact_match"] == 0.0
    assert result["bbox_label_token_f1"] == 0.0


def test_mean_accuracy_50_95_is_not_mislabeled_as_map() -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    # An IoU of 0.75 succeeds at .50, .55, .60, .65, .70 and .75: 6/10.
    parsed = ParsedOutput(True, label="alarm clock", bbox=(100, 120, 175, 220))
    result = score_predictions([(sample, parsed)])
    assert result["bbox_mean_accuracy_50_95"] == pytest.approx(0.6)
    assert "not detection mAP" in result["metric_notes"]["bbox_mean_accuracy_50_95"]


def test_batched_chat_content_labels_both_incontext_images() -> None:
    content = VLLMVQARunner._chat_content([Path("reference.png"), Path("target.png")], "question")
    assert [item.get("text") for item in content if item["type"] == "text"] == [
        "Image 1 — reference object:",
        "Image 2 — target scene:",
        "question",
    ]


def test_roster_matches_rpx_draft() -> None:
    assert [model.display_name for model in MODELS] == [
        "Florence 2 Base",
        "Florence 2 Large",
        "PaliGemma 2 3B",
        "Qwen2.5-VL 3B",
        "Qwen3-VL 2B",
        "DeepSeek-VL2 Tiny (1B active)",
        "InternVL 3.5 1B",
        "Gemma 4 E4B",
        "Phi-3.5-Vision 4.2B",
        "LLaVA-OneVision 7B",
        "Qwen2.5-VL 7B",
        "Qwen3-VL 8B",
        "Idefics3 8B",
        "InternVL 2.5 8B",
        "InternVL 3.5 14B",
        "DeepSeek-VL2 (4.5B active)",
        "PaliGemma 2 10B",
        "Gemma 4 12B",
    ]
    assert get_model("gemma4-12b").capabilities == {"bbox"}


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
    assert matrix["single_image_only"] is False
    assert matrix["max_images_per_prompt"] == 2
    assert matrix["inference_backend"] == "model-specific"
    assert matrix["inference_backends"] == [
        "vllm",
        "transformers-florence2",
        "transformers-paligemma2",
        "transformers-internvl",
    ]
    assert matrix["vllm_version"] == "0.28.0"
    assert [model["key"] for model in matrix["models"]] == [model.key for model in MODELS]
    assert [set(model["tasks"]) for model in matrix["models"]] == [
        set(model.capabilities) for model in MODELS
    ]
    florence = [model for model in matrix["models"] if model["key"].startswith("florence2-")]
    assert all(
        model["diagnostic_tasks"] == ["semantic_label", "phrase_grounding"] for model in florence
    )
    paligemma = [model for model in matrix["models"] if model["key"].startswith("paligemma2-")]
    assert all(
        model["diagnostic_tasks"] == ["semantic_label", "object_detection"] for model in paligemma
    )
    internvl35 = [model for model in matrix["models"] if model["key"].startswith("internvl3.5-")]
    assert len(internvl35) == 2
    assert all(model["backend"] == "transformers-internvl" for model in internvl35)


def test_vqa_docker_has_explicit_pinned_backends() -> None:
    root = Path(__file__).parents[2]
    dockerfile = (root / "docker" / "vqa-smoke" / "Dockerfile").read_text()
    entrypoint = (root / "docker" / "vqa-smoke" / "entrypoint.sh").read_text()
    runner = (root / "benchmark" / "scripts" / "run_vllm_vqa.py").read_text()
    assert " AS vqa_vllm" in dockerfile
    assert "run_vllm_vqa.py" in entrypoint
    assert "provenance(args.model)" in runner
    assert "transformers-florence2" in entrypoint
    assert "transformers==4.49.0" in dockerfile
    assert "/opt/rpx-envs/florence2" in dockerfile
    assert "run_gemma4_smoke.py" not in entrypoint
    assert "run_paligemma2_smoke.py" not in entrypoint


def test_paligemma_uses_native_transformers_backend() -> None:
    assert backend_name("paligemma2-10b") == "transformers-paligemma2"
    runner = object.__new__(PaliGemmaVQARunner)
    runner._last_adapter_metadata = {}
    assert runner.prediction_metadata() == {}


def test_internvl35_uses_native_transformers_backend() -> None:
    assert backend_name("internvl3.5-1b") == "transformers-internvl"
    assert backend_name("internvl3.5-14b") == "transformers-internvl"


def test_internvl35_uses_isolated_supported_transformers_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = (repo_root / "docker/vqa-smoke/Dockerfile").read_text()
    entrypoint = (repo_root / "docker/vqa-smoke/entrypoint.sh").read_text()

    assert "/opt/rpx-envs/internvl/bin/python -m pip install --no-cache-dir" in dockerfile
    assert "transformers==4.57.1" in dockerfile
    assert "/opt/rpx-envs/internvl/bin/python -m pip install --no-cache-dir --no-deps" in dockerfile
    assert '[[ "$1" == internvl3.5-* ]]' in entrypoint
    assert "/opt/rpx-envs/internvl/bin/python" in entrypoint


def test_internvl_dynamic_preprocess_preserves_official_tile_contract() -> None:
    from PIL import Image

    tiles = _dynamic_preprocess(Image.new("RGB", (1920, 1080)), max_num=6)
    assert 1 < len(tiles) <= 7
    assert all(tile.size == (448, 448) for tile in tiles)


def test_internvl_one_stage_call_preserves_image_order(monkeypatch, tmp_path: Path) -> None:
    from PIL import Image

    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (8, 8), "red").save(reference)
    Image.new("RGB", (8, 8), "blue").save(target)
    runner = object.__new__(InternVLVQARunner)
    runner._last_adapter_metadata = {}
    calls = []
    monkeypatch.setattr(
        runner,
        "_generate",
        lambda images, prompt, tokens: (
            calls.append(([image.getpixel((0, 0)) for image in images], prompt, tokens))
            or "object[[1,2,3,4]]"
        ),
    )
    raw = runner.predict(
        [reference, target], "question", 96, "bbox_native_internvl_grounding"
    )
    assert raw == "object[[1,2,3,4]]"
    assert calls == [([(255, 0, 0), (0, 0, 255)], "question", 96)]
    assert runner.prediction_metadata()["image_order"] == "reference_then_target"
    assert runner.prediction_metadata()["single_scored_model_call"] is True
    assert runner.prediction_metadata()["native_coordinate_format"] == "internvl_bbox_0_1000"


def test_internvl_native_bbox_is_scaled_strictly() -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    parsed = parse_output(
        sample,
        "<ref>alarm clock</ref><box>[[250,200,750,800]]</box>",
        "internvl3.5-14b",
    )
    assert parsed.valid
    assert parsed.label == "alarm clock"
    assert parsed.coordinate_format == "internvl_bbox_0_1000"
    assert parsed.bbox == pytest.approx((159.75, 95.8, 479.25, 383.2))


def test_internvl_native_bbox_rejects_ambiguous_or_bad_boxes() -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    parsed = parse_output(
        sample,
        "clock[[100,120,200,220]] other[[300,320,400,420]]",
        "internvl3.5-1b",
    )
    assert not parsed.valid
    assert parsed.error == "expected exactly one InternVL bbox"
    reversed_box = parse_output(
        sample, "clock[[300,400,200,500]]", "internvl3.5-1b"
    )
    assert not reversed_box.valid
    assert reversed_box.error == "InternVL bbox corners are not ordered XYXY"


def test_deepseek_native_bbox_decoding_is_strict() -> None:
    raw, metadata = VLLMVQARunner._decode_deepseek_grounding(
        "<|ref|>alarm clock<|/ref|><|det|>[[100,200,300,400]]<|/det|>"
    )
    decoded = json.loads(raw)
    assert decoded["label"] == "alarm clock"
    assert decoded["bbox"] == pytest.approx(
        [100 * 1000 / 999, 200 * 1000 / 999, 300 * 1000 / 999, 400 * 1000 / 999]
    )
    assert metadata["native_candidate_count"] == 1
    assert metadata["native_coordinate_format"] == "deepseek_vl2_normalized_0_999"
    assert metadata["coordinate_format"] == "normalized_0_1000"
    ambiguous, ambiguous_metadata = VLLMVQARunner._decode_deepseek_grounding(
        "<|ref|>object<|/ref|><|det|>[[1,2,3,4],[5,6,7,8]]<|/det|>"
    )
    assert "bbox" not in json.loads(ambiguous)
    assert json.loads(ambiguous)["candidate_count"] == 2
    assert ambiguous_metadata["native_candidate_count"] == 2

    endpoint, _ = VLLMVQARunner._decode_deepseek_grounding(
        "<|ref|>scene<|/ref|><|det|>[[0,0,999,999]]<|/det|>"
    )
    endpoint_bbox = json.loads(endpoint)["bbox"]
    assert endpoint_bbox == [0.0, 0.0, 1000.0, 1000.0]
    endpoint_parsed = parse_output(
        VQASample.from_dict(row("depth_closest")),
        endpoint,
        "deepseek-vl2",
    )
    assert endpoint_parsed.valid
    assert endpoint_parsed.bbox == (0.0, 0.0, 639.0, 479.0)
    rounded_endpoint = parse_output(
        VQASample.from_dict(row("depth_closest")),
        '{"label":"scene","bbox":[0,0,1000.0000000000001,1000.0000000000001]}',
        "deepseek-vl2",
    )
    assert rounded_endpoint.valid
    assert rounded_endpoint.bbox == (0.0, 0.0, 639.0, 479.0)
    genuinely_outside = parse_output(
        VQASample.from_dict(row("depth_closest")),
        '{"label":"scene","bbox":[0,0,1000.01,1000]}',
        "deepseek-vl2",
    )
    assert not genuinely_outside.valid


def test_deepseek_checkpoint_gpu_topology_is_explicit() -> None:
    assert VLLMCheckpoint.__dataclass_fields__["tensor_parallel_size"].default == 1
    assert VLLM_CHECKPOINTS["deepseek-vl2-tiny"].tensor_parallel_size == 1
    assert VLLM_CHECKPOINTS["deepseek-vl2"].tensor_parallel_size == 2


def test_deepseek_full_uses_vllm_live_text_config_override() -> None:
    overrides = VLLM_CHECKPOINTS["deepseek-vl2"].engine_kwargs["hf_overrides"]
    assert overrides["architectures"] == ["DeepseekVLV2ForCausalLM"]
    assert overrides["text_config"]["use_mla"] is True
    assert overrides["text_config"]["kv_lora_rank"] == 512
    assert overrides["text_config"]["qk_rope_head_dim"] == 64
    assert overrides["text_config"]["qk_nope_head_dim"] == 128
    assert overrides["text_config"]["v_head_dim"] == 128
    # q_lora_rank remains the checkpoint's explicit null, selecting q_proj.
    assert "q_lora_rank" not in overrides["text_config"]


def test_remote_vqa_client_does_not_enforce_local_engine_gpu_count(monkeypatch) -> None:
    def unexpected() -> bool:
        raise AssertionError("remote client must not inspect CUDA")

    monkeypatch.setattr("run_vllm_vqa.torch.cuda.is_available", unexpected)
    validate_gpu_environment("http://127.0.0.1:8000")


def test_local_vqa_engine_still_requires_exactly_one_gpu(monkeypatch) -> None:
    monkeypatch.setattr("run_vllm_vqa.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("run_vllm_vqa.torch.cuda.device_count", lambda: 2)
    with pytest.raises(SystemExit, match="exactly one visible GPU"):
        validate_gpu_environment(None)


def test_florence_target_bbox_remaps_composite_without_gt_selection() -> None:
    geometry = ImageGeometry(
        width=1288,
        height=508,
        target_left=648,
        target_top=28,
        target_width=640,
        target_height=480,
    )
    assert FlorenceVQARunner._target_bbox([712, 76, 968, 268], geometry) == [
        100.0,
        100.0,
        500.0,
        500.0,
    ]
    assert FlorenceVQARunner._target_bbox([10, 50, 200, 250], geometry) is None


def test_florence_decodes_only_unambiguous_target_panel_candidate() -> None:
    class Processor:
        @staticmethod
        def post_process_generation(*_args, **_kwargs):
            return {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                    "bboxes": [[10, 50, 200, 250], [712, 76, 968, 268]],
                    "labels": ["reference", "target object"],
                }
            }

    runner = object.__new__(FlorenceVQARunner)
    runner.processor = Processor()
    geometry = ImageGeometry(1288, 508, 648, 28, 640, 480)
    raw, metadata = runner._decode_grounding("native loc tokens", geometry)
    assert json.loads(raw) == {
        "label": "target object",
        "bbox": [100.0, 100.0, 500.0, 500.0],
    }
    assert metadata["native_candidate_count"] == 2
    assert metadata["target_candidate_count"] == 1
    assert metadata["single_scored_model_call"] is False
    assert metadata["diagnostic_only"] is True
    assert metadata["adapter"] == "florence2_composite_phrase_grounding"
    assert metadata["multi_image_accommodation"] == "labelled_side_by_side_composite"
    assert metadata["image_count"] == 2


def test_florence_direct_grounding_is_marked_as_one_scored_call() -> None:
    class Processor:
        @staticmethod
        def post_process_generation(*_args, **_kwargs):
            return {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                    "bboxes": [[100, 120, 200, 220]],
                    "labels": ["answer"],
                }
            }

    runner = object.__new__(FlorenceVQARunner)
    runner.processor = Processor()
    raw, metadata = runner._decode_grounding(
        "native loc tokens",
        ImageGeometry(640, 480),
        "bbox_native_question_grounding",
    )
    assert "bbox" in json.loads(raw)
    assert metadata["adapter"] == "direct_native_question_grounding"
    assert metadata["single_scored_model_call"] is True
    assert metadata["diagnostic_only"] is False


def test_gallery_exposes_rejected_florence_candidates_without_selecting() -> None:
    raw = json.dumps(
        {
            "error": "expected exactly one target-region candidate",
            "candidates": [
                {"label": "cup", "bbox": [10, 20, 30, 40]},
                {"label": "bottle", "bbox": [100, 200, 300, 400]},
            ],
        }
    )
    row = {"adapter_metadata": {"adapter": "florence2_native_phrase_grounding"}}
    assert _native_candidates(raw, row) == [
        {"label": "cup", "bbox": [10, 20, 30, 40]},
        {"label": "bottle", "bbox": [100, 200, 300, 400]},
    ]


def test_florence_candidate_best_iou_is_diagnostic_only() -> None:
    sample = VQASample.from_dict(row("depth_closest"))
    raw = json.dumps(
        {
            "error": "expected exactly one target-region candidate",
            "candidates": [
                {"label": "wrong", "bbox": [0, 0, 50, 50]},
                {"label": "alarm clock", "bbox": [156.5, 250, 313, 459]},
            ],
        }
    )
    result = diagnostic_bbox(sample, raw, "florence2-large")
    assert result["strict_valid"] is False
    assert result["best_coordinate_format"] == "native_candidate_2_gt_selected"
    assert result["best_iou"] > 0.95
