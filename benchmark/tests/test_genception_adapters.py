from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from rpx_benchmark.api import VideoSample
from rpx_benchmark.vqa.contract import VQASample
from rpx_benchmark.vqa.prompts import build_prompt

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from depth_models import MODEL_REGISTRY  # noqa: E402
from depth_models.genception import GenCeptionImageDepth  # noqa: E402
from genception_capabilities import capability_report  # noqa: E402
from tracking_models import TRACKER_CLASSES  # noqa: E402
from tracking_models.genception_tracker import GenCeption13BTracker  # noqa: E402
from video_depth_models.genception import GenCeptionVideoDepth  # noqa: E402
from vqa_models.backend_registry import backend_name  # noqa: E402
from vqa_models.genception_backend import GenCeptionVQARunner  # noqa: E402


class FakeRuntime:
    segmentation_threshold = 0.5

    def __init__(self) -> None:
        self.setup_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1

    def predict_depth(self, video: np.ndarray) -> np.ndarray:
        return np.arange(np.prod(video.shape[:3]), dtype=np.float32).reshape(video.shape[:3])

    def predict_mask_scores(self, video: np.ndarray, expression: str) -> np.ndarray:
        scores = np.zeros(video.shape[:3], dtype=np.float32)
        if "left" in expression:
            scores[..., : video.shape[2] // 2] = 0.8
        else:
            scores[..., video.shape[2] // 2 :] = 0.9
        return scores


def test_both_image_and_video_depth_adapters() -> None:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    for variant in ("1.3b", "14b"):
        runtime = FakeRuntime()
        image_model = GenCeptionImageDepth(variant, runtime=runtime)
        assert image_model(rgb).shape == (3, 4)
        video_model = GenCeptionVideoDepth(variant=variant, runtime=runtime)
        video_model.setup()
        prediction = video_model.predict([VideoSample("clip", rgb[None], None)])[0]
        assert prediction.depth_map_seq.shape == (1, 3, 4)


def test_text_tracking_merges_prompts_by_foreground_score(tmp_path: Path) -> None:
    for index in range(2):
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(tmp_path / f"{index:05d}.jpg")
    tracker = GenCeption13BTracker(runtime=FakeRuntime())
    prompts = [
        SimpleNamespace(mask_index=3, prompt_text="left object", source_catalog_id="a", object_id="a"),
        SimpleNamespace(mask_index=7, prompt_text="right object", source_catalog_id="b", object_id="b"),
    ]
    predictions, latencies = tracker.track(tmp_path, (4, 6), 2, prompts)
    np.testing.assert_array_equal(predictions[0][:, :3], 3)
    np.testing.assert_array_equal(predictions[0][:, 3:], 7)
    assert len(latencies) == 2
    assert tracker.prediction_metadata()["prompt_count"] == 2


def test_vqa_regular_and_in_context_return_target_bbox(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(target)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(reference)
    runner = GenCeptionVQARunner(
        "genception-1.3b", tmp_path, runtime=FakeRuntime()
    )
    regular = json.loads(
        runner.predict([target], "right object", 0, "bbox_native_genception_refvos")
    )
    in_context = json.loads(
        runner.predict(
            [reference, target], "right matching object", 0, "bbox_native_genception_refvos"
        )
    )
    assert regular["bbox"] == [600.0, 0.0, 1000.0, 1000.0]
    assert in_context["bbox"] == regular["bbox"]
    assert runner.prediction_metadata()["image_count"] == 2


def test_capability_report_is_explicit_about_pose_release_gap() -> None:
    report = capability_report()
    assert set(report["variants"]) == {"1.3b", "14b"}
    assert "camera_pose" in report["unavailable"]
    assert "camera_pose" not in report["supported"]


def test_all_public_registries_contain_both_variants() -> None:
    for key in ("genception-1.3b", "genception-14b"):
        assert key in MODEL_REGISTRY
        assert key in TRACKER_CLASSES
        assert backend_name(key) == "jax-genception"


def test_vqa_prompt_uses_gt_free_native_expression() -> None:
    sample = VQASample.from_dict(
        {
            "scene_id": "scene001",
            "kind": "mos",
            "phase": 0,
            "frame": "00001",
            "img_w": 6,
            "img_h": 4,
            "type": "depth_closest",
            "question": "Which object is closest to the camera?",
            "answer": "cup",
            "answer_bbox": [3, 0, 5, 3],
        }
    )
    prompt = build_prompt(sample, "genception-14b")
    assert prompt.text == "the object closest to the camera"
    assert prompt.output_kind == "bbox_native_genception_refvos"
    assert "cup" not in prompt.text


def test_vqa_docker_pins_official_genception_source() -> None:
    dockerfile = (Path(__file__).parents[2] / "docker/vqa-smoke/Dockerfile").read_text()
    entrypoint = (Path(__file__).parents[2] / "docker/vqa-smoke/entrypoint.sh").read_text()
    assert "a47e55120cc2b00027c19c9f1937831e77541056" in dockerfile
    assert "'jax[cuda12]==0.11.2'" in dockerfile
    assert '[[ "$1" == genception-* ]]' in entrypoint

    dedicated = (Path(__file__).parents[2] / "docker/genception/Dockerfile").read_text()
    dedicated_entrypoint = (
        Path(__file__).parents[2] / "docker/genception/entrypoint.sh"
    ).read_text()
    assert "a47e55120cc2b00027c19c9f1937831e77541056" in dedicated
    assert "RPX_GENCEPTION_ROOT=/models/genception" in dedicated
    assert "download_genception.py" in dedicated_entrypoint
