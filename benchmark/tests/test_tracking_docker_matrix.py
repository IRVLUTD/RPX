import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = ROOT / "docker" / "tracking-smoke"


def test_paper_tracking_matrix_is_complete_and_ordered():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    models = payload["models"]

    assert [model["order"] for model in models] == list(range(1, 11))
    assert [model["id"] for model in models] == [
        "sam3.1",
        "sam2",
        "sam2-plus",
        "samurai",
        "edgetam",
        "deaot",
        "cutie",
        "motip",
        "masa",
        "grounded-sam2",
    ]
    assert [model["prompt"] for model in models] == [
        "mask",
        "mask",
        "mask",
        "mask",
        "mask",
        "mask",
        "mask",
        "detector",
        "detector",
        "text",
    ]


def test_every_model_has_a_pinned_source_and_cumulative_target():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    previous = "tracking_models_base"

    for model in payload["models"]:
        assert re.fullmatch(r"[0-9a-f]{40}", model["source_revision"])
        assert model["source_revision"] in dockerfile
        if model["order"] == 1:
            declaration = f"FROM {previous} AS {model['target']}"
        else:
            base_arg = model["base_arg"]
            assert f"ARG {base_arg}={previous}" in dockerfile
            declaration = f"FROM ${{{base_arg}}} AS {model['target']}"
        assert declaration in dockerfile
        if model["order"] > 1:
            stage = dockerfile.split(declaration, 1)[1].split("\nFROM ", 1)[0]
            assert "ARG RPX_GIT_SHA" in stage
            assert "ENV RPX_GIT_SHA=${RPX_GIT_SHA}" in stage
        previous = model["target"]


def test_samurai_runtime_declares_its_missing_logger_dependency():
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    assert "loguru==0.7.3" in dockerfile


def test_deaot_correlation_has_versioned_torch27_compatibility_patch():
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    patch_name = "pytorch-correlation-torch27-openmp.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert f"COPY docker/tracking-smoke/patches/{patch_name}" in dockerfile
    assert "git -C /opt/rpx-models/deaot-correlation apply --check" in dockerfile
    assert patch.count("-  #pragma omp parallel for") == 2
    assert "correlation.cpp" in patch


def test_builder_supports_one_model_overlay_mode():
    builder = (DOCKER_DIR / "build_and_push_models.sh").read_text()
    assert "--model MODEL" in builder
    assert "--base-image IMAGE" in builder


def test_base_image_is_digest_pinned_and_weights_are_not_copied():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()

    assert payload["base_image"]["linux_amd64_digest"] in dockerfile
    assert "weights_baked_into_image" in (
        DOCKER_DIR / "register_model_env.py"
    ).read_text()
    assert not re.search(r"\b(COPY|ADD)\b.*\.(pt|pth|ckpt|safetensors)\b", dockerfile)
