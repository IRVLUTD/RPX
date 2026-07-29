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
        declaration = f"FROM {previous} AS {model['target']}"
        assert declaration in dockerfile
        previous = model["target"]


def test_base_image_is_digest_pinned_and_weights_are_not_copied():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()

    assert payload["base_image"]["linux_amd64_digest"] in dockerfile
    assert "weights_baked_into_image" in (
        DOCKER_DIR / "register_model_env.py"
    ).read_text()
    assert not re.search(r"\b(COPY|ADD)\b.*\.(pt|pth|ckpt|safetensors)\b", dockerfile)
