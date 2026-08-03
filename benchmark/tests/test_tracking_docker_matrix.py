import json
import importlib.util
import re
import sys
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


def test_edgetam_rpx_overlay_patches_noncontiguous_expand_view():
    dockerfile = (DOCKER_DIR / "Dockerfile.edgetam-rpx").read_text()
    patch_name = "edgetam-pytorch-noncontiguous.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert f"patches/{patch_name}" in dockerfile
    assert "git -C /opt/rpx-models/edgetam apply --check" in dockerfile
    assert (
        "-        latents_2d = self.latents_2d.unsqueeze(0)"
        ".expand(B, -1, -1).view(-1, 1, C)"
    ) in patch
    assert (
        "+        latents_2d = self.latents_2d.unsqueeze(0)"
        ".expand(B, -1, -1).reshape(-1, 1, C)"
    ) in patch


def test_environment_registration_supports_namespace_packages(tmp_path):
    script_path = DOCKER_DIR / "register_model_env.py"
    spec = importlib.util.spec_from_file_location("register_model_env", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    package_root = tmp_path / "namespace_root"
    namespace = package_root / "rpx_test_namespace"
    namespace.mkdir(parents=True)
    sys.path.insert(0, str(package_root))
    try:
        assert module._module_location("rpx_test_namespace") == str(
            namespace.resolve()
        )
    finally:
        sys.path.remove(str(package_root))
        sys.modules.pop("rpx_test_namespace", None)


def test_deaot_correlation_has_versioned_torch27_compatibility_patch():
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    patch_name = "pytorch-correlation-torch27-openmp.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert f"COPY docker/tracking-smoke/patches/{patch_name}" in dockerfile
    assert "git -C /opt/rpx-models/deaot-correlation apply --check" in dockerfile
    assert patch.count("-  #pragma omp parallel for") == 2
    assert "correlation.cpp" in patch


def test_motip_uses_toolkit_presence_for_image_build():
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    patch_name = "motip-cuda-toolkit-build.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert f"COPY docker/tracking-smoke/patches/{patch_name}" in dockerfile
    assert "git -C /opt/rpx-models/motip apply --check" in dockerfile
    assert "-    if torch.cuda.is_available() and CUDA_HOME is not None:" in patch
    assert "+    if CUDA_HOME is not None:" in patch
    assert patch.count(
        "-        AT_DISPATCH_FLOATING_TYPES(value.type(),"
    ) == 2
    assert patch.count(
        "+        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
    ) == 2


def test_masa_installs_pinned_mmdetection_runtime_requirements():
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    lock = (DOCKER_DIR / "requirements-masa-runtime.lock").read_text()

    assert "COPY docker/tracking-smoke/requirements-masa-runtime.lock" in dockerfile
    assert "-r /tmp/requirements-masa-runtime.lock" in dockerfile
    assert "shapely==" in lock
    assert "six==" in lock
    assert "terminaltables==" in lock
    assert "pydantic==2.7.0" in lock
    assert "motmetrics==" in lock
    assert "nanoid==" in lock
    assert "plyfile==" in lock
    assert "rpx-clone-pinned https://github.com/scalabel/scalabel.git" in dockerfile
    assert "78b26be9800596a3c0c2a999f9e04deb8e3fc29a" in dockerfile
    assert "--no-deps --no-build-isolation /opt/rpx-models/scalabel" in dockerfile
    assert "rpx-clone-pinned https://github.com/SysCV/tet.git" in dockerfile
    assert "a62a9c0affec3a97f2cd0263141c53bcfb9c79f7" in dockerfile
    assert "--no-deps --no-build-isolation /opt/rpx-models/tet/teta" in dockerfile
    assert "--import teta" in dockerfile
    assert "--import scalabel" in dockerfile


def test_builder_supports_one_model_overlay_mode():
    builder = (DOCKER_DIR / "build_and_push_models.sh").read_text()
    assert "--model MODEL" in builder
    assert "--base-image IMAGE" in builder


def test_every_incremental_overlay_refreshes_registration_helper():
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()
    copy_line = (
        "COPY --chmod=0755 docker/tracking-smoke/register_model_env.py "
        "/usr/local/bin/rpx-register-model-env"
    )

    # One copy in the base plus one in each of the nine externally overridable
    # stages prevents an old published base from retaining stale build tooling.
    assert dockerfile.count(copy_line) == 10


def test_base_image_is_digest_pinned_and_weights_are_not_copied():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()

    assert payload["base_image"]["linux_amd64_digest"] in dockerfile
    assert "weights_baked_into_image" in (
        DOCKER_DIR / "register_model_env.py"
    ).read_text()
    assert not re.search(r"\b(COPY|ADD)\b.*\.(pt|pth|ckpt|safetensors)\b", dockerfile)
