import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = ROOT / "docker" / "tracking-smoke"


def test_paper_tracking_matrix_is_complete_and_ordered():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    models = payload["models"]

    assert [model["order"] for model in models] == list(range(1, 10))
    assert [model["id"] for model in models] == [
        "sam3.1",
        "sam2",
        "sam2-plus",
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
        "box",
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


def test_environment_registration_can_prepend_partial_source_packages():
    script_path = DOCKER_DIR / "register_model_env.py"
    spec = importlib.util.spec_from_file_location("register_model_env", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    contents = module._pth_contents(["/opt/rpx-models/sam2_plus"], prepend=True)

    assert contents == (
        "import sys; sys.path[:0] = ['/opt/rpx-models/sam2_plus']\n"
    )


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
    cumulative = (DOCKER_DIR / "Dockerfile.motip-cumulative").read_text()
    patch_name = "motip-cuda-toolkit-build.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert f"COPY docker/tracking-smoke/patches/{patch_name}" in dockerfile
    assert "git -C /opt/rpx-models/motip apply --check" in dockerfile
    assert "--python-path models/ops" in dockerfile
    assert "--python-path models/ops" in cumulative
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
    cumulative = (DOCKER_DIR / "Dockerfile.masa-cumulative").read_text()
    lock = (DOCKER_DIR / "requirements-masa-runtime.lock").read_text()
    matrix = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    masa = next(model for model in matrix["models"] if model["id"] == "masa")

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
    assert "--prepend-source" in dockerfile
    assert "--import projects.Detic_new.detic" in dockerfile
    assert "dereksiyuanli/masa" in cumulative
    assert masa["checkpoint_repo"] == "dereksiyuanli/masa"
    assert masa["checkpoint_revision"] == (
        "25ed372c47f2c46cf36fd446d1b657b656bc7ea9"
    )
    assert "Dockerfile.masa-cumulative" in (
        DOCKER_DIR / "build_masa_rpx.sh"
    ).read_text()


def test_mits_overlay_is_pinned_and_uses_box_initialization():
    dockerfile = (DOCKER_DIR / "Dockerfile.mits-cumulative").read_text()
    builder = (DOCKER_DIR / "build_mits_rpx.sh").read_text()
    lock = (DOCKER_DIR / "requirements-mits-runtime.lock").read_text()
    patch_name = "mits-torch27-checkpoint.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert "462ebee2c995818998d5f96ab6b615dd86c42688" in dockerfile
    assert "4aa4431bf2f6057c9d40d4990227e3e655699ed3" in dockerfile
    assert "--model mits" in dockerfile
    assert "--prompt box" in dockerfile
    assert "gdrive-1Db9DxXc-gyRkxhKs0AMXJ2RH6DyHWCfq" in dockerfile
    assert f"patches/{patch_name}" in dockerfile
    assert "weights_only=False" in patch
    assert "gdown==5.2.0" in lock
    assert "timm==0.9.16" in lock
    assert "Dockerfile.mits-cumulative" in builder
    assert "sam2-plus-rpx-sha-711b53f7b7e2" in builder


def test_xmem_overlay_is_pinned_and_uses_mask_initialization():
    dockerfile = (DOCKER_DIR / "Dockerfile.xmem-cumulative").read_text()
    builder = (DOCKER_DIR / "build_xmem_rpx.sh").read_text()
    patch_name = "xmem-torch27-checkpoint.patch"
    patch = (DOCKER_DIR / "patches" / patch_name).read_text()

    assert "f3b841d50df058910bbf690229ddc15fb1aef7d6" in dockerfile
    assert "--model xmem" in dockerfile
    assert "--prompt mask" in dockerfile
    assert "--checkpoint-revision v1.0" in dockerfile
    assert f"patches/{patch_name}" in dockerfile
    assert "weights_only=True" in patch
    assert "Dockerfile.xmem-cumulative" in builder
    assert "mits-rpx-latest" in builder


def test_incremental_milestone_registry_checks_allow_later_adapters():
    mits = (DOCKER_DIR / "Dockerfile.mits-cumulative").read_text()
    xmem = (DOCKER_DIR / "Dockerfile.xmem-cumulative").read_text()

    assert "required <= TRACKER_CLASSES.keys()" in mits
    assert "required <= TRACKER_CLASSES.keys()" in xmem


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

    # One copy in the base plus one in each of the eight externally overridable
    # stages prevents an old published base from retaining stale build tooling.
    assert dockerfile.count(copy_line) == 9


def test_base_image_is_digest_pinned_and_weights_are_not_copied():
    payload = json.loads((DOCKER_DIR / "model-matrix.json").read_text())
    dockerfile = (DOCKER_DIR / "Dockerfile.models").read_text()

    assert payload["base_image"]["linux_amd64_digest"] in dockerfile
    assert "weights_baked_into_image" in (
        DOCKER_DIR / "register_model_env.py"
    ).read_text()
    assert not re.search(r"\b(COPY|ADD)\b.*\.(pt|pth|ckpt|safetensors)\b", dockerfile)
