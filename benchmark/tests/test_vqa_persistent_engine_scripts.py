from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCRIPT_DIR = REPO_ROOT / "docker" / "vqa-smoke"


def _engine_name(instance: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if instance is None:
        env.pop("RPX_VQA_ENGINE_INSTANCE", None)
    else:
        env["RPX_VQA_ENGINE_INSTANCE"] = instance
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; rpx_vqa_engine_name "internvl3.5-1b"',
            "bash",
            str(SCRIPT_DIR / "persistent_engine_lib.sh"),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_engine_name_is_backward_compatible_without_instance() -> None:
    result = _engine_name(None)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "rpx-vqa-internvl3.5-1b"


def test_engine_name_supports_same_model_replicas() -> None:
    gpu3 = _engine_name("gpu3")
    gpu4 = _engine_name("gpu4")
    assert gpu3.returncode == 0, gpu3.stderr
    assert gpu4.returncode == 0, gpu4.stderr
    assert gpu3.stdout == "rpx-vqa-internvl3.5-1b-gpu3"
    assert gpu4.stdout == "rpx-vqa-internvl3.5-1b-gpu4"


def test_engine_name_rejects_shell_metacharacters() -> None:
    result = _engine_name("gpu3;false")
    assert result.returncode == 2
    assert "must match" in result.stderr


def test_persistent_shell_scripts_parse() -> None:
    scripts = [
        SCRIPT_DIR / "persistent_engine_lib.sh",
        SCRIPT_DIR / "start_persistent_engine.sh",
        SCRIPT_DIR / "run_persistent_smoke.sh",
        SCRIPT_DIR / "run_persistent_acceptance.sh",
        SCRIPT_DIR / "run_persistent_benchmark.sh",
        SCRIPT_DIR / "run_persistent_diagnostic.sh",
        SCRIPT_DIR / "run_persistent_diagnostic_acceptance.sh",
        SCRIPT_DIR / "run_persistent_diagnostic_smoke.sh",
        SCRIPT_DIR / "run_dual_gpu_persistent_benchmark.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
