"""Static operator-contract tests that do not require a GPU or Docker daemon."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark" / "scripts"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "rpx_depth_paper_launcher", SCRIPTS / "run_depth_paper.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_pins_dataset_checkpoint_and_split_contracts() -> None:
    launcher = _load_launcher()
    assert launcher.PINNED_REVISION == "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
    assert launcher.MODEL_CHECKPOINT == "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    assert launcher.EXPECTED == {
        "easy": (24_750, 99),
        "medium": (24_750, 99),
        "hard": (25_500, 102),
    }
    assert sum(value[0] for value in launcher.EXPECTED.values()) == 75_000
    assert sum(value[1] for value in launcher.EXPECTED.values()) == 300


def test_latency_cache_is_reused_only_for_exact_protocol(tmp_path: Path) -> None:
    launcher = _load_launcher()
    path = tmp_path / "latency.json"
    payload = {
        "schema_version": "rpx-d1f-latency-v1",
        "model_checkpoint": launcher.MODEL_CHECKPOINT,
        "actual_torch_dtype": "torch.float16",
        "warmup_forwards": 3,
        "measured_forwards": 100,
        "prediction_writes": False,
        "median_ms": 1.0,
    }
    path.write_text(json.dumps(payload))
    assert launcher._valid_latency_file(path)
    payload["actual_torch_dtype"] = "torch.float32"
    path.write_text(json.dumps(payload))
    assert not launcher._valid_latency_file(path)


def test_thin_docker_overlay_is_immutable_and_checks_required_imports() -> None:
    dockerfile = (ROOT / "docker" / "depth-paper" / "Dockerfile").read_text()
    digest = "sha256:3b86de0e4e136587d6ea7d816a482af15cb115dc6a14f16ea8f7d9356ab220cc"
    assert f"FROM vndhiran123/rpx-depth-smoke@{digest}" in dockerfile
    assert "COPY benchmark /opt/rpx/benchmark" in dockerfile
    assert "import numpy, scipy, pyarrow" in dockerfile
    assert "--no-deps --force-reinstall" in dockerfile
