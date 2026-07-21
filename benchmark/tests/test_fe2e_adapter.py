from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))

from depth_models.fe2e import FE2EAdapter  # noqa: E402

from rpx_benchmark.exceptions import AdapterError  # noqa: E402


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.dit = SimpleNamespace(parameters=lambda: iter(()))

    def generate_image(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        ramp = torch.linspace(0.0, 1.0, 24, dtype=torch.float32).reshape(1, 1, 4, 6)
        depth = ramp.repeat(1, 3, 1, 1)
        return [], depth, torch.zeros_like(depth)


def _bare_adapter() -> FE2EAdapter:
    adapter = FE2EAdapter.__new__(FE2EAdapter)
    adapter._pipeline = _FakePipeline()
    adapter._args = SimpleNamespace(prompt_type="empty", single_denoise=True)
    adapter.seed = 1234
    return adapter


def test_fe2e_inference_uses_official_one_step_empty_prompt_path():
    adapter = _bare_adapter()
    rgb = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)

    depth = adapter(rgb)

    assert depth.shape == (8, 10)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert float(np.ptp(depth)) > 0
    assert depth.min() == 0.0  # preserve official normalized endpoint
    call = adapter._pipeline.calls[0]
    assert call["prompt"] == ""
    assert call["negative_prompt"] == ""
    assert call["num_steps"] == 1
    assert call["cfg_guidance"] == 6.0
    assert call["seed"] == 1234
    assert call["num_samples"] == 1
    assert call["args"] is adapter._args
    tensor = call["ref_images"]
    assert tuple(tensor.shape) == (1, 3, 8, 10)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_fe2e_sequence_dispatch_preserves_order_and_count():
    adapter = _bare_adapter()
    images = [
        np.zeros((8, 10, 3), dtype=np.uint8),
        np.full((8, 10, 3), 255, dtype=np.uint8),
    ]
    outputs = adapter(images)
    assert len(outputs) == 2
    assert len(adapter._pipeline.calls) == 2


def test_fe2e_rejects_invalid_rgb_shape():
    adapter = _bare_adapter()
    with pytest.raises(AdapterError, match="HxWx3"):
        adapter(np.zeros((8, 10), dtype=np.uint8))


def test_fe2e_constructor_pins_all_official_downloads(tmp_path, monkeypatch):
    source = tmp_path / "fe2e"
    (source / "latent").mkdir(parents=True)
    (source / "latent" / "no_info.npz").write_bytes(b"fixture")
    downloads: list[tuple[str, str, str]] = []

    hub = types.ModuleType("huggingface_hub")

    def fake_download(repo_id, filename, revision):
        downloads.append((repo_id, filename, revision))
        return str(tmp_path / filename.replace("/", "__"))

    hub.hf_hub_download = fake_download
    infer_pkg = types.ModuleType("infer")
    infer_pkg.__path__ = []
    infer_module = types.ModuleType("infer.inference")
    constructed: dict = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.dit = object()

    infer_module.ImageGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "infer", infer_pkg)
    monkeypatch.setitem(sys.modules, "infer.inference", infer_module)

    adapter = FE2EAdapter(device="cuda", source_path=str(source))

    assert downloads == [
        (FE2EAdapter.MODEL_ID, FE2EAdapter.DIT_FILENAME, FE2EAdapter.MODEL_REVISION),
        (FE2EAdapter.MODEL_ID, FE2EAdapter.VAE_FILENAME, FE2EAdapter.MODEL_REVISION),
        (FE2EAdapter.MODEL_ID, FE2EAdapter.LORA_FILENAME, FE2EAdapter.MODEL_REVISION),
    ]
    assert constructed["qwen2vl_model_path"] is None
    assert constructed["quantized"] is False
    assert constructed["offload"] is False
    assert constructed["device"] == "cuda"
    assert adapter._args.prompt_type == "empty"
    assert adapter._args.single_denoise is True
    assert adapter._args.norm_type == "ln"
    assert adapter._args.task_name == "depth"


def test_fe2e_requires_pinned_empty_prompt_latent(tmp_path):
    with pytest.raises(AdapterError, match="no_info.npz"):
        FE2EAdapter(device="cuda", source_path=str(tmp_path))


def test_fe2e_docker_overlay_pins_official_runtime():
    dockerfile = (REPO_ROOT / "docker" / "depth-fe2e" / "Dockerfile").read_text()
    requirements = (
        REPO_ROOT / "docker" / "depth-fe2e" / "requirements-fe2e.lock"
    ).read_text()

    assert FE2EAdapter.UPSTREAM_REVISION in dockerfile
    assert "torch==2.6.0" in dockerfile
    assert "torchvision==0.21.0" in dockerfile
    assert "flash_attn-2.7.2.post1" in dockerfile
    assert "transformers==4.49.0" in requirements
    assert "liger-kernel==0.7.0" in requirements
    assert "diffusers[torch]==0.32.2" in requirements
    assert "accelerate==1.13.0" in requirements
    assert "numpy==1.25.0" in requirements
    assert FE2EAdapter.DIT_SHA256.startswith("f378db31")
    assert FE2EAdapter.VAE_SHA256.startswith("afc8e282")
    assert FE2EAdapter.LORA_SHA256.startswith("9c2112d3")
