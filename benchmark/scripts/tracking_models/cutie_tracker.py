"""Official Cutie memory-based VOS adapter for the RPX D3 protocol."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

CUTIE_MODEL_ID = "hkchengrex/Cutie"
CUTIE_MODEL_REVISION = "v1.0"
CUTIE_CHECKPOINT = "cutie-base-mega.pth"
CUTIE_CHECKPOINT_MD5 = "a6071de6136982e396851903ab4c083a"
CUTIE_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/Cutie/releases/download/"
    f"{CUTIE_MODEL_REVISION}/{CUTIE_CHECKPOINT}"
)
CUTIE_CONFIG_DIR = "/opt/rpx-models/cutie/cutie/config"


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _download_checkpoint(cache_root: Path) -> Path:
    destination = cache_root / "cutie" / CUTIE_MODEL_REVISION / CUTIE_CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _digest(destination, "md5") == CUTIE_CHECKPOINT_MD5:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(CUTIE_CHECKPOINT_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if _digest(temporary, "md5") != CUTIE_CHECKPOINT_MD5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Cutie checkpoint failed its official MD5 integrity check.")
    temporary.replace(destination)
    return destination


class CutieTracker:
    """Run the official Cutie base-mega model from the first-frame GT mask."""

    model_name = "cutie"
    model_id = CUTIE_MODEL_ID
    model_revision = CUTIE_MODEL_REVISION
    checkpoint_filename = CUTIE_CHECKPOINT
    config_directory = CUTIE_CONFIG_DIR
    adapter_label = "Cutie"
    prompt_type = "mask"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production Cutie adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the Cutie adapter.")

        config_dir = Path(self.config_directory)
        if not (config_dir / "eval_config.yaml").is_file():
            raise RuntimeError(f"Cutie config is missing under {config_dir}; use the pinned image.")

        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(config_dir), version_base="1.3.2")
        cfg = compose(config_name="eval_config")
        checkpoint = _download_checkpoint(Path(os.environ.get("HF_HOME", "/cache/huggingface")))
        backbone_cache = Path(os.environ.get("TORCH_HOME", "/cache/torch")) / "checkpoints"
        backbone_cache.mkdir(parents=True, exist_ok=True)
        with open_dict(cfg):
            cfg.dataset = "generic"
            cfg.weights = str(checkpoint)
            cfg.max_internal_size = 480
            cfg.model.resnet_model_path = str(backbone_cache)
        get_dataset_cfg(cfg)

        network = CUTIE(cfg).to(device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        network.load_weights(state)
        self.network = network
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _digest(checkpoint, "sha256")
        self.parameter_count = int(sum(parameter.numel() for parameter in network.parameters()))

    @staticmethod
    def _image_tensor(path: Path) -> torch.Tensor:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().div_(255.0).cuda()

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        initial = np.asarray(first_frame_mask)
        if initial.ndim != 2 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("first_frame_mask must be a 2-D integer instance mask.")
        object_ids = [int(value) for value in np.unique(initial) if value > 0]
        if not object_ids:
            raise ValueError("first_frame_mask contains no positive object IDs.")

        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(f"Cutie received {len(frames)} frames; expected {frame_count}.")

        from cutie.inference.inference_core import InferenceCore

        processor = InferenceCore(self.network, cfg=self.cfg)
        processor.max_internal_size = 480
        initial_tensor = torch.from_numpy(initial.astype(np.int64, copy=False)).cuda()
        predictions: list[np.ndarray] = []
        latencies_ms: list[float] = [0.0] * frame_count

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for index, frame_path in enumerate(frames):
                image = self._image_tensor(frame_path)
                torch.cuda.synchronize()
                started = time.perf_counter()
                if index == 0:
                    output = processor.step(image, initial_tensor, objects=object_ids)
                else:
                    output = processor.step(image, end=index == frame_count - 1)
                torch.cuda.synchronize()
                if index == 0:
                    predictions.append(initial.astype(np.int32, copy=True))
                else:
                    latencies_ms[index] = (time.perf_counter() - started) * 1000.0
                    prediction = processor.output_prob_to_mask(output)
                    predictions.append(prediction.cpu().numpy().astype(np.int32, copy=False))

        del processor
        torch.cuda.empty_cache()
        return predictions, latencies_ms
