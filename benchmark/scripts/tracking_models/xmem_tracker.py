"""Official XMem long-term multi-object VOS adapter for the RPX D3 protocol."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

XMEM_MODEL_ID = "hkchengrex/XMem"
XMEM_SOURCE_REVISION = "f3b841d50df058910bbf690229ddc15fb1aef7d6"
XMEM_MODEL_REVISION = "v1.0"
XMEM_CHECKPOINT = "XMem.pth"
XMEM_CHECKPOINT_BYTES = 249_026_057
XMEM_CHECKPOINT_SHA256 = (
    "27776291d2f0639b4e6b372a67651579b51180aa4d8f8f89bbff2dcc09ebf6f9"
)
XMEM_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/XMem/releases/download/"
    f"{XMEM_MODEL_REVISION}/{XMEM_CHECKPOINT}"
)
XMEM_INPUT_SIZE = 480


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == XMEM_CHECKPOINT_BYTES
        and _sha256(path) == XMEM_CHECKPOINT_SHA256
    )


def _download_checkpoint(cache_root: Path) -> Path:
    destination = cache_root / "xmem" / XMEM_MODEL_REVISION / XMEM_CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_checkpoint(destination):
        return destination

    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(
            XMEM_CHECKPOINT_URL, stream=True, timeout=(10, 120)
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not _valid_checkpoint(temporary):
            raise RuntimeError(
                "XMem checkpoint failed its pinned size/SHA-256 integrity check."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class XMemTracker:
    """Run official XMem from a multi-object first-frame instance mask."""

    model_name = "xmem"
    model_id = XMEM_MODEL_ID
    model_revision = XMEM_MODEL_REVISION
    source_revision = XMEM_SOURCE_REVISION
    checkpoint_filename = XMEM_CHECKPOINT
    adapter_label = "XMem"
    prompt_type = "mask"
    tracking_mode = "multi-object-long-term-memory-vos"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production XMem adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the XMem adapter.")

        from model.network import XMem

        self.config = {
            "enable_long_term": True,
            "max_mid_term_frames": 10,
            "min_mid_term_frames": 5,
            "max_long_term_elements": 10_000,
            "num_prototypes": 128,
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
        }
        checkpoint = _download_checkpoint(
            Path(os.environ.get("HF_HOME", "/cache/huggingface"))
        )
        network = XMem(self.config, str(checkpoint), map_location="cpu").cuda().eval()

        self.network = network
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.parameter_count = int(
            sum(parameter.numel() for parameter in network.parameters())
        )

    @staticmethod
    def _target_size(height: int, width: int) -> tuple[int, int]:
        short_edge = min(height, width)
        return (
            int(height / short_edge * XMEM_INPUT_SIZE),
            int(width / short_edge * XMEM_INPUT_SIZE),
        )

    @classmethod
    def _image_tensor(cls, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        image = Image.open(path).convert("RGB")
        original_size = (image.height, image.width)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        tensor = TF.resize(
            tensor,
            cls._target_size(*original_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return tensor.cuda(non_blocking=True), original_size

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
            raise RuntimeError(f"XMem received {len(frames)} frames; expected {frame_count}.")

        from inference.inference_core import InferenceCore

        config = dict(self.config)
        config["enable_long_term_count_usage"] = (
            frame_count
            / (config["max_mid_term_frames"] - config["min_mid_term_frames"])
            * config["num_prototypes"]
            >= config["max_long_term_elements"]
        )
        processor = InferenceCore(self.network, config=config)
        local_labels = list(range(1, len(object_ids) + 1))
        processor.set_all_labels(local_labels)

        one_hot = torch.stack(
            [torch.from_numpy(initial == object_id) for object_id in object_ids]
        ).float()
        predictions: list[np.ndarray] = []
        latencies_ms: list[float] = [0.0] * frame_count

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for index, frame_path in enumerate(frames):
                image, original_size = self._image_tensor(frame_path)
                prompt = None
                valid_labels = None
                if index == 0:
                    prompt = F.interpolate(
                        one_hot.unsqueeze(0),
                        size=image.shape[-2:],
                        mode="nearest",
                    )[0].cuda(non_blocking=True)
                    valid_labels = local_labels

                torch.cuda.synchronize()
                started = time.perf_counter()
                probability = processor.step(
                    image,
                    prompt,
                    valid_labels,
                    end=index == frame_count - 1,
                )
                if probability.shape[-2:] != original_size:
                    probability = F.interpolate(
                        probability.unsqueeze(1),
                        size=original_size,
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                torch.cuda.synchronize()

                if index == 0:
                    predictions.append(initial.astype(np.int32, copy=True))
                    continue

                latencies_ms[index] = (time.perf_counter() - started) * 1000.0
                local = probability.argmax(dim=0).cpu().numpy()
                output = np.zeros(initial.shape, dtype=np.int32)
                for local_id, original_id in enumerate(object_ids, start=1):
                    output[local == local_id] = original_id
                predictions.append(output)

        del processor
        torch.cuda.empty_cache()
        return predictions, latencies_ms
