"""Official MOTIP detector-driven multi-object adapter for RPX D3."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

MOTIP_MODEL_ID = "MCG-NJU/MOTIP"
MOTIP_MODEL_REVISION = "v0.1"
MOTIP_CHECKPOINT = "r50_deformable_detr_motip_dancetrack.pth"
MOTIP_CHECKPOINT_BYTES = 709_007_050
MOTIP_CHECKPOINT_URL = (
    "https://github.com/MCG-NJU/MOTIP/releases/download/"
    f"{MOTIP_MODEL_REVISION}/{MOTIP_CHECKPOINT}"
)
MOTIP_CONFIG = "r50_deformable_detr_motip_dancetrack.yaml"
MOTIP_SOURCE_DIR = "/opt/rpx-models/motip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checkpoint(cache_root: Path) -> Path:
    destination = cache_root / "motip" / MOTIP_MODEL_REVISION / MOTIP_CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == MOTIP_CHECKPOINT_BYTES:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(MOTIP_CHECKPOINT_URL, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if temporary.stat().st_size != MOTIP_CHECKPOINT_BYTES:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("MOTIP checkpoint size differs from the official v0.1 asset.")
    temporary.replace(destination)
    return destination


class MOTIPTracker:
    """Run official MOTIP without RPX prompts and rasterize its tracked boxes."""

    model_name = "motip"
    model_id = MOTIP_MODEL_ID
    model_revision = MOTIP_MODEL_REVISION
    checkpoint_filename = MOTIP_CHECKPOINT
    config_name = MOTIP_CONFIG
    source_directory = MOTIP_SOURCE_DIR
    adapter_label = "MOTIP"
    prompt_type = "detector"
    tracking_mode = "native-joint-mot"

    def __init__(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise RuntimeError("The production MOTIP adapter requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the MOTIP adapter.")

        source_dir = Path(os.environ.get("MOTIP_SOURCE_DIR", self.source_directory)).resolve()
        config_path = source_dir / "configs" / self.config_name
        if not config_path.is_file():
            raise RuntimeError(
                f"MOTIP config is missing: {config_path}. Use the pinned cumulative image."
            )

        from configs.util import load_super_config
        from models.misc import load_checkpoint
        from models.motip import build as build_model
        from utils.misc import yaml_to_dict

        config = yaml_to_dict(str(config_path))
        config = load_super_config(config, config["SUPER_CONFIG_PATH"])
        checkpoint = _download_checkpoint(
            Path(os.environ.get("HF_HOME", "/cache/huggingface"))
        )
        model, _ = build_model(config)
        load_checkpoint(model, str(checkpoint))
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    @staticmethod
    def _image_tensor(path: Path) -> torch.Tensor:
        from torchvision.transforms import functional as functional

        image = Image.open(path).convert("RGB")
        tensor = functional.to_tensor(image)
        tensor = functional.resize(tensor, size=800, max_size=1440)
        tensor = functional.normalize(
            tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return tensor.cuda()

    @staticmethod
    def _rasterize_boxes(
        boxes_xywh: np.ndarray,
        track_ids: np.ndarray,
        scores: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = shape
        output = np.zeros(shape, dtype=np.int32)
        # Higher-confidence detections win where detector boxes overlap.
        for index in np.argsort(scores):
            x, y, box_width, box_height = boxes_xywh[index]
            x1 = max(0, min(width, int(np.floor(x))))
            y1 = max(0, min(height, int(np.floor(y))))
            x2 = max(0, min(width, int(np.ceil(x + box_width))))
            y2 = max(0, min(height, int(np.ceil(y + box_height))))
            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = int(track_ids[index]) + 1
        return output

    def track(
        self,
        video_dir: Path,
        first_frame_mask: np.ndarray,
        frame_count: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        # Detector-driven MOT deliberately ignores the RPX first-frame annotation.
        initial = np.asarray(first_frame_mask)
        if initial.ndim != 2:
            raise ValueError("first_frame_mask must be a 2-D instance mask.")
        frames = sorted(video_dir.glob("*.jpg"))
        if len(frames) != frame_count:
            raise RuntimeError(f"MOTIP received {len(frames)} frames; expected {frame_count}.")

        from models.runtime_tracker import RuntimeTracker
        from utils.nested_tensor import nested_tensor_from_tensor_list

        runtime = RuntimeTracker(
            model=self.model,
            sequence_hw=initial.shape,
            use_sigmoid=self.config.get("USE_FOCAL_LOSS", False),
            assignment_protocol=self.config.get("ASSIGNMENT_PROTOCOL", "object-max"),
            miss_tolerance=self.config["MISS_TOLERANCE"],
            det_thresh=self.config["DET_THRESH"],
            newborn_thresh=self.config["NEWBORN_THRESH"],
            id_thresh=self.config["ID_THRESH"],
            area_thresh=self.config.get("AREA_THRESH", 0),
            only_detr=False,
            dtype=torch.float32,
        )
        predictions = []
        latencies_ms = []
        with torch.inference_mode():
            for frame_path in frames:
                image = nested_tensor_from_tensor_list([self._image_tensor(frame_path)])
                torch.cuda.synchronize()
                started = time.perf_counter()
                runtime.update(image)
                torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                result = runtime.get_track_results()
                predictions.append(
                    self._rasterize_boxes(
                        result["bbox"].detach().float().cpu().numpy(),
                        result["id"].detach().cpu().numpy(),
                        result["score"].detach().float().cpu().numpy(),
                        initial.shape,
                    )
                )

        del runtime
        torch.cuda.empty_cache()
        return predictions, latencies_ms
