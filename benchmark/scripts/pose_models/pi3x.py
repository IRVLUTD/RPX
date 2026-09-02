"""Pi3X adapter using the official approximate-metric camera head."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _relative_from_camera_to_world(camera_poses: np.ndarray) -> np.ndarray:
    """Convert ordered OpenCV c2w poses into RPX's A-to-B convention."""
    poses = np.asarray(camera_poses, dtype=np.float64)
    if poses.shape != (2, 4, 4):
        raise ValueError(f"expected two 4x4 camera-to-world poses, got {poses.shape}")
    return np.linalg.inv(poses[0]) @ poses[1]


def _target_size(height: int, width: int, pixel_limit: int = 255000) -> tuple[int, int]:
    """Mirror Pi3X's official uniform resize, including its patch-14 limit."""
    scale = math.sqrt(pixel_limit / (width * height)) if width * height else 1.0
    target_w, target_h = width * scale, height * scale
    k, m = round(target_w / 14), round(target_h / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > target_w / target_h:
            k -= 1
        else:
            m -= 1
    return max(1, m) * 14, max(1, k) * 14


class Pi3X:
    """Two ordered RGB frames -> approximate-metric relative camera pose."""

    DEFAULT_MODEL_DIR = "/opt/rpx-models/pi3x/checkpoints/Pi3X"
    native_alignment: str = "none"
    native_precision: str = "bf16"

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR,
                 device: str = "cuda", batch_size: int = 1) -> None:
        source_root = Path("/opt/rpx-models/pi3x")
        value = str(source_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

        try:
            import torch
            from pi3.models.pi3x import Pi3X as OfficialPi3X
        except ImportError as exc:
            raise ImportError("Pi3X requires its official source and dependencies") from exc

        model_path = Path(model_dir)
        for required in ("config.json", "model.safetensors"):
            if not (model_path / required).is_file():
                raise FileNotFoundError(
                    f"Pi3X checkpoint file is missing: {model_path / required}"
                )

        self.model_dir = str(model_path)
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = OfficialPi3X.from_pretrained(self.model_dir)
        self._model.disable_multimodal(free_cuda_cache=False)
        self._model = self._model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def _preprocess(self, rgb_a: np.ndarray, rgb_b: np.ndarray):
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        height, width = rgb_a.shape[:2]
        target_h, target_w = _target_size(height, width)
        tensors = []
        for image in (rgb_a, rgb_b):
            resized = Image.fromarray(image).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
            tensors.append(pil_to_tensor(resized).float().div_(255.0))
        return self._torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)

    def _infer_pair(self, pair: dict) -> dict:
        rgb_a, rgb_b = validate_pair(pair)
        images = self._preprocess(rgb_a, rgb_b)

        use_cuda = str(self.device).startswith("cuda")
        if use_cuda and self._torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = self._torch.bfloat16
        else:
            amp_dtype = self._torch.float16

        with self._torch.inference_mode():
            with self._torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_cuda
            ):
                output = self._model(imgs=images)

        camera_poses = output["camera_poses"][0].float().cpu().numpy()
        relative = _relative_from_camera_to_world(camera_poses)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["Pi3X", "_relative_from_camera_to_world", "_target_size"]
