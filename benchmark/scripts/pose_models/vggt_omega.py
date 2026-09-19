"""VGGT-Ω (official VGGT-1B) relative-camera-pose adapter.

VGGT returns OpenCV world-to-camera extrinsics. RPX stores camera-to-world
poses and defines frame B relative to frame A as ``inv(T_a) @ T_b``. The
conversion below makes the two conventions explicit.
"""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _relative_from_world_to_camera(extrinsics: np.ndarray) -> np.ndarray:
    """Return RPX's A^-1 B transform from two world-to-camera matrices."""
    ext = np.asarray(extrinsics, dtype=np.float64)
    if ext.shape != (2, 3, 4):
        raise ValueError(f"expected two 3x4 extrinsics, got {ext.shape}")
    w2c = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    w2c[:, :3, :4] = ext
    c2w = np.linalg.inv(w2c)
    return np.linalg.inv(c2w[0]) @ c2w[1]


class VGGTOmega:
    """Two RGB views -> relative rotation and up-to-scale translation."""

    DEFAULT_MODEL_ID = "facebook/VGGT-1B"
    native_alignment: str = "unit"
    native_precision: str = "bf16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        try:
            import torch
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "VGGT-Ω requires the official facebookresearch/vggt package"
            ) from exc

        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = VGGT.from_pretrained(model_id).to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def _infer_pair(self, pair: dict) -> dict:
        from PIL import Image
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        rgb_a, rgb_b = validate_pair(pair)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            images = load_and_preprocess_images(paths).to(self.device)

        torch = self._torch
        if self.device.startswith("cuda"):
            device_index = int(self.device.split(":")[-1]) if ":" in self.device else 0
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(device_index)[0] >= 8
                else torch.float16
            )
            autocast = torch.amp.autocast("cuda", dtype=dtype)
        else:
            autocast = nullcontext()

        with torch.inference_mode(), autocast:
            predictions = self._model(images)
            pose_encoding = predictions["pose_enc"]
            extrinsics, _ = pose_encoding_to_extri_intri(
                pose_encoding, images.shape[-2:], build_intrinsics=False
            )

        ext = extrinsics.detach().cpu().float().numpy()
        if ext.ndim == 4 and ext.shape[0] == 1:
            ext = ext[0]
        relative = _relative_from_world_to_camera(ext)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["VGGTOmega", "_relative_from_world_to_camera"]
