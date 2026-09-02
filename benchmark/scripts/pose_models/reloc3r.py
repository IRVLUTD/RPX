"""Reloc3r adapter using the official 512px relative-pose model."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _rpx_from_pose2to1(pose: np.ndarray) -> np.ndarray:
    """Validate the native pose2-to-1 transform used by RPX unchanged."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose2to1 transform, got {value.shape}")
    return value


class Reloc3r:
    """Two ordered RGB frames -> scale-invariant relative camera pose."""

    DEFAULT_MODEL_DIR = "/opt/rpx-models/reloc3r/checkpoints/reloc3r-512"
    native_alignment: str = "unit"
    native_precision: str = "fp16"

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR,
                 device: str = "cuda", batch_size: int = 1) -> None:
        reloc3r_root = Path("/opt/rpx-models/reloc3r")
        value = str(reloc3r_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

        try:
            import torch
            from reloc3r.reloc3r_relpose import Reloc3rRelpose
        except ImportError as exc:
            raise ImportError(
                "Reloc3r requires its official source and pinned CroCo submodule"
            ) from exc

        model_path = Path(model_dir)
        for required in ("config.json", "model.safetensors"):
            if not (model_path / required).is_file():
                raise FileNotFoundError(
                    f"Reloc3r checkpoint file is missing: {model_path / required}"
                )
        self.model_dir = str(model_path)
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = Reloc3rRelpose.from_pretrained(self.model_dir).to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def _infer_pair(self, pair: dict) -> dict:
        from PIL import Image
        from reloc3r.reloc3r_relpose import inference_relpose
        from reloc3r.utils.image import check_images_shape_format, load_images

        rgb_a, rgb_b = validate_pair(pair)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            images = load_images(paths, size=512, verbose=False)

        images = check_images_shape_format(images, self.device)
        pose2to1 = inference_relpose(
            [images[0], images[1]],
            self._model,
            self.device,
            use_amp=str(self.device).startswith("cuda"),
        )[0].detach().cpu().numpy()
        relative = _rpx_from_pose2to1(pose2to1)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["Reloc3r", "_rpx_from_pose2to1"]
