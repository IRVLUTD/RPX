"""CUT3R relative-camera-pose adapter using the official final checkpoint.

CUT3R predicts OpenCV camera-to-world poses for an ordered image stream.
For the RPX pair protocol we reset its persistent state for every ordered
``(A, B)`` pair and return ``inv(T_c2w_A) @ T_c2w_B``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _relative_from_camera_to_world(poses: np.ndarray) -> np.ndarray:
    """Convert two 4x4 camera-to-world poses to RPX's A-to-B pose."""
    c2w = np.asarray(poses, dtype=np.float64)
    if c2w.shape != (2, 4, 4):
        raise ValueError(f"expected two 4x4 camera-to-world poses, got {c2w.shape}")
    return np.linalg.inv(c2w[0]) @ c2w[1]


class CUT3R:
    """Two ordered RGB frames -> metric relative camera pose."""

    DEFAULT_CHECKPOINT = "/opt/rpx-models/cut3r/src/cut3r_512_dpt_4_64.pth"
    native_alignment: str = "none"
    native_precision: str = "fp32"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        try:
            import torch
            from dust3r.model import ARCroco3DStereo
        except ImportError as exc:
            raise ImportError(
                "CUT3R requires the official CUT3R source tree and compiled cuRoPE"
            ) from exc

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CUT3R checkpoint is missing: {checkpoint_path}")

        self.checkpoint = str(checkpoint_path)
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = ARCroco3DStereo.from_pretrained(self.checkpoint).to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def _infer_pair(self, pair: dict) -> dict:
        from PIL import Image
        from dust3r.inference import inference
        from dust3r.utils.camera import pose_encoding_to_camera
        from dust3r.utils.image import load_images

        rgb_a, rgb_b = validate_pair(pair)
        torch = self._torch

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            images = load_images(paths, size=512, verbose=False)

        views = []
        for index, image in enumerate(images):
            views.append(
                {
                    "img": image["img"],
                    "ray_map": torch.full(
                        (image["img"].shape[0], 6, *image["img"].shape[-2:]),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(image["true_shape"]),
                    "idx": index,
                    "instance": str(index),
                    "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    # Each adapter call starts a fresh CUT3R state already;
                    # match the official two-image inference recipe exactly.
                    "reset": torch.tensor(False).unsqueeze(0),
                }
            )

        outputs, _ = inference(views, self._model, self.device, verbose=False)
        predictions = outputs["pred"]
        if len(predictions) != 2:
            raise RuntimeError(f"CUT3R returned {len(predictions)} poses for two images")
        poses = np.stack(
            [
                pose_encoding_to_camera(prediction["camera_pose"].clone())[0]
                .detach()
                .cpu()
                .numpy()
                for prediction in predictions
            ]
        )
        relative = _relative_from_camera_to_world(poses)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["CUT3R", "_relative_from_camera_to_world"]
