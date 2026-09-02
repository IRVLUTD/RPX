"""DUSt3R adapter using the official symmetric two-view PairViewer path."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _relative_from_camera_to_world(poses: np.ndarray) -> np.ndarray:
    """Convert two OpenCV camera-to-world poses to RPX's A-to-B pose."""
    c2w = np.asarray(poses, dtype=np.float64)
    if c2w.shape != (2, 4, 4):
        raise ValueError(f"expected two 4x4 camera-to-world poses, got {c2w.shape}")
    return np.linalg.inv(c2w[0]) @ c2w[1]


class DUSt3R:
    """Two ordered RGB frames -> scale-invariant relative camera pose."""

    DEFAULT_CHECKPOINT = (
        "/opt/rpx-models/dust3r/checkpoints/"
        "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    )
    native_alignment: str = "unit"
    native_precision: str = "fp32"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT,
                 device: str = "cuda", batch_size: int = 1) -> None:
        dust3r_root = Path("/opt/rpx-models/dust3r")
        # Select standalone DUSt3R ahead of incompatible forks inherited from
        # CUT3R, MASt3R and MUSt3R in the cumulative image.
        value = str(dust3r_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

        try:
            import torch
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as exc:
            raise ImportError(
                "DUSt3R requires its official source and recursive CroCo submodule"
            ) from exc

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DUSt3R checkpoint is missing: {checkpoint_path}")
        self.checkpoint = str(checkpoint_path)
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = (AsymmetricCroCo3DStereo.from_pretrained(self.checkpoint)
                       .to(device).eval())

    @property
    def torch_module(self):
        return self._model

    def _infer_pair(self, pair: dict) -> dict:
        from PIL import Image
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        rgb_a, rgb_b = validate_pair(pair)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            images = load_images(paths, size=512, verbose=False)

        image_pairs = make_pairs(images, scene_graph="complete", prefilter=None,
                                 symmetrize=True)
        output = inference(image_pairs, self._model, self.device,
                           batch_size=self.batch_size, verbose=False)
        # Official DUSt3R uses PairViewer for one/two-image reconstruction.
        scene = global_aligner(output, device=self.device,
                               mode=GlobalAlignerMode.PairViewer, verbose=False)
        poses = scene.get_im_poses().detach().cpu().numpy()
        relative = _relative_from_camera_to_world(poses)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["DUSt3R", "_relative_from_camera_to_world"]
