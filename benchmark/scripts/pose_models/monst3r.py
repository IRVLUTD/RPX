"""MonST3R adapter using the released symmetric two-view PairViewer path."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _relative_from_camera_to_world(poses: np.ndarray) -> np.ndarray:
    """Convert ordered OpenCV camera-to-world poses into RPX A-to-B pose."""
    c2w = np.asarray(poses, dtype=np.float64)
    if c2w.shape != (2, 4, 4):
        raise ValueError(f"expected two 4x4 camera-to-world poses, got {c2w.shape}")
    return np.linalg.inv(c2w[0]) @ c2w[1]


class MonST3R:
    """Two ordered RGB frames -> scale-invariant relative camera pose."""

    DEFAULT_MODEL_DIR = (
        "/opt/rpx-models/monst3r/checkpoints/"
        "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"
    )
    native_alignment: str = "unit"
    native_precision: str = "fp32"

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR,
                 device: str = "cuda", batch_size: int = 1) -> None:
        source_root = Path("/opt/rpx-models/monst3r")
        value = str(source_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

        try:
            import torch
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as exc:
            raise ImportError(
                "MonST3R requires its official source and CroCo submodule"
            ) from exc

        checkpoint = Path(model_dir)
        for required in ("config.json", "model.safetensors"):
            if not (checkpoint / required).is_file():
                raise FileNotFoundError(
                    f"MonST3R checkpoint file is missing: {checkpoint / required}"
                )

        self.model_dir = str(checkpoint)
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = (AsymmetricCroCo3DStereo.from_pretrained(self.model_dir)
                       .to(device).eval())

    @property
    def torch_module(self):
        return self._model

    def _infer_pair(self, pair: dict) -> dict:
        from dust3r.cloud_opt.pair_viewer import PairViewer
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from PIL import Image

        rgb_a, rgb_b = validate_pair(pair)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            views = load_images(paths, size=512, verbose=False)

        pairs = make_pairs(
            views, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output = inference(
            pairs, self._model, self.device,
            batch_size=self.batch_size, verbose=False,
        )
        scene = PairViewer(
            output["view1"], output["view2"],
            output["pred1"], output["pred2"], verbose=False,
        ).to(self.device)
        poses = scene.get_im_poses().detach().cpu().numpy()
        relative = _relative_from_camera_to_world(poses)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["MonST3R", "_relative_from_camera_to_world"]
