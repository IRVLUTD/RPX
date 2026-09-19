"""MUSt3R adapter using the official two-view memory inference path."""

from __future__ import annotations

import sys
import tempfile
from contextlib import nullcontext
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


class MUSt3R:
    """Two ordered RGB frames -> scale-invariant relative camera pose."""

    DEFAULT_CHECKPOINT = "/opt/rpx-models/must3r/checkpoints/MUSt3R_512.pth"
    native_alignment: str = "unit"
    native_precision: str = "bf16"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        must3r_root = Path("/opt/rpx-models/must3r")
        # Cumulative images contain other, mutually incompatible DUSt3R forks.
        for path in (must3r_root, must3r_root / "dust3r"):
            value = str(path)
            if value in sys.path:
                sys.path.remove(value)
            sys.path.insert(0, value)

        try:
            import torch
            from must3r.model import get_pointmaps_activation, load_model
        except ImportError as exc:
            raise ImportError(
                "MUSt3R requires its official source and recursive DUSt3R/CroCo submodules"
            ) from exc

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MUSt3R checkpoint is missing: {checkpoint_path}")

        self.checkpoint = str(checkpoint_path)
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        self._encoder, self._decoder = load_model(
            self.checkpoint, device=device, verbose=False
        )
        self._pointmaps_activation = get_pointmaps_activation(
            self._decoder, verbose=False
        )

    @property
    def torch_module(self):
        # Both halves are registered torch modules; expose the pair for callers
        # that understand ModuleList while retaining all parameters.
        return self._torch.nn.ModuleList([self._encoder, self._decoder])

    def _infer_pair(self, pair: dict) -> dict:
        from dust3r.utils.image import load_images
        from must3r.engine.inference import inference, postprocess
        from PIL import Image

        rgb_a, rgb_b = validate_pair(pair)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            views = load_images(paths, size=512, verbose=False)

        torch = self._torch
        imgs = torch.stack([view["img"] for view in views], dim=1).to(self.device)
        true_shape = torch.stack(
            [torch.from_numpy(view["true_shape"]) for view in views], dim=1
        ).to(self.device)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            _, pointmaps = inference(
                self._encoder,
                self._decoder,
                imgs,
                true_shape,
                mem_batches=[2],
                verbose=False,
                max_bs=2,
            )
        prediction = postprocess(
            pointmaps,
            pointmaps_activation=self._pointmaps_activation,
            compute_cam=True,
        )
        poses = prediction["c2w"][0].detach().cpu().numpy()
        relative = _relative_from_camera_to_world(poses)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["MUSt3R", "_relative_from_camera_to_world"]
