"""Fast3R adapter using the official global-pointmap plus PnP path."""

from __future__ import annotations

import sys
import tempfile
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


class Fast3R:
    """Two ordered RGB frames -> up-to-scale relative camera pose."""

    DEFAULT_MODEL_DIR = "/opt/rpx-models/fast3r/checkpoints/Fast3R_ViT_Large_512"
    native_alignment: str = "unit"
    native_precision: str = "bf16"

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR,
                 device: str = "cuda", batch_size: int = 1) -> None:
        source_root = Path("/opt/rpx-models/fast3r")
        value = str(source_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

        try:
            import torch
            from fast3r.models.fast3r import Fast3R as OfficialFast3R
        except ImportError as exc:
            raise ImportError("Fast3R requires its official source and dependencies") from exc

        model_path = Path(model_dir)
        for required in ("config.json", "model.safetensors"):
            if not (model_path / required).is_file():
                raise FileNotFoundError(
                    f"Fast3R checkpoint file is missing: {model_path / required}"
                )

        self.model_dir = str(model_path)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self._torch = torch
        self._model = OfficialFast3R.from_pretrained(self.model_dir).to(self.device).eval()

    @property
    def torch_module(self):
        return self._model

    def _estimate_poses(self, predictions: list[dict]) -> np.ndarray:
        """Run the released shared-focal, per-view PnP estimator."""
        from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
        from fast3r.dust3r.post_process import (
            estimate_focal_knowing_depth_and_confidence_mask,
        )

        first_points = predictions[0]["pts3d_in_other_view"]
        first_confidence = predictions[0]["conf"]
        _, height, width, _ = first_points.shape
        principal_point = self._torch.tensor(
            (width / 2, height / 2), device=first_points.device
        ).view(1, 2)
        threshold = self._torch.quantile(first_confidence.reshape(-1), 0.10)
        confidence_mask = (first_confidence >= threshold).view(1, height, width)
        focal = estimate_focal_knowing_depth_and_confidence_mask(
            first_points,
            principal_point.unsqueeze(0),
            confidence_mask,
            focal_mode="weiszfeld",
        ).ravel()
        shared_focal = float(focal)

        camera_poses = []
        for index, prediction in enumerate(predictions):
            points = prediction["pts3d_in_other_view"].cpu().squeeze(0)
            valid = prediction["conf"].cpu().squeeze(0) > 1.0
            _, pose_c2w = fast_pnp(
                points,
                shared_focal,
                valid,
                "cpu",
                pp=None,
                niter_PnP=100,
            )
            if pose_c2w is None:
                raise RuntimeError(f"Fast3R PnP failed for ordered view {index}")
            camera_poses.append(pose_c2w.cpu().numpy())
        return np.stack(camera_poses)

    def _infer_pair(self, pair: dict) -> dict:
        from fast3r.dust3r.inference_multiview import inference
        from fast3r.dust3r.utils.image import load_images
        from PIL import Image

        rgb_a, rgb_b = validate_pair(pair)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate((rgb_a, rgb_b)):
                path = Path(temporary) / f"{index}.png"
                Image.fromarray(image).save(path)
                paths.append(str(path))
            views = load_images(paths, size=512, verbose=False)

        output = inference(
            views,
            self._model,
            self.device,
            dtype=self._torch.bfloat16,
            verbose=False,
            profiling=False,
        )
        camera_poses = self._estimate_poses(output["preds"])
        relative = _relative_from_camera_to_world(camera_poses)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["Fast3R", "_relative_from_camera_to_world"]
