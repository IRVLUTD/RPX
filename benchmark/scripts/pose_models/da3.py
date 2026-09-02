"""Depth Anything 3 relative-camera-pose adapter.

The official DA3 camera decoder returns OpenCV world-to-camera extrinsics.
RPX evaluates frame B relative to frame A as ``inv(T_c2w_a) @ T_c2w_b``.
DA3's translation is only determined up to scale, so the benchmark reports
translation-direction errors and suppresses metric-translation metrics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


def _relative_from_world_to_camera(extrinsics: np.ndarray) -> np.ndarray:
    """Convert two OpenCV world-to-camera extrinsics to RPX A-to-B pose."""
    ext = np.asarray(extrinsics, dtype=np.float64)
    if ext.shape == (2, 4, 4):
        w2c = ext
    elif ext.shape == (2, 3, 4):
        w2c = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        w2c[:, :3, :4] = ext
    else:
        raise ValueError(f"expected two 3x4 or 4x4 extrinsics, got {ext.shape}")
    c2w = np.linalg.inv(w2c)
    return np.linalg.inv(c2w[0]) @ c2w[1]


class DA3:
    """Two RGB views -> DA3-GIANT relative pose using its camera decoder."""

    DEFAULT_MODEL_ID = "depth-anything/DA3-GIANT"
    native_alignment: str = "unit"
    native_precision: str = "bf16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "DA3 requires the official ByteDance-Seed/Depth-Anything-3 package"
            ) from exc

        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._model = DepthAnything3.from_pretrained(model_id).to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def _infer_pair(self, pair: dict) -> dict:
        rgb_a, rgb_b = validate_pair(pair)
        prediction = self._model.inference(
            image=[rgb_a, rgb_b],
            use_ray_pose=False,
            ref_view_strategy="first",
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        if prediction.extrinsics is None:
            raise RuntimeError("DA3 inference returned no camera extrinsics")
        relative = _relative_from_world_to_camera(prediction.extrinsics)
        return coerce_pose_output(relative[:3, :3], relative[:3, 3])

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        return [self._infer_pair(pair) for pair in pairs]


__all__ = ["DA3", "_relative_from_world_to_camera"]
