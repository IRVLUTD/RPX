"""MicKey adapter — Metric Keypoints (CVPR 2024 Oral, Niantic).

Predicts 3D keypoint positions directly, gives metric-scale relative
pose without depth input.

Upstream: github.com/nianticlabs/mickey.

Install
-------
    pip install git+https://github.com/nianticlabs/mickey
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._pose_base import coerce_pose_output, validate_pair


class MicKey:
    native_alignment: str = "none"  # metric translation by construction
    native_precision: str = "fp32"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        weights_path: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from mickey.model import MicKeyModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "MicKey needs the `mickey` package. Install with: "
                "pip install git+https://github.com/nianticlabs/mickey"
            ) from e
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch

        try:
            model = (
                MicKeyModel.from_pretrained(weights_path)
                if weights_path
                else MicKeyModel.from_pretrained()
            )
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"MicKey load failed: {e}",
                hint="Pass `weights_path=...` to point at a local checkpoint.",
            ) from e
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            model = model.to(dtype=target_dtype)
        self._model = model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        torch = self._torch
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        outs: list[dict] = []
        with torch.inference_mode():
            for pair in pairs:
                rgb_a, rgb_b = validate_pair(pair)
                try:
                    pose = self._model.infer_pair(rgb_a, rgb_b)
                    rot = pose["rotation"] if isinstance(pose, dict) else pose[0]
                    trans = pose["translation"] if isinstance(pose, dict) else pose[1]
                    outs.append(coerce_pose_output(rot, trans))
                except Exception as e:
                    from rpx_benchmark.exceptions import AdapterError

                    raise AdapterError(
                        f"MicKey inference failed on a pair: {e}",
                    ) from e
        return outs
