"""FAR adapter — Hybrid Regression + Matching (CVPR 2024, Niantic Labs).

FAR was the SOTA on Niantic's MapFree benchmark at release. Combines
direct pose regression with feature-matching solver in one network.

Upstream: github.com/crockwell/far. Weights are released alongside the
repo (no HF mirror); FAR ships its own loader entry point.

Install
-------
    pip install far-pose torch pillow            # if released on PyPI
    # or:  pip install git+https://github.com/crockwell/far
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._pose_base import coerce_pose_output, validate_pair


class FAR:
    """rgb pair → 4×4 SE(3) via hybrid regression + matching."""

    native_alignment: str = "none"
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
            from far.model import FARModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "FAR needs the `far` package. Install with: "
                "pip install git+https://github.com/crockwell/far"
            ) from e
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch

        try:
            if weights_path is None:
                model = FARModel.from_pretrained()
            else:
                model = FARModel.from_pretrained(weights_path)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"FAR load failed: {e}",
                hint="Pass `weights_path=...` if the default download URL "
                "moved. Repo: https://github.com/crockwell/far.",
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
                        f"FAR inference failed on a pair: {e}",
                    ) from e
        return outs
