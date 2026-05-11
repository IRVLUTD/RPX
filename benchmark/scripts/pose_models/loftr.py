"""LoFTR adapter — dense matching + 5-point pose solver (CVPR 2021).

LoFTR predicts dense correspondences; relative pose comes from
running OpenCV's ``findEssentialMat`` + ``recoverPose`` on those
correspondences. Translation is up-to-scale (essential-matrix
decomposition gives a unit translation), so this adapter declares
``native_alignment="unit"`` — the runner / metrics compare
translations as unit vectors only (translation_angular_deg).

Upstream: github.com/zju3dv/LoFTR. Hosted weights available via
kornia.feature.LoFTR with ``pretrained="indoor"`` / ``"outdoor"``.

Install
-------
    pip install kornia opencv-python torch pillow
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._pose_base import coerce_pose_output, identity_pose, validate_pair


class LoFTR:
    DEFAULT_PRETRAINED = "indoor"  # | "outdoor"

    native_alignment: str = "unit"  # essential-matrix decomp → unit translation
    native_precision: str = "fp16"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        pretrained: str = DEFAULT_PRETRAINED,
        focal_length_px: float = 615.0,  # D435 RGB factory cal
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import cv2  # noqa: F401
            import torch
            from kornia.feature import LoFTR as _LoFTRMatcher
        except ImportError as e:
            raise ImportError(
                "LoFTR needs kornia + opencv-python. Install with: "
                "pip install kornia opencv-python torch pillow"
            ) from e
        self.device = device
        self.batch_size = int(batch_size)
        self.focal_length_px = float(focal_length_px)
        self._torch = torch

        try:
            matcher = _LoFTRMatcher(pretrained=pretrained)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"LoFTR matcher load failed for pretrained={pretrained!r}: {e}",
                hint="Try pretrained='outdoor' or upgrade kornia.",
            ) from e
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            matcher = matcher.to(dtype=target_dtype)
        self._matcher = matcher.to(device).eval()

    @property
    def torch_module(self):
        return self._matcher

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        torch = self._torch
        import cv2

        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        outs: list[dict] = []
        with torch.inference_mode():
            for pair in pairs:
                rgb_a, rgb_b = validate_pair(pair)
                try:
                    # LoFTR expects grayscale [B, 1, H, W] in [0, 1].
                    gray_a = _to_gray_tensor(rgb_a, self.device, torch)
                    gray_b = _to_gray_tensor(rgb_b, self.device, torch)
                    matches = self._matcher({"image0": gray_a, "image1": gray_b})
                    kpts_a = matches["keypoints0"].detach().cpu().numpy()
                    kpts_b = matches["keypoints1"].detach().cpu().numpy()
                    if len(kpts_a) < 8:
                        outs.append(identity_pose())
                        continue

                    # Camera intrinsics — assume single-camera dataset (D435).
                    H, W = rgb_a.shape[:2]
                    K = np.array(
                        [
                            [self.focal_length_px, 0, W / 2],
                            [0, self.focal_length_px, H / 2],
                            [0, 0, 1],
                        ],
                        dtype=np.float64,
                    )
                    E, mask = cv2.findEssentialMat(
                        kpts_a,
                        kpts_b,
                        K,
                        method=cv2.USAC_MAGSAC,
                        prob=0.99999,
                        threshold=1.0,
                    )
                    if E is None or E.shape != (3, 3):
                        outs.append(identity_pose())
                        continue
                    _, R, t, _ = cv2.recoverPose(E, kpts_a, kpts_b, K, mask=mask)
                    outs.append(coerce_pose_output(R, t.reshape(3)))
                except Exception as e:
                    from rpx_benchmark.exceptions import AdapterError

                    raise AdapterError(
                        f"LoFTR inference failed on a pair: {e}",
                    ) from e
        return outs


def _to_gray_tensor(rgb: np.ndarray, device: str, torch) -> "torch.Tensor":  # noqa: F821
    """RGB uint8 (H, W, 3) → grayscale float [B=1, 1, H, W] in [0, 1]."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
