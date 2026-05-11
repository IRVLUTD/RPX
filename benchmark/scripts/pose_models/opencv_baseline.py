"""OpenCV essential-matrix baseline — classical no-learning pose floor.

SIFT keypoints + ratio-test matching + ``findEssentialMat`` (USAC-MAGSAC)
+ ``recoverPose``. The "no learning" floor every learned pose model
should beat. Translation is up-to-scale (essential-matrix
decomposition).

Install
-------
    pip install opencv-contrib-python

(``opencv-contrib`` is needed for SIFT; the base ``opencv-python``
ships ORB only.)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, identity_pose, validate_pair


class OpenCVEssentialMat:
    """rgb pair → essential-matrix-recovered pose. No torch dependency."""

    native_alignment: str = "unit"  # E-matrix decomp → unit translation
    native_precision: str = "fp32"

    def __init__(
        self,
        batch_size: int = 1,
        focal_length_px: float = 615.0,
        feature: str = "sift",  # | "orb"
        ratio: float = 0.75,
        ransac_threshold: float = 1.0,
    ) -> None:
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCVEssentialMat needs opencv. Install with: pip install opencv-contrib-python"
            ) from e
        self.batch_size = int(batch_size)
        self.focal_length_px = float(focal_length_px)
        self.ratio = float(ratio)
        self.ransac_threshold = float(ransac_threshold)
        if feature == "sift":
            try:
                self._detector = cv2.SIFT_create()
            except AttributeError as e:
                raise ImportError(
                    "SIFT requires opencv-contrib-python (not opencv-python). "
                    "Install with: pip install opencv-contrib-python"
                ) from e
            self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif feature == "orb":
            self._detector = cv2.ORB_create(nfeatures=4000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            from rpx_benchmark.exceptions import ConfigError

            raise ConfigError(
                f"unknown feature {feature!r}",
                hint="Expected 'sift' or 'orb'.",
            )

    @property
    def torch_module(self):
        return None  # no learning → no torch module to profile

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:

        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        outs: list[dict] = []
        for pair in pairs:
            rgb_a, rgb_b = validate_pair(pair)
            outs.append(self._solve_one(rgb_a, rgb_b))
        return outs

    def _solve_one(self, rgb_a: np.ndarray, rgb_b: np.ndarray) -> dict:
        import cv2

        gray_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
        kp_a, des_a = self._detector.detectAndCompute(gray_a, None)
        kp_b, des_b = self._detector.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return identity_pose()

        # Lowe's ratio test.
        matches = self._matcher.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in matches if m.distance < self.ratio * n.distance]
        if len(good) < 8:
            return identity_pose()
        pts_a = np.array([kp_a[m.queryIdx].pt for m in good], dtype=np.float64)
        pts_b = np.array([kp_b[m.trainIdx].pt for m in good], dtype=np.float64)

        H, W = rgb_a.shape[:2]
        K = np.array(
            [
                [self.focal_length_px, 0, W / 2.0],
                [0, self.focal_length_px, H / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        E, mask = cv2.findEssentialMat(
            pts_a,
            pts_b,
            K,
            method=cv2.USAC_MAGSAC,
            prob=0.99999,
            threshold=self.ransac_threshold,
        )
        if E is None or E.shape != (3, 3):
            return identity_pose()
        _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
        return coerce_pose_output(R, t.reshape(3))
