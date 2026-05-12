"""Trivial NVS baseline — return the closest context view as the rendering.

Picks the context view whose camera-to-world translation is closest (L2)
to the target's, and returns it verbatim along with its depth map. Any
real NVS model must beat this on PSNR/SSIM/depth-AbsRel — if it doesn't,
the rendering pipeline is broken or the model is worse than a
**single-frame lookup**.

Useful as:
1. A smoke fixture for ``run_nvs.py`` (zero external dependencies).
2. A floor baseline in the paper table (the bar real models must clear).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class IdentityPassthroughNVS:
    """Closest-pose passthrough NVS baseline.

    Attributes
    ----------
    name : str
        Registry key.
    native_precision : str
        ``"fp32"`` — there is no learned forward pass; this attribute
        only exists so the runner can populate ``OperatingPoint.precision``
        uniformly across all adapters.
    torch_module : None
        No torch module — params/FLOPs are reported as zero.
    """

    name = "identity_passthrough"
    native_precision = "fp32"
    torch_module = None  # no learned weights

    def __init__(self, device: str = "cpu", **_kwargs: Any) -> None:
        # device is accepted for API uniformity but does nothing here.
        self.device = device

    def __call__(
        self,
        context_rgbs:   "List[np.ndarray[Any, Any]]",   # list of (H, W, 3) uint8
        context_depths: "List[np.ndarray[Any, Any]]",   # list of (H, W) float32 in metres
        context_poses:  "List[np.ndarray[Any, Any]]",   # list of (4, 4) float64, cam-to-world
        target_pose:    "np.ndarray[Any, Any]",         # (4, 4) float64
    ) -> Dict[str, Any]:
        """Return the closest context view as the rendered output.

        Parameters
        ----------
        context_rgbs, context_depths, context_poses : list
            Lists of equal length. Empty `context_depths` is tolerated;
            in that case the returned ``depth`` is ``None``.
        target_pose : (4, 4) array
            Query camera-to-world transform.

        Returns
        -------
        dict
            ``{"rgb": (H, W, 3) uint8, "depth": (H, W) float32 | None}``.
        """
        if not context_rgbs:
            raise ValueError("identity_passthrough received zero context views")

        t_target = target_pose[:3, 3]
        dists = [float(np.linalg.norm(p[:3, 3] - t_target)) for p in context_poses]
        best = int(np.argmin(dists))

        rendered_rgb = context_rgbs[best]
        rendered_depth = (
            context_depths[best]
            if context_depths and len(context_depths) > best
            else None
        )
        return {"rgb": rendered_rgb, "depth": rendered_depth}
