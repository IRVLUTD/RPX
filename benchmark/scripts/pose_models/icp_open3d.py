"""Colored ICP adapter — classical RGBD pose baseline (Open3D).

Point-to-plane ICP with a colour-consistency term. Requires depth on
both views; reads them from the sample's metadata when available
(stereo / rgbd_relative_pose tasks). Translation is metric (depth-driven).

Install
-------
    pip install open3d numpy
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._pose_base import coerce_pose_output, identity_pose, validate_pair


class ColoredICP:
    """rgb+depth pair → 4×4 SE(3) via Open3D's colored-ICP."""

    native_alignment: str = "none"  # metric translation from depth
    native_precision: str = "fp64"

    def __init__(
        self,
        batch_size: int = 1,
        focal_length_px: float = 615.0,
        voxel_sizes: Sequence[float] = (0.04, 0.02, 0.01),
        max_iterations: Sequence[int] = (50, 30, 14),
    ) -> None:
        try:
            import open3d as o3d  # noqa: F401
        except ImportError as e:
            raise ImportError("ColoredICP needs Open3D. Install with: pip install open3d") from e
        self.batch_size = int(batch_size)
        self.focal_length_px = float(focal_length_px)
        self.voxel_sizes = list(voxel_sizes)
        self.max_iterations = list(max_iterations)

    @property
    def torch_module(self):
        return None  # classical → no torch module

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        outs: list[dict] = []
        for pair in pairs:
            rgb_a, rgb_b = validate_pair(pair)
            depth_a = pair.get("depth_a") or pair.get("depth")
            depth_b = pair.get("depth_b")
            if depth_a is None or depth_b is None:
                # No depth on this pair — fall back to identity (the
                # rgbd_relative_pose recipe carries depth_a / depth_b;
                # the pure relative_pose recipe doesn't, so this adapter
                # is a noop on RGB-only splits).
                outs.append(identity_pose())
                continue
            outs.append(self._solve_one(rgb_a, depth_a, rgb_b, depth_b))
        return outs

    def _solve_one(
        self,
        rgb_a: np.ndarray,
        depth_a: np.ndarray,
        rgb_b: np.ndarray,
        depth_b: np.ndarray,
    ) -> dict:
        import open3d as o3d

        H, W = rgb_a.shape[:2]
        K = o3d.camera.PinholeCameraIntrinsic(
            W,
            H,
            self.focal_length_px,
            self.focal_length_px,
            W / 2.0,
            H / 2.0,
        )
        pcd_a = self._pointcloud_from_rgbd(rgb_a, depth_a, K, o3d)
        pcd_b = self._pointcloud_from_rgbd(rgb_b, depth_b, K, o3d)
        if len(pcd_a.points) < 100 or len(pcd_b.points) < 100:
            return identity_pose()

        # Coarse-to-fine colored ICP.
        T = np.eye(4)
        for v, n_it in zip(self.voxel_sizes, self.max_iterations, strict=False):
            src = pcd_a.voxel_down_sample(v)
            dst = pcd_b.voxel_down_sample(v)
            src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
            dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
            try:
                result = o3d.pipelines.registration.registration_colored_icp(
                    src,
                    dst,
                    v,
                    T,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=n_it,
                    ),
                )
                T = result.transformation
            except Exception:
                # Coloured ICP can diverge on tiny overlap — accept the
                # current T and move on rather than crashing the run.
                break
        return coerce_pose_output(T[:3, :3], T[:3, 3])

    @staticmethod
    def _pointcloud_from_rgbd(rgb, depth, K, o3d):
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        # Open3D depth is uint16 millimetres or float metres; assume float metres.
        depth_arr = np.asarray(depth, dtype=np.float32)
        depth_o3d = o3d.geometry.Image(depth_arr)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, K)
