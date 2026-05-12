"""Splatter Image (CVPR 2024) — single-view feed-forward 3DGS adapter for RPX.

Upstream: https://github.com/szymanowiczs/splatter-image
Paper:    Szymanowicz et al., *Splatter Image: Ultra-Fast Single-View 3D
          Reconstruction*, CVPR 2024.
Weights:  HuggingFace Hub `szymanowiczs/splatter-image-multi-category-v1`
          (auto-downloaded on first use).

Status
------
**Scaffolded, not end-to-end smoke-validated.** This file follows the
upstream's documented inference pipeline (`gradio_app.py` + `eval.py`)
faithfully — `GaussianSplatPredictor` construct + `render_predicted`
render — but it cannot be executed without the upstream's `scene/`,
`gaussian_renderer/`, and `utils/` packages on PYTHONPATH. The first
contributor with the upstream installed should run a smoke and
confirm the three TODOs flagged inline below (pose convention, depth
output, intrinsics).

Single-view limitation
----------------------
Splatter Image takes **one** input image and predicts a 3D Gaussian
representation. The RPX NVS contract gives K context views; this
adapter reduces them to the **closest-pose context view** (same trick
as the identity baseline). Multi-view feed-forward NVS adapters
(MVSplat, pixelSplat) would use all K views; Splatter Image cannot.

Install
-------
::

    git clone https://github.com/szymanowiczs/splatter-image
    cd splatter-image && pip install -e .          # exposes `scene`, `gaussian_renderer`, `utils`

    # First call to this adapter auto-downloads the checkpoint via
    # huggingface_hub from szymanowiczs/splatter-image-multi-category-v1.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class SplatterImage:
    """Single-view feed-forward 3DGS rendering at a target pose.

    Attributes
    ----------
    name : str
    native_precision : str
    torch_module : torch.nn.Module
        Exposed so the profiler can count params + FLOPs.
    """

    name = "splatter_image"
    native_precision = "fp32"
    torch_module: Any = None  # populated in __init__

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = "multi-category-v1",
        config_path: str | None = None,
        **_kwargs: Any,
    ) -> None:
        # Lazy upstream imports — fail with an actionable AdapterError if
        # the upstream repo isn't on PYTHONPATH. This is the same pattern
        # used by the `_not_yet_wired` registry entries; the difference
        # is that on success the model loads end-to-end.
        try:
            import torch  # noqa: PLC0415
            from gaussian_renderer import render_predicted  # noqa: PLC0415
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            from omegaconf import OmegaConf  # noqa: PLC0415
            from scene.gaussian_predictor import GaussianSplatPredictor  # noqa: PLC0415
            from utils.app_utils import to_tensor  # noqa: PLC0415
        except ImportError as e:
            from rpx_benchmark.exceptions import AdapterError  # noqa: PLC0415

            raise AdapterError(
                "splatter_image: upstream not importable.",
                hint=(
                    "Clone https://github.com/szymanowiczs/splatter-image, "
                    "`pip install -e .` from that dir, and make sure `scene`, "
                    "`gaussian_renderer`, `utils` are on PYTHONPATH. The "
                    "checkpoint is auto-downloaded from the HF Hub repo "
                    f"szymanowiczs/splatter-image-{checkpoint} on first call."
                ),
            ) from e

        self._torch = torch
        self._render_predicted = render_predicted
        self._to_tensor = to_tensor
        self._OmegaConf = OmegaConf
        self.device = device

        # Config — the upstream ships `gradio_config.yaml` for the
        # multi-category checkpoint. For other checkpoints, the upstream
        # repo has matching configs; pass `config_path` to override.
        cfg_path = config_path or "gradio_config.yaml"
        self.model_cfg = OmegaConf.load(cfg_path)

        # Weights
        repo_id = f"szymanowiczs/splatter-image-{checkpoint}"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_latest.pth")

        # Build + load
        self.model = GaussianSplatPredictor(self.model_cfg)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(device).eval()
        self.torch_module = self.model

    # ── helpers ─────────────────────────────────────────────────────────

    def _closest_context_index(
        self,
        context_poses: List[np.ndarray],
        target_pose: np.ndarray,
    ) -> int:
        """Single-view fallback: pick context view whose translation is
        L2-closest to the target. Same heuristic as the identity baseline."""
        t_target = target_pose[:3, 3]
        dists = [float(np.linalg.norm(p[:3, 3] - t_target)) for p in context_poses]
        return int(np.argmin(dists))

    def _build_camera_inputs(
        self,
        src_pose_c2w: np.ndarray,
        target_pose_c2w: np.ndarray,
    ) -> Dict[str, Any]:
        """Convert a pair of RPX 4×4 camera-to-world poses to the four
        camera tensors Splatter Image expects.

        Splatter Image's convention (per ``utils/app_utils.py``):
          - ``view_to_world_transform = c2w_cmo.transpose(0, 1)``
          - ``world_view_transform   = c2w_cmo.inverse().transpose(0, 1)``
          - ``camera_center          = view_to_world[3, :3]``
          - ``full_proj              = world_view.bmm(projection_matrix)``
            (projection matrix is config-driven; we re-use the cfg's
            FOV / near / far through their ``get_projection_matrix`` helper.)

        TODO[smoke]: the ``.transpose(0, 1)`` step assumes the upstream
        stores poses in column-major-order, matching their ``_cmo``
        suffix. Verify against a reference loop-camera output before
        trusting the geometry on RPX poses.
        """
        torch = self._torch

        # RPX poses are row-major (4, 4) cam-to-world float64.
        v2w_src_t = torch.from_numpy(src_pose_c2w).float().to(self.device)
        v2w_tgt_t = torch.from_numpy(target_pose_c2w).float().to(self.device)

        # Splatter Image's transpose accounts for col-major storage.
        view_to_world_src = v2w_src_t.transpose(0, 1)
        view_to_world_tgt = v2w_tgt_t.transpose(0, 1)
        world_view_tgt    = v2w_tgt_t.inverse().transpose(0, 1)
        camera_center_tgt = view_to_world_tgt[3, :3].clone()

        # Source quaternion (used to align the predicted Gaussians
        # with the source camera frame). Use the rotation block from
        # the *transposed* c2w to match utils/app_utils.py:
        #
        #   qs.append(matrix_to_quaternion(source_camera[..., :3, :3].transpose(0, 1)))
        from utils.general_utils import matrix_to_quaternion  # noqa: PLC0415

        rot_src_quat = matrix_to_quaternion(
            view_to_world_src[:3, :3].transpose(0, 1)
        ).unsqueeze(0)  # (1, 4)

        # Projection matrix — reuse the config's view setup. Splatter
        # Image's render uses a single fixed FOV per dataset; we honour
        # that and avoid feeding RPX's D435 intrinsics directly (the
        # model wasn't trained at that FOV).
        #
        # TODO[smoke]: confirm projection_matrix construction matches
        # what utils/general_utils.getProjectionMatrix expects.
        from utils.general_utils import getProjectionMatrix  # noqa: PLC0415

        proj = getProjectionMatrix(
            znear=self.model_cfg.data.znear,
            zfar=self.model_cfg.data.zfar,
            fovX=self.model_cfg.data.fov * np.pi / 180.0,
            fovY=self.model_cfg.data.fov * np.pi / 180.0,
        ).to(self.device)
        full_proj_tgt = world_view_tgt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

        return {
            "view_to_world_src":  view_to_world_src.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
            "rot_src_quat":       rot_src_quat.unsqueeze(0),                    # (1, 1, 4)
            "world_view_tgt":     world_view_tgt,                                # (4, 4)
            "full_proj_tgt":      full_proj_tgt,                                 # (4, 4)
            "camera_center_tgt":  camera_center_tgt,                             # (3,)
        }

    # ── main call ───────────────────────────────────────────────────────

    def __call__(
        self,
        context_rgbs:   List[np.ndarray],
        context_depths: List[np.ndarray],
        context_poses:  List[np.ndarray],
        target_pose:    np.ndarray,
    ) -> Dict[str, Any]:
        """Render the target view by predicting a 3DGS from the closest
        context view, then rasterising at ``target_pose``.

        Returns
        -------
        dict
            ``{"rgb": (H, W, 3) uint8, "depth": (H, W) float32 | None}``.

            TODO[smoke]: ``render_predicted`` returns a dict whose primary
            key is ``"render"`` (the RGB). Verify whether ``"depth"`` (or
            similar) is also returned by the gaussian-splat renderer the
            upstream uses; if so, surface it here.
        """
        if not context_rgbs:
            raise ValueError("splatter_image received zero context views")
        torch = self._torch

        # Single-view: pick the closest context view.
        src_idx = self._closest_context_index(context_poses, target_pose)
        src_rgb = context_rgbs[src_idx]   # (H, W, 3) uint8
        src_pose = context_poses[src_idx]  # (4, 4) c2w

        # Preprocess: HWC uint8 → CHW float in [0, 1] → (1, 1, 3, H, W).
        img_tensor = self._to_tensor(src_rgb).to(self.device)
        img_input = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)

        # Camera tensors.
        cams = self._build_camera_inputs(src_pose, target_pose)

        # Forward — predict gaussians from the single source view.
        # focals_pixels_pred=None: model uses its config's default FOV.
        with torch.no_grad():
            reconstruction = self.model(
                img_input,
                cams["view_to_world_src"],
                cams["rot_src_quat"],
                None,
            )

        # Render at the target pose.
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            render_out = self._render_predicted(
                {k: v[0].contiguous() for k, v in reconstruction.items()},
                cams["world_view_tgt"],
                cams["full_proj_tgt"],
                cams["camera_center_tgt"],
                background,
                self.model_cfg,
            )

        # Convert rendered RGB to (H, W, 3) uint8.
        rgb_t = render_out["render"]  # (3, H, W) float in [0, 1]
        rgb = (
            rgb_t.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)

        depth = render_out.get("depth")
        if depth is not None:
            depth = depth.squeeze().cpu().numpy().astype(np.float32)

        return {"rgb": rgb, "depth": depth}
