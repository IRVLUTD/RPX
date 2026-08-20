"""DepthSplat adapter for calibrated RPX novel-view synthesis.

The adapter uses the released two-view, 256x256 small checkpoint.  RPX stores
raw librealsense T265 poses; DepthSplat expects OpenCV camera-to-world poses.
The fixed ``diag(1, -1, -1)`` basis conversion below changes conventions only
and deliberately does not fit a D435-to-T265 extrinsic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DEPTHSPLAT_GIT_SHA = "2dad25a7b9ba7c5537aba08463ef88d8511a06be"
DEPTHSPLAT_REPO_ID = "haofeixu/depthsplat"
DEPTHSPLAT_CHECKPOINT = "depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth"

# FewSOL/RPX's published D435 RGB calibration at 640x480.
_K_D435 = np.array(
    [
        [611.10888672, 0.0, 315.51083374],
        [0.0, 610.02844238, 237.73669434],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_T265_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _opencv_relative_poses(poses: List[np.ndarray], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw T265 C2W matrices and remove their arbitrary world origin."""

    converted = [_T265_TO_OPENCV @ np.asarray(p, np.float32) @ _T265_TO_OPENCV for p in poses]
    converted_target = _T265_TO_OPENCV @ np.asarray(target, np.float32) @ _T265_TO_OPENCV
    world_from_first = np.linalg.inv(converted[0])
    contexts = np.stack([world_from_first @ p for p in converted], axis=0)
    return contexts, world_from_first @ converted_target


class DepthSplatNVS:
    """Real DepthSplat forward adapter at the released two-view operating point."""

    name = "depthsplat"
    native_precision = "fp32"
    required_context_views = 2

    def __init__(self, device: str = "cuda", **_kwargs: Any) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("DepthSplat requires CUDA, but CUDA is not available")

        self.device = torch.device(device)
        source = Path(os.environ.get("DEPTHSPLAT_SOURCE_DIR", "/opt/rpx-models/depthsplat"))
        if not (source / "config" / "main.yaml").is_file():
            raise RuntimeError(
                f"DepthSplat source is missing at {source}; use the RPX DepthSplat image"
            )
        sys.path.insert(0, str(source))

        # Import only after the pinned source root is on sys.path.
        from hydra import compose, initialize_config_dir
        from src.config import load_typed_root_config
        from src.global_cfg import set_cfg
        from src.loss import get_losses
        from src.model.decoder import get_decoder
        from src.model.encoder import get_encoder
        from src.model.model_wrapper import ModelWrapper

        with initialize_config_dir(version_base=None, config_dir=str(source / "config")):
            cfg_dict = compose(
                config_name="main",
                overrides=[
                    "+experiment=re10k",
                    "mode=test",
                    "wandb.mode=disabled",
                    "dataset.image_shape=[256,256]",
                    "dataset.near=0.3",
                    "dataset.far=5.0",
                    "dataset.baseline_scale_bounds=false",
                    "dataset.make_baseline_1=false",
                    "model.encoder.monodepth_vit_type=vits",
                    "model.encoder.num_scales=1",
                    "model.encoder.upsample_factor=4",
                    "model.encoder.lowest_feature_resolution=4",
                    "test.compute_scores=false",
                ],
            )
        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        encoder, visualizer = get_encoder(cfg.model.encoder)
        wrapper = ModelWrapper(
            cfg.optimizer,
            cfg.test,
            cfg.train,
            encoder,
            visualizer,
            get_decoder(cfg.model.decoder, cfg.dataset),
            get_losses(cfg.loss),
            None,
            eval_data_cfg=None,
        )

        checkpoint = hf_hub_download(
            repo_id=DEPTHSPLAT_REPO_ID,
            filename=DEPTHSPLAT_CHECKPOINT,
            cache_dir=os.environ.get("HF_HOME"),
            token=os.environ.get("HF_TOKEN") or None,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        wrapper.load_state_dict(state, strict=True)
        self.model = wrapper.eval().to(self.device)
        self.torch_module = self.model

    @staticmethod
    def _intrinsics(n: int) -> np.ndarray:
        k = _K_D435.copy()
        k[0, :] /= 640.0
        k[1, :] /= 480.0
        return np.repeat(k[None], n, axis=0)

    def __call__(
        self,
        context_rgbs: List[np.ndarray],
        context_depths: List[np.ndarray],
        context_poses: List[np.ndarray],
        target_pose: np.ndarray,
    ) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        del context_depths  # DepthSplat predicts geometry from RGB + calibrated poses.
        if len(context_rgbs) != self.required_context_views:
            raise ValueError(
                f"DepthSplat's released RPX operating point requires exactly "
                f"{self.required_context_views} context views; received {len(context_rgbs)}"
            )
        if len(context_poses) != len(context_rgbs):
            raise ValueError("context RGB and pose counts differ")

        output_h, output_w = context_rgbs[0].shape[:2]
        images_np = np.stack(context_rgbs).astype(np.float32) / 255.0
        images = torch.from_numpy(images_np).permute(0, 3, 1, 2)
        images = F.interpolate(images, (256, 256), mode="bilinear", align_corners=False)

        context_c2w, target_c2w = _opencv_relative_poses(context_poses, target_pose)
        intrinsics = self._intrinsics(len(context_rgbs))
        target_intrinsics = self._intrinsics(1)

        context = {
            "image": images.unsqueeze(0).to(self.device),
            "extrinsics": torch.from_numpy(context_c2w).unsqueeze(0).to(self.device),
            "intrinsics": torch.from_numpy(intrinsics).unsqueeze(0).to(self.device),
            "near": torch.full((1, len(context_rgbs)), 0.3, device=self.device),
            "far": torch.full((1, len(context_rgbs)), 5.0, device=self.device),
            "index": torch.arange(len(context_rgbs), device=self.device).unsqueeze(0),
        }
        with torch.inference_mode():
            encoded = self.model.encoder(context, 0, deterministic=True)
            gaussians = encoded["gaussians"] if isinstance(encoded, dict) else encoded
            rendered = self.model.decoder.forward(
                gaussians,
                torch.from_numpy(target_c2w).view(1, 1, 4, 4).to(self.device),
                torch.from_numpy(target_intrinsics).view(1, 1, 3, 3).to(self.device),
                torch.full((1, 1), 0.3, device=self.device),
                torch.full((1, 1), 5.0, device=self.device),
                (256, 256),
                depth_mode="depth",
            )

        color = F.interpolate(
            rendered.color[:, 0], (output_h, output_w), mode="bilinear", align_corners=False
        )[0]
        rgb = (
            color.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
        )
        depth = None
        if rendered.depth is not None:
            depth = F.interpolate(
                rendered.depth[:, :1], (output_h, output_w), mode="bilinear", align_corners=False
            )[0, 0].float().cpu().numpy()
        return {"rgb": rgb, "depth": depth}


__all__ = [
    "DEPTHSPLAT_CHECKPOINT",
    "DEPTHSPLAT_GIT_SHA",
    "DEPTHSPLAT_REPO_ID",
    "DepthSplatNVS",
    "_opencv_relative_poses",
]
