"""Metric3D V2 official ViT-Giant2 metric-depth adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


class Metric3DV2:
    """Metric3D V2 ViT-Giant2: RGB image to depth in metres."""

    DEFAULT_HUB_ENTRY = "metric3d_vit_giant2"
    DEFAULT_HUB_REPO = (
        "YvanYin/Metric3D:eb5b6fac0dc155e4e52f576e304fbf11655ff339"
    )
    DEFAULT_SOURCE_PATH = "/opt/rpx-envs/sources/metric3d"

    INPUT_HEIGHT = 616
    INPUT_WIDTH = 1064
    CANONICAL_FOCAL_PX = 1000.0
    D435_RGB_FOCAL_PX = 615.0
    MAX_DEPTH_METRES = 300.0

    native_alignment: str = "none"
    native_precision: str = "fp32"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        hub_entry: str = DEFAULT_HUB_ENTRY,
        hub_repo: str = DEFAULT_HUB_REPO,
        dtype: Optional[str] = None,
        focal_length_px: Optional[float] = None,
        source_path: Optional[str] = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Metric3D V2 needs PyTorch.") from exc

        self.device = device
        self.batch_size = int(batch_size)
        self.hub_entry = hub_entry
        self.hub_repo = hub_repo
        self.focal_length_px = float(
            focal_length_px or self.D435_RGB_FOCAL_PX
        )
        self._torch = torch

        local_source = Path(
            source_path
            or os.environ.get(
                "RPX_METRIC3D_SOURCE",
                self.DEFAULT_SOURCE_PATH,
            )
        )

        try:
            if local_source.is_dir():
                model = torch.hub.load(
                    str(local_source),
                    hub_entry,
                    pretrain=True,
                    trust_repo=True,
                    source="local",
                )
            else:
                model = torch.hub.load(
                    hub_repo,
                    hub_entry,
                    pretrain=True,
                    trust_repo=True,
                    source="github",
                )
        except Exception as exc:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"Metric3D V2 load failed: {exc}",
                hint=(
                    "Confirm the pinned Metric3D source exists and mount "
                    "TORCH_HOME=/cache/torch for the public checkpoint."
                ),
            ) from exc

        if dtype:
            target_dtype = (
                getattr(torch, dtype)
                if isinstance(dtype, str)
                else dtype
            )
            model = model.to(dtype=target_dtype)

        self._model = model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def _predict_one(self, rgb: np.ndarray) -> np.ndarray:
        import cv2
        import torch.nn.functional as functional

        torch = self._torch
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"expected H×W×3 RGB uint8, got shape {image.shape}"
            )

        original_height, original_width = image.shape[:2]

        resize_scale = min(
            self.INPUT_HEIGHT / original_height,
            self.INPUT_WIDTH / original_width,
        )
        resized_height = max(1, int(original_height * resize_scale))
        resized_width = max(1, int(original_width * resize_scale))

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_height = self.INPUT_HEIGHT - resized_height
        pad_width = self.INPUT_WIDTH - resized_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        mean_values = (123.675, 116.28, 103.53)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=mean_values,
        )

        mean = torch.tensor(mean_values).float()[:, None, None]
        std = torch.tensor((58.395, 57.12, 57.375)).float()[:, None, None]
        tensor = torch.from_numpy(
            padded.transpose(2, 0, 1)
        ).float()
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            result = self._model.inference({"input": tensor})

        if isinstance(result, (tuple, list)):
            depth_tensor = result[0]
        elif isinstance(result, dict):
            depth_tensor = result.get("prediction")
            if depth_tensor is None:
                depth_tensor = result.get("depth")
        else:
            depth_tensor = result

        if depth_tensor is None:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError("Metric3D returned no depth tensor.")

        depth_tensor = depth_tensor.squeeze()
        if depth_tensor.ndim != 2:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"Metric3D returned depth shape "
                f"{tuple(depth_tensor.shape)}; expected 2-D."
            )

        bottom = (
            depth_tensor.shape[0] - pad_bottom
            if pad_bottom
            else depth_tensor.shape[0]
        )
        right = (
            depth_tensor.shape[1] - pad_right
            if pad_right
            else depth_tensor.shape[1]
        )
        depth_tensor = depth_tensor[
            pad_top:bottom,
            pad_left:right,
        ]

        depth_tensor = functional.interpolate(
            depth_tensor[None, None],
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        processed_focal_px = self.focal_length_px * (
            resized_width / float(original_width)
        )
        depth_tensor = depth_tensor * (
            processed_focal_px / self.CANONICAL_FOCAL_PX
        )
        depth_tensor = torch.clamp(
            depth_tensor,
            0.0,
            self.MAX_DEPTH_METRES,
        )

        return (
            depth_tensor.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        is_batch = isinstance(rgb, (list, tuple))
        images = list(rgb) if is_batch else [rgb]
        outputs = [self._predict_one(image) for image in images]
        return outputs if is_batch else outputs[0]
