"""Official MetaDepth HyDen adapters.

The canonical RPX row uses the released HyDen-MoGeV2 metric-point model and
extracts camera-forward Z depth. The same wrapper can also load HyDen-DA2's
released relative-depth checkpoint for the larger legacy registry. Neither
checkpoint is a Transformers auto-model; both require Meta's pinned
``facebookresearch/metadepth`` source tree.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class HyDen:
    """HyDen RGB-to-depth wrapper using the official MetaDepth classes."""

    METRIC_MODEL_ID = "facebook/hyden-mogev2-metric-point"
    METRIC_FILENAME = "hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth"
    RELATIVE_MODEL_ID = "facebook/hyden-da2-relative-depth"
    RELATIVE_FILENAME = "hyden_da2_reldepth_vitl_fp32_f809396504.pth"
    UPSTREAM_REVISION = "810b77c3e56712813a0de42a130cfbe6f5b19b90"

    def __init__(
        self,
        model_id: str = METRIC_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        native_alignment: str = "none",
        native_precision: str = "fp32",
        dtype: Optional[str] = None,
    ) -> None:
        if dtype not in {None, "float32", "fp32"}:
            raise ValueError("Official HyDen checkpoints are evaluated in FP32.")
        if model_id not in {self.METRIC_MODEL_ID, self.RELATIVE_MODEL_ID}:
            raise ValueError(f"unsupported official HyDen checkpoint: {model_id}")
        try:
            import torch
            from huggingface_hub import hf_hub_download

            if model_id == self.METRIC_MODEL_ID:
                from metadepth.mogev2 import MODEL_CONFIGS, HyDenMoGe

                from copy import deepcopy
                model_config = deepcopy(MODEL_CONFIGS["vitl_dinov2"])
                # Metric-point checkpoint does not contain the surface-normal head.
                model_config.pop("normal_head", None)
                model = HyDenMoGe(**model_config)
                filename = self.METRIC_FILENAME
                self._metric = True
            else:
                from metadepth.da2 import HyDenDepthAnything

                model = HyDenDepthAnything(encoder="vitl")
                filename = self.RELATIVE_FILENAME
                self._metric = False
            checkpoint = hf_hub_download(repo_id=model_id, filename=filename)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Present in the released checkpoint but unused by this inference model.
            state.pop("encoder.backbone.mask_token", None)
            model.load_state_dict(state, strict=True)
            self._model = model.to(device).eval()
        except Exception as exc:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"Official HyDen load failed for {model_id!r}: {exc}",
                hint=(
                    "Run setup_depth_smoke_env.py --model hyden, accept the "
                    "Meta checkpoint terms, and authenticate with `hf auth login`."
                ),
            ) from exc

        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.native_alignment = native_alignment
        self.native_precision = native_precision
        self._torch = torch

    @property
    def torch_module(self):
        return self._model

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        torch = self._torch
        is_batch = isinstance(rgb, (list, tuple))
        images = list(rgb) if is_batch else [rgb]
        tensors = []
        target_shapes = []
        for image in images:
            array = np.asarray(image, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(f"expected H×W×3 RGB uint8, got {array.shape}")
            target_shapes.append(array.shape[:2])
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(518, 518),
                mode="bilinear",
                align_corners=False,
            )[0]
            if not self._metric:
                mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
                std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
                tensor = (tensor - mean) / std
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            if self._metric:
                output = self._model(batch, return_mask_and_scale=True)
                depth = output["points"][..., 2]
                scale = output.get("metric_scale")
                if scale is not None:
                    depth = depth * scale[:, None, None]
            else:
                depth = self._model(batch)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]

        results = []
        for item, target_shape in zip(depth, target_shapes, strict=True):
            item = torch.nn.functional.interpolate(
                item[None, None].float(),
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            results.append(item.detach().cpu().numpy().astype(np.float32))
        return results if is_batch else results[0]
