"""GemDepth official video inverse-depth adapter.

The paper authors' official repository links the ``YuechengLiu/GemDepth``
checkpoint directly. The upstream inference API returns inverse depth, not
metres. ``native_alignment = "ls_disparity"`` tells the RPX video runner to
reproduce the official evaluation protocol: fit one scale and shift over the
whole scene-phase clip in disparity space, then invert to depth.
"""

from __future__ import annotations

import hashlib

import numpy as np

from rpx_benchmark.api import VideoSample
from rpx_benchmark.exceptions import AdapterError

from ._video_adapter_base import VideoDepthAdapterBase

_MODEL_ID = "YuechengLiu/GemDepth"
_MODEL_REVISION = "c901a724f2fc9a6d8764a25bec8799602c6126d6"
_MODEL_FILENAME = "gemdepth.pth"
_MODEL_SHA256 = "f7c3fb7791862cc82684ddb1804480fb0314fddba4cf0e3b706553d98292fcad"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class GemDepthAdapter(VideoDepthAdapterBase):
    """Wrap the official GemDepth ViT-L checkpoint and inference API."""

    DISPLAY_NAME = "GemDepth"
    OUTPUT_KIND = "relative"
    native_alignment = "ls_disparity"
    native_precision = "fp16"

    def __init__(
        self,
        device: str = "cuda",
        *,
        input_size: int = 518,
        fp32: bool = False,
    ) -> None:
        super().__init__(device=device)
        self.input_size = int(input_size)
        self.fp32 = bool(fp32)
        self._model = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from model.gemdepth import GemDepth
        except ImportError as exc:
            raise AdapterError(
                "GemDepth official runtime is unavailable.",
                hint=(
                    "Build docker/depth-gemdepth/Dockerfile and run this adapter "
                    "with /opt/rpx-envs/gemdepth/bin/python."
                ),
            ) from exc

        checkpoint_path = hf_hub_download(
            repo_id=_MODEL_ID,
            filename=_MODEL_FILENAME,
            revision=_MODEL_REVISION,
        )
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != _MODEL_SHA256:
            raise AdapterError(
                "GemDepth checkpoint checksum mismatch.",
                hint=(
                    f"Expected {_MODEL_SHA256}, found {actual_sha256}. "
                    "Delete only this cached checkpoint and fetch the pinned revision again."
                ),
            )

        model = GemDepth(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        # weights_only=False is the authors' documented load path. The exact
        # repository revision and file SHA-256 are verified immediately above.
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval()
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        if self._model is None:
            raise AdapterError("GemDepth model is not loaded")

        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        if rgb_seq.ndim != 4 or rgb_seq.shape[-1] != 3:
            raise AdapterError(
                f"GemDepth expected RGB sequence (T,H,W,3), found {rgb_seq.shape}"
            )

        inverse_depth, _fps = self._model.infer_video_depth(
            rgb_seq,
            target_fps=30,
            input_size=self.input_size,
            device=self.device,
            fp32=self.fp32,
        )
        inverse_depth = np.asarray(inverse_depth, dtype=np.float32)
        if inverse_depth.shape != rgb_seq.shape[:3]:
            raise AdapterError(
                "GemDepth returned an unexpected inverse-depth shape "
                f"{inverse_depth.shape}; expected {rgb_seq.shape[:3]}"
            )
        if not np.isfinite(inverse_depth).all():
            raise AdapterError("GemDepth returned non-finite inverse depth")
        if float(np.ptp(inverse_depth)) <= 1e-8:
            raise AdapterError("GemDepth returned degenerate inverse depth")
        return inverse_depth


def build(device: str = "cuda", **kwargs):
    return GemDepthAdapter(device=device, **kwargs)


__all__ = ["GemDepthAdapter", "build"]
