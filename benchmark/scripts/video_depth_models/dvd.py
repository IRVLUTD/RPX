"""DVD v1.1 official relative video inverse-depth adapter.

DVD (Deterministic Video Depth Estimation with Generative Priors) publishes
relative inverse depth.  The RPX runner therefore fits one scale and shift in
disparity space over the complete scene-phase clip, then converts the aligned
prediction to metric depth.  This is the same non-cheating, per-clip alignment
scope used by the official DVD evaluator.

Long clips use the authors' overlapping-window inference and Global Affine
Coherence implementation.  The official v1.1 DVD checkpoint and the Wan2.1
base files are all fetched from immutable Hugging Face revisions.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from rpx_benchmark.api import VideoSample
from rpx_benchmark.exceptions import AdapterError

from ._video_adapter_base import VideoDepthAdapterBase

_DVD_SOURCE_REVISION = "62f52d16faa2cac10e31eb7815ece6e61d5449ec"
_DVD_MODEL_ID = "FayeHongfeiZhang/DVD"
_DVD_MODEL_REVISION = "769534dec2d77f8da3667719568fbe57fe69cd9b"
_DVD_MODEL_FILENAME = "dvd_1.1.safetensors"
_DVD_MODEL_SHA256 = "63079861484dfd6397a04f6327a0e20bbcee95ff1440cb7e61a83be2ecc6d854"
_WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
_WAN_MODEL_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
_WAN_FILENAMES = (
    "diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DVDAdapter(VideoDepthAdapterBase):
    """Wrap the authors' DVD v1.1 checkpoint and long-video inference."""

    DISPLAY_NAME = "DVD v1.1"
    OUTPUT_KIND = "relative"
    native_alignment = "ls_disparity"
    native_precision = "bf16"

    def __init__(
        self,
        device: str = "cuda",
        *,
        window_size: int = 81,
        overlap: int = 21,
    ) -> None:
        super().__init__(device=device)
        self.window_size = int(window_size)
        self.overlap = int(overlap)
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError("overlap must satisfy 0 <= overlap < window_size")
        self._model = None
        self._generate_depth_sliced = None

    def setup(self) -> None:
        if self._loaded:
            return
        if not self.device.startswith("cuda"):
            raise AdapterError("DVD is GPU-only in the RPX benchmark")

        try:
            import torch
            from accelerate import Accelerator
            from diffsynth.pipelines.wan_video_new_determine import (
                ModelConfig,
                WanVideoPipeline,
            )
            from examples.wanvideo.model_training.WanTrainingModule import (
                WanTrainingModule,
            )
            from huggingface_hub import hf_hub_download
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
            from test_script.test_single_video import generate_depth_sliced
        except ImportError as exc:
            raise AdapterError(
                "DVD official runtime is unavailable.",
                hint=(
                    "Build docker/depth-dvd/Dockerfile and run with "
                    "/opt/rpx-envs/dvd/bin/python."
                ),
            ) from exc

        dvd_path = hf_hub_download(
            repo_id=_DVD_MODEL_ID,
            filename=_DVD_MODEL_FILENAME,
            revision=_DVD_MODEL_REVISION,
        )
        actual_sha256 = _sha256(dvd_path)
        if actual_sha256 != _DVD_MODEL_SHA256:
            raise AdapterError(
                "DVD v1.1 checkpoint checksum mismatch.",
                hint=(
                    f"Expected {_DVD_MODEL_SHA256}, found {actual_sha256}. "
                    "Delete only this cached checkpoint and fetch the pinned "
                    "revision again."
                ),
            )

        wan_paths = [
            hf_hub_download(
                repo_id=_WAN_MODEL_ID,
                filename=filename,
                revision=_WAN_MODEL_REVISION,
            )
            for filename in _WAN_FILENAMES
        ]
        args = OmegaConf.create(
            {
                "mode": "regression",
                "denoise_step": 0.5,
                "training_target": "x",
                "lora_base_model": "dit",
                "lora_rank": 512,
            }
        )
        # DVD's fork disables the Wan text encoder and always supplies zero
        # prompt embeddings, but the upstream ``from_pretrained`` default
        # still downloads an unused ``google/*`` tokenizer into ``./models``.
        # Supply an already-pinned local path as the inert tokenizer config:
        # this removes an unversioned network side effect and lets the
        # container run safely as the invoking server UID.
        original_from_pretrained = WanVideoPipeline.from_pretrained

        def _from_pretrained_without_unused_tokenizer(*call_args, **call_kwargs):
            call_kwargs.setdefault(
                "tokenizer_config",
                ModelConfig(path=wan_paths[0]),
            )
            return original_from_pretrained(*call_args, **call_kwargs)

        WanVideoPipeline.from_pretrained = staticmethod(
            _from_pretrained_without_unused_tokenizer
        )
        try:
            model = WanTrainingModule(
                accelerator=Accelerator(),
                model_paths=json.dumps(wan_paths),
                model_id_with_origin_paths=None,
                trainable_models=None,
                use_gradient_checkpointing=False,
                lora_rank=args.lora_rank,
                lora_base_model=args.lora_base_model,
                args=args,
            )
        finally:
            WanVideoPipeline.from_pretrained = staticmethod(
                original_from_pretrained
            )
        state = load_file(dvd_path, device="cpu")
        dit_state = {
            key.removeprefix("pipe.dit."): value
            for key, value in state.items()
            if key.startswith("pipe.dit.")
        }
        if not dit_state:
            raise AdapterError("DVD checkpoint contains no pipe.dit parameters")
        model.pipe.dit.load_state_dict(dit_state, strict=True)
        model.merge_lora_layer()
        self._model = model.to(self.device).eval()
        self._generate_depth_sliced = generate_depth_sliced
        self._loaded = True

        if not torch.cuda.is_available():  # pragma: no cover - defensive
            raise AdapterError("DVD setup completed without a visible CUDA device")

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        if self._model is None or self._generate_depth_sliced is None:
            raise AdapterError("DVD model is not loaded")

        import torch

        rgb = np.asarray(sample.rgb_seq, dtype=np.uint8)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise AdapterError(
                f"DVD expected RGB sequence (T,H,W,3), found {rgb.shape}"
            )
        frame_count, height, width, _ = rgb.shape
        if height % 16 or width % 16:
            raise AdapterError(
                "DVD requires spatial dimensions divisible by 16; "
                f"RPX adapter received {height}x{width}"
            )

        video = (
            torch.from_numpy(rgb)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        inverse_depth = self._generate_depth_sliced(
            self._model,
            video,
            window_size=min(self.window_size, frame_count),
            overlap=min(self.overlap, max(0, frame_count - 1)),
        )
        inverse_depth = np.asarray(inverse_depth, dtype=np.float32)
        if inverse_depth.ndim == 5 and inverse_depth.shape[0] == 1:
            inverse_depth = inverse_depth[0]
        if inverse_depth.ndim == 4 and inverse_depth.shape[-1] in {1, 3}:
            inverse_depth = inverse_depth.mean(axis=-1)
        elif inverse_depth.ndim == 4 and inverse_depth.shape[1] == 1:
            inverse_depth = inverse_depth[:, 0]

        expected = (frame_count, height, width)
        if inverse_depth.shape != expected:
            raise AdapterError(
                f"DVD returned inverse depth {inverse_depth.shape}; expected {expected}"
            )
        if not np.isfinite(inverse_depth).all():
            raise AdapterError("DVD returned non-finite inverse depth")
        if float(np.ptp(inverse_depth)) <= 1e-8:
            raise AdapterError("DVD returned degenerate inverse depth")
        return inverse_depth


def build(device: str = "cuda", **kwargs):
    return DVDAdapter(device=device, **kwargs)


__all__ = ["DVDAdapter", "build"]
