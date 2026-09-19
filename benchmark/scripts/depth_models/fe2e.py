"""Official FE2E single-image relative-depth adapter.

FE2E was initially kept behind an unverified-weights safety rail.  The paper
authors subsequently released ``AMAP-ML/FE2E`` and linked the model files in
``exander/FE2E`` from that repository.  This adapter follows the authors'
published depth evaluation path: Step1X-Edit base model, FE2E LDRN LoRA,
empty prompt, one denoising step, and the left (depth) decoder output.

The returned map is the authors' normalized relative-depth prediction. It is
not metric depth; RPX evaluates it with least-squares log-depth alignment
(``native_alignment = "ls_log"``), matching FE2E's official ``norm_type=ln``
output convention. RPX pools that log-space fit over each
``(scene, phase)`` cell, as required by the RPX paper protocol.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Union

import numpy as np

from rpx_benchmark.exceptions import AdapterError


class FE2EAdapter:
    """FE2E: one RGB image to affine-invariant dense depth."""

    MODEL_ID = "exander/FE2E"
    MODEL_REVISION = "4c52e5e3d133e4111f6099b57879d517dd8677b8"
    UPSTREAM_REVISION = "262d4c9d4c37752c984304f57ca7d7066f34ad5d"
    DIT_FILENAME = "pretrain/step1x-edit-i1258.safetensors"
    VAE_FILENAME = "pretrain/vae.safetensors"
    LORA_FILENAME = "LDRN.safetensors"
    DIT_SHA256 = "f378db311e6db3535bbed367160fda7f65661c1d2474cffeaded47288c00a24a"
    VAE_SHA256 = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
    LORA_SHA256 = "9c2112d3697d6057a60d59b3d83fc0d88ea12d73f933666eded0c5dbf5e0f410"
    DEFAULT_SOURCE_PATH = "/opt/rpx-envs/sources/fe2e"

    native_alignment: str = "ls_log"
    native_precision: str = "bf16"
    UNVERIFIED = False

    def __init__(
        self,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        source_path: str = DEFAULT_SOURCE_PATH,
        seed: int = 1234,
        offload: Optional[bool] = None,
    ) -> None:
        if int(batch_size) != 1:
            raise AdapterError("FE2E official inference supports batch size 1 only.")
        if device == "cpu":
            raise AdapterError("FE2E production inference requires CUDA.")
        if dtype not in {None, "auto", "bf16", "bfloat16"}:
            raise AdapterError(
                f"FE2E official inference requires BF16; received dtype={dtype!r}."
            )

        source = Path(source_path)
        empty_prompt_cache = source / "latent" / "no_info.npz"
        if not empty_prompt_cache.is_file():
            raise AdapterError(
                f"Pinned FE2E source is missing {empty_prompt_cache}.",
                hint=(
                    "Build docker/depth-fe2e/Dockerfile on top of the current "
                    "rpx-depth-paper image; it installs AMAP-ML/FE2E at the "
                    "verified upstream revision."
                ),
            )

        try:
            from huggingface_hub import hf_hub_download
            from infer.inference import ImageGenerator
        except ImportError as exc:
            raise AdapterError(
                f"Official FE2E environment is unavailable: {exc}",
                hint=(
                    "Run this model with /opt/rpx-envs/fe2e/bin/python from the "
                    "docker/depth-fe2e image."
                ),
            ) from exc

        try:
            if offload is None:
                offload = os.environ.get("RPX_FE2E_OFFLOAD", "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }
            dit_path = hf_hub_download(model_id, self.DIT_FILENAME, revision=revision)
            vae_path = hf_hub_download(model_id, self.VAE_FILENAME, revision=revision)
            lora_path = hf_hub_download(model_id, self.LORA_FILENAME, revision=revision)
            args = SimpleNamespace(
                prompt_type="empty",
                single_denoise=True,
                norm_type="ln",
                task_name="depth",
                empty_prompt_cache=str(empty_prompt_cache),
            )
            self._pipeline = ImageGenerator(
                ae_path=vae_path,
                dit_path=dit_path,
                qwen2vl_model_path=None,
                max_length=640,
                quantized=False,
                offload=bool(offload),
                lora=lora_path,
                device=device,
                args=args,
            )
        except Exception as exc:
            raise AdapterError(
                f"Official FE2E load failed for {model_id}@{revision}: {exc}",
                hint=(
                    "Confirm the FE2E Docker overlay was built, the Hugging Face "
                    "cache mount is writable, and the selected GPU has enough free VRAM."
                ),
            ) from exc

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = 1
        self.seed = int(seed)
        self.offload = bool(offload)
        self._args = args

    @property
    def torch_module(self):
        return getattr(self._pipeline, "dit", None)

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        is_batch = isinstance(rgb, (list, tuple))
        images = list(rgb) if is_batch else [rgb]
        outputs = [self._infer_one(image) for image in images]
        return outputs if is_batch else outputs[0]

    def _infer_one(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise AdapterError(f"FE2E expected HxWx3 RGB, got {image.shape}.")
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
                raise AdapterError("FE2E RGB input must be finite numeric data.")
            image = np.clip(image, 0, 255).astype(np.uint8)

        try:
            import torch

            # Match the official evaluator: BCHW float RGB followed by the
            # upstream tensor/bilinear preprocessing path.
            rgb_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
            rgb_tensor = rgb_tensor.unsqueeze(0).float().div_(255.0)
            _normal_images, depth_tensor, _normal_tensor = self._pipeline.generate_image(
                "",
                negative_prompt="",
                ref_images=rgb_tensor,
                num_steps=1,
                cfg_guidance=6.0,
                seed=self.seed,
                num_samples=1,
                show_progress=False,
                size_level=768,
                args=self._args,
            )
            depth = depth_tensor[0].detach().float().cpu().numpy()
        except Exception as exc:
            raise AdapterError(f"Official FE2E inference failed: {exc}") from exc

        # The official evaluator averages the decoded three-channel depth image.
        if depth.ndim == 3:
            depth = depth.mean(axis=0)
        if depth.ndim != 2:
            raise AdapterError(f"FE2E returned an unexpected depth shape {depth.shape}.")
        if depth.shape != image.shape[:2]:
            depth = _resize_bilinear(depth, image.shape[:2])

        depth = np.asarray(depth, dtype=np.float32)
        if not np.isfinite(depth).all() or float(np.ptp(depth)) <= 0.0:
            raise AdapterError("FE2E returned a non-finite or degenerate depth map.")
        return depth


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.asarray(src, dtype=np.float32), mode="F")
    image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)
