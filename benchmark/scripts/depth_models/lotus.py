"""Official Lotus-2 monocular relative-depth adapter.

This adapter follows the pinned upstream inference path from
``EnVision-Research/Lotus-2``. Lotus-2 uses a FLUX.1-dev transformer plus
three published Lotus-2 weight files (core predictor, local-continuity
module and detail sharpener). Its output is relative depth and is aligned
once per RPX ``(scene, phase)`` cell by the benchmark runner.

The upstream source must be importable. The reproducible setup helper
``scripts/setup_depth_smoke_env.py --model lotus-2`` checks out the exact
source revision and installs a ``.pth`` entry for it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class Lotus:
    """Lotus-2: RGB image to affine-invariant depth."""

    DEFAULT_MODEL_ID = "jingheya/Lotus-2"
    DEFAULT_BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
    UPSTREAM_REVISION = "2d5e4522f7213611184fd31992d0fac17ec36035"

    native_alignment: str = "ls_affine"
    native_precision: str = "bf16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        num_inference_steps: int = 10,
        dtype: Optional[str] = None,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
    ) -> None:
        try:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
            from huggingface_hub import hf_hub_download
            from infer import (
                CORE_PREDICTOR_FILENAME,
                DETAIL_SHARPENER_FILENAME,
                LCM_FILENAME,
                load_lora_and_lcm_weights,
            )
            from pipeline import Lotus2Pipeline
        except ImportError as e:
            raise ImportError(
                "Lotus-2 needs its pinned upstream checkout plus diffusers/peft. "
                "Run: python scripts/setup_depth_smoke_env.py --model lotus-2 "
                "--env-root <env-root>"
            ) from e

        self.model_id = model_id
        self.base_model_id = base_model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.num_inference_steps = int(num_inference_steps)
        self._torch = torch
        weight_dtype = getattr(torch, dtype) if isinstance(dtype, str) else torch.bfloat16

        try:
            core_path = hf_hub_download(
                model_id,
                CORE_PREDICTOR_FILENAME["depth"],
            )
            lcm_path = hf_hub_download(model_id, LCM_FILENAME["depth"])
            sharpener_path = hf_hub_download(
                model_id,
                DETAIL_SHARPENER_FILENAME["depth"],
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                base_model_id,
                subfolder="scheduler",
                num_train_timesteps=10,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                base_model_id,
                subfolder="transformer",
                torch_dtype=weight_dtype,
            )
            transformer.requires_grad_(False)
            transformer.to(device=device, dtype=weight_dtype)
            transformer, local_continuity = load_lora_and_lcm_weights(
                transformer,
                core_path,
                lcm_path,
                sharpener_path,
                "depth",
            )
            pipe = Lotus2Pipeline.from_pretrained(
                base_model_id,
                scheduler=scheduler,
                transformer=transformer,
                torch_dtype=weight_dtype,
            )
            pipe.local_continuity_module = local_continuity
            self._pipe = pipe.to(device)
            self._pipe.set_progress_bar_config(disable=True)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"Official Lotus-2 load failed: {e}",
                hint=(
                    "Lotus-2 requires accepted access to black-forest-labs/FLUX.1-dev "
                    "and a high-memory GPU. Confirm HF_TOKEN access and use the "
                    "pinned Lotus-2 environment from setup_depth_smoke_env.py."
                ),
            ) from e

    @property
    def torch_module(self):
        return getattr(self._pipe, "transformer", None)

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        torch = self._torch
        is_batch = isinstance(rgb, (list, tuple))
        rgbs = list(rgb) if is_batch else [rgb]
        outputs: list[np.ndarray] = []
        for image in rgbs:
            image = np.asarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(f"expected H×W×3 RGB uint8, got shape {image.shape}")
            tensor = torch.from_numpy(image.astype(np.float32))
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            tensor = tensor.to(self.device)
            max_edge = max(image.shape[:2])
            process_res = 1024 if max_edge > 1024 else 512 if max_edge < 512 else None
            with torch.inference_mode():
                prediction = self._pipe(
                    rgb_in=tensor,
                    prompt="",
                    num_inference_steps=self.num_inference_steps,
                    output_type="np",
                    process_res=process_res,
                ).images[0]
            depth = np.asarray(prediction, dtype=np.float32).mean(axis=-1)
            if depth.shape != image.shape[:2]:
                depth = _resize_bilinear(depth, image.shape[:2])
            outputs.append(depth)
        return outputs if is_batch else outputs[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(src.astype(np.float32), mode="F")
    image = image.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)
