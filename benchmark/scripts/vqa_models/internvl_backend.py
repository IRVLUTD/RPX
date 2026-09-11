"""Native Transformers backend for InternVL 3.5 visual grounding.

InternVL is evaluated through its documented ``model.chat`` path, including
dynamic 448-pixel tiling and explicit Image-1/Image-2 markers.  The scored
prompt requests InternVL's native ``label[[x0,y0,x1,y1]]`` 0--1000 grounding
format; no answer-first call or ground-truth label is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class InternVLCheckpoint:
    repo_id: str
    revision: str
    backend: str = "transformers-internvl"


CHECKPOINTS = {
    "internvl3.5-1b": InternVLCheckpoint(
        "OpenGVLab/InternVL3_5-1B",
        "2f71cf52542334823e48a46ffba0e2bc9add3446",
    ),
    "internvl3.5-14b": InternVLCheckpoint(
        "OpenGVLab/InternVL3_5-14B",
        "a1e37197b393ce9eec9df700fef65c11f4a6ffbd",
    ),
}


def _build_transform(input_size: int = 448):
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_difference = float("inf")
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Reproduce InternVL's official dynamic high-resolution preprocessing."""
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = _closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )
    blocks = columns * rows
    resized = image.resize((image_size * columns, image_size * rows))
    images: list[Image.Image] = []
    for index in range(blocks):
        x0 = (index % columns) * image_size
        y0 = (index // columns) * image_size
        images.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and blocks != 1:
        images.append(image.resize((image_size, image_size)))
    return images


def _load_pixels(image: Image.Image, max_num: int) -> torch.Tensor:
    transform = _build_transform()
    tiles = _dynamic_preprocess(image, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


class InternVLVQARunner:
    """Run pinned InternVL 3.5 checkpoints through their native chat API."""

    def __init__(
        self,
        model_key: str,
        image_root: str | Path,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 1,
    ) -> None:
        del image_root, gpu_memory_utilization, max_num_seqs
        from transformers import AutoModel, AutoTokenizer

        checkpoint = CHECKPOINTS[model_key]
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.max_batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            trust_remote_code=True,
            use_fast=False,
        )
        # ``device_map=auto`` is the checkpoint's documented loading path and
        # keeps 1B on its single visible GPU while splitting 14B over the two
        # GPUs exposed by the Server 3 launch command.
        self.model = AutoModel.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map="auto",
        ).eval()
        self._last_adapter_metadata: dict[str, Any] = {}
        self._last_batch_adapter_metadata: list[dict[str, Any]] = []

    @staticmethod
    def _open_images(
        image_paths: str | Path | Sequence[str | Path],
    ) -> list[Image.Image]:
        paths = [image_paths] if isinstance(image_paths, (str, Path)) else list(image_paths)
        if len(paths) not in {1, 2}:
            raise ValueError("RPX VQA rows must contain one or two images")
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return images

    def _generate(self, images: Sequence[Image.Image], prompt: str, max_tokens: int) -> str:
        # InternVL training distributes a bounded tile budget over multi-image
        # inputs.  Six per image retains detail while capping the total at the
        # same twelve-tile budget used for a single image.
        per_image_max = 12 if len(images) == 1 else 6
        pixel_groups = [_load_pixels(image, per_image_max) for image in images]
        num_patches_list = [pixels.shape[0] for pixels in pixel_groups]
        pixel_values = torch.cat(pixel_groups).to(self.device, dtype=self.dtype)
        if len(images) == 1:
            model_prompt = f"<image>\n{prompt}"
        else:
            model_prompt = f"Image-1: <image>\nImage-2: <image>\n{prompt}"
        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }
        with torch.inference_mode():
            return self.model.chat(
                self.tokenizer,
                pixel_values,
                model_prompt,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            ).strip()

    def predict(
        self,
        image_paths: str | Path | Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        if output_kind != "bbox_native_internvl_grounding":
            raise ValueError(f"unsupported InternVL output kind: {output_kind}")
        images = self._open_images(image_paths)
        raw = self._generate(images, prompt, max_tokens)
        self._last_adapter_metadata = {
            "adapter": "direct_internvl_grounding",
            "backend": "transformers-internvl",
            "image_count": len(images),
            "image_order": "target_only" if len(images) == 1 else "reference_then_target",
            "single_model_call": True,
            "single_scored_model_call": True,
            "native_coordinate_format": "internvl_bbox_0_1000",
            "ground_truth_label_disclosed": False,
        }
        return raw

    def predict_batch(
        self, requests: Sequence[tuple[Sequence[str | Path], str, int, str]]
    ) -> list[str]:
        values: list[str] = []
        metadata: list[dict[str, Any]] = []
        for request in requests:
            values.append(self.predict(*request))
            metadata.append(self.prediction_metadata())
        self._last_batch_adapter_metadata = metadata
        return values

    def prediction_metadata(self) -> dict[str, Any]:
        return dict(self._last_adapter_metadata)

    def batch_prediction_metadata(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._last_batch_adapter_metadata]
