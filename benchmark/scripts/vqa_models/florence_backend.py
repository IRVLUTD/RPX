"""Native Florence-2 question-grounding and diagnostic adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class FlorenceCheckpoint:
    repo_id: str
    revision: str
    backend: str = "transformers-florence2"


CHECKPOINTS = {
    "florence2-base": FlorenceCheckpoint(
        "microsoft/Florence-2-base-ft",
        "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
    ),
    "florence2-large": FlorenceCheckpoint(
        "microsoft/Florence-2-large-ft",
        "4a12a2b54b7016a48a22037fbd62da90cd566f2a",
    ),
}


@dataclass(frozen=True)
class ImageGeometry:
    width: int
    height: int
    target_left: int = 0
    target_top: int = 0
    target_width: int | None = None
    target_height: int | None = None

    @property
    def target_w(self) -> int:
        return self.target_width or self.width

    @property
    def target_h(self) -> int:
        return self.target_height or self.height


class FlorenceVQARunner:
    """One native Florence-2 model for scored and diagnostic inference."""

    TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
    VQA_TASK = "<VQA>"

    def __init__(
        self,
        model_key: str,
        image_root: str | Path,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 1,
    ) -> None:
        del image_root, gpu_memory_utilization
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        checkpoint = CHECKPOINTS[model_key]
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.max_batch_size = max(1, max_num_seqs)
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self._last_adapter_metadata: dict[str, Any] = {}
        self._last_batch_adapter_metadata: list[dict[str, Any]] = []

    @staticmethod
    def _side_by_side(images: Sequence[Image.Image]) -> tuple[Image.Image, ImageGeometry]:
        if len(images) != 2:
            raise ValueError("Florence in-context composition requires exactly two images")
        gap, header = 8, 28
        height = max(image.height for image in images)
        scaled = [
            image.resize((max(1, round(image.width * height / image.height)), height))
            for image in images
        ]
        width = scaled[0].width + gap + scaled[1].width
        canvas = Image.new("RGB", (width, height + header), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 7), "IMAGE 1: REFERENCE", fill=(0, 0, 0))
        target_left = scaled[0].width + gap
        draw.text((target_left + 4, 7), "IMAGE 2: TARGET", fill=(0, 0, 0))
        canvas.paste(scaled[0], (0, header))
        canvas.paste(scaled[1], (target_left, header))
        geometry = ImageGeometry(
            width=canvas.width,
            height=canvas.height,
            target_left=target_left,
            target_top=header,
            target_width=scaled[1].width,
            target_height=scaled[1].height,
        )
        return canvas, geometry

    @staticmethod
    def _normal_geometry(image: Image.Image) -> ImageGeometry:
        return ImageGeometry(image.width, image.height)

    @staticmethod
    def _target_bbox(
        bbox: Sequence[float], geometry: ImageGeometry
    ) -> list[float] | None:
        if len(bbox) != 4:
            return None
        x0, y0, x1, y1 = (float(value) for value in bbox)
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        left, top = geometry.target_left, geometry.target_top
        right, bottom = left + geometry.target_w, top + geometry.target_h
        if not (left <= center_x <= right and top <= center_y <= bottom):
            return None
        x0, x1 = max(left, x0), min(right, x1)
        y0, y1 = max(top, y0), min(bottom, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return [
            (x0 - left) * 1000 / geometry.target_w,
            (y0 - top) * 1000 / geometry.target_h,
            (x1 - left) * 1000 / geometry.target_w,
            (y1 - top) * 1000 / geometry.target_h,
        ]

    def _decode_grounding(
        self,
        generated_text: str,
        geometry: ImageGeometry,
        output_kind: str = "diagnostic_predicted_label_bbox",
        referring_expression: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        postprocess_error = None
        try:
            parsed = self.processor.post_process_generation(
                generated_text,
                task=self.TASK,
                image_size=(geometry.width, geometry.height),
            )
            result = parsed.get(self.TASK) or {}
        except Exception as error:  # model text can be outside the native loc grammar
            result = {}
            postprocess_error = f"{type(error).__name__}: {error}"
        boxes = list(result.get("bboxes") or [])
        labels = list(result.get("labels") or [])
        candidates = []
        for index, bbox in enumerate(boxes):
            target_bbox = self._target_bbox(bbox, geometry)
            if target_bbox is not None:
                candidates.append(
                    {
                        "label": str(labels[index]) if index < len(labels) else "",
                        "bbox": target_bbox,
                    }
                )
        diagnostic = output_kind.startswith("diagnostic_")
        composite = geometry.target_left != 0 or geometry.target_top != 0
        metadata = {
            "adapter": (
                "florence2_composite_phrase_grounding"
                if composite
                else "florence2_native_phrase_grounding"
                if diagnostic
                else "direct_native_question_grounding"
            ),
            "image_count": 2 if composite else 1,
            "image_order": "reference_then_target" if composite else "target_only",
            "multi_image_accommodation": (
                "labelled_side_by_side_composite" if composite else "none"
            ),
            "single_model_call": True,
            "single_scored_model_call": not diagnostic,
            "diagnostic_only": diagnostic,
            "task_token": self.TASK,
            "native_output": generated_text,
            "native_candidate_count": len(boxes),
            "target_candidate_count": len(candidates),
            "candidate_policy": "exactly_one_region_in_target_panel",
            "num_beams": 3,
        }
        if referring_expression is not None:
            metadata["referring_expression"] = referring_expression
        if postprocess_error is not None:
            metadata["postprocess_error"] = postprocess_error
        if len(candidates) == 1:
            return json.dumps(candidates[0], separators=(",", ":")), metadata
        # Deliberately omit `bbox`: the strict evaluator records a model-format
        # failure instead of choosing among candidates with GT information.
        return (
            json.dumps(
                {
                    "error": "expected exactly one target-region candidate",
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                },
                separators=(",", ":"),
            ),
            metadata,
        )

    def _prepare_image(
        self, image_paths: str | Path | Sequence[str | Path]
    ) -> tuple[Image.Image, ImageGeometry]:
        paths = [image_paths] if isinstance(image_paths, (str, Path)) else list(image_paths)
        if len(paths) not in {1, 2}:
            raise ValueError("RPX VQA rows must contain one or two images")
        with_images = []
        for path in paths:
            with Image.open(path) as image:
                with_images.append(image.convert("RGB"))
        if len(with_images) == 1:
            return with_images[0], self._normal_geometry(with_images[0])
        return self._side_by_side(with_images)

    def _generate_batch(
        self,
        requests: Sequence[tuple[Sequence[str | Path], str, int, str]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        import torch

        images: list[Image.Image] = []
        geometries: list[ImageGeometry] = []
        prompts: list[str] = []
        output_kinds: list[str] = []
        max_tokens = 1
        for paths, prompt, row_max_tokens, output_kind in requests:
            if not (
                output_kind.startswith("diagnostic_")
                or output_kind == "bbox_native_question_grounding"
            ):
                raise ValueError(
                    f"unsupported Florence-2 output kind: {output_kind}"
                )
            image, geometry = self._prepare_image(paths)
            images.append(image)
            geometries.append(geometry)
            task = self.VQA_TASK if output_kind == "diagnostic_semantic_label" else self.TASK
            prompts.append(task + prompt)
            output_kinds.append(output_kind)
            max_tokens = max(max_tokens, row_max_tokens)
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device, dtype=self.dtype)
            if key == "pixel_values"
            else value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=3,
            )
        generated = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        values: list[str] = []
        metadata: list[dict[str, Any]] = []
        for text, geometry, output_kind, full_prompt in zip(
            generated, geometries, output_kinds, prompts, strict=True
        ):
            if output_kind == "diagnostic_semantic_label":
                # Retain native VQA output as-is; it is analysis only and is
                # never substituted for the one-stage scored grounding result.
                values.append(text)
                metadata.append(
                    {
                        "adapter": "florence2_native_vqa_label",
                        "diagnostic_only": True,
                        "single_model_call": True,
                        "single_scored_model_call": False,
                        "task_token": self.VQA_TASK,
                        "native_output": text,
                    }
                )
            else:
                value, row_metadata = self._decode_grounding(
                    text,
                    geometry,
                    output_kind,
                    referring_expression=full_prompt[len(self.TASK):],
                )
                if output_kind.startswith("diagnostic_"):
                    row_metadata["ground_truth_label_disclosed"] = (
                        output_kind == "diagnostic_oracle_bbox"
                    )
                values.append(value)
                metadata.append(row_metadata)
        return values, metadata

    def predict(
        self,
        image_paths: str | Path | Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        values, metadata = self._generate_batch(
            [(image_paths, prompt, max_tokens, output_kind)]
        )
        self._last_adapter_metadata = metadata[0]
        return values[0]

    def predict_batch(
        self, requests: Sequence[tuple[Sequence[str | Path], str, int, str]]
    ) -> list[str]:
        values, metadata = self._generate_batch(requests)
        self._last_batch_adapter_metadata = metadata
        return values

    def prediction_metadata(self) -> dict[str, Any]:
        return dict(self._last_adapter_metadata)

    def batch_prediction_metadata(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._last_batch_adapter_metadata]
