"""Native Transformers backend for scored PaliGemma 2 question grounding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PaliGemmaCheckpoint:
    repo_id: str
    revision: str
    backend: str = "transformers-paligemma2"


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


_LOC_RE = re.compile(
    r"<loc(?P<y0>\d{4})><loc(?P<x0>\d{4})>"
    r"<loc(?P<y1>\d{4})><loc(?P<x1>\d{4})>"
)


CHECKPOINTS = {
    "paligemma2-10b": PaliGemmaCheckpoint(
        "google/paligemma2-10b-mix-448",
        "b26d16fb4251090ba4a4aa5af9fca1f8248ed5b6",
    ),
    "paligemma2-3b": PaliGemmaCheckpoint(
        "google/paligemma2-3b-mix-448",
        "1406c92ec87d32cc6b983239278901b904ba7a51",
    ),
}


class PaliGemmaVQARunner:
    """Run official answer/detect task prefixes through the HF processor."""

    def __init__(
        self,
        model_key: str,
        image_root: str | Path,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 1,
    ) -> None:
        del image_root, gpu_memory_utilization
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        checkpoint = CHECKPOINTS[model_key]
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.max_batch_size = max(1, max_num_seqs)
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(
            checkpoint.repo_id, revision=checkpoint.revision
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self._last_adapter_metadata: dict[str, Any] = {}
        self._last_batch_adapter_metadata: list[dict[str, Any]] = []

    @staticmethod
    def _side_by_side(
        images: Sequence[Image.Image],
    ) -> tuple[Image.Image, ImageGeometry]:
        if len(images) != 2:
            raise ValueError("PaliGemma in-context composition requires exactly two images")
        gap, header = 8, 28
        height = max(image.height for image in images)
        scaled = [
            image.resize((max(1, round(image.width * height / image.height)), height))
            for image in images
        ]
        canvas = Image.new(
            "RGB", (scaled[0].width + gap + scaled[1].width, height + header), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 7), "IMAGE 1: REFERENCE", fill="black")
        target_left = scaled[0].width + gap
        draw.text((target_left + 4, 7), "IMAGE 2: TARGET", fill="black")
        canvas.paste(scaled[0], (0, header))
        canvas.paste(scaled[1], (target_left, header))
        return canvas, ImageGeometry(
            width=canvas.width,
            height=canvas.height,
            target_left=target_left,
            target_top=header,
            target_width=scaled[1].width,
            target_height=scaled[1].height,
        )

    @staticmethod
    def _target_bbox(
        locs: dict[str, int], geometry: ImageGeometry
    ) -> list[float] | None:
        """Map PaliGemma's composite-relative loc tokens to Image 2."""
        x0 = locs["x0"] * geometry.width / 1024
        x1 = locs["x1"] * geometry.width / 1024
        y0 = locs["y0"] * geometry.height / 1024
        y1 = locs["y1"] * geometry.height / 1024
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

    @classmethod
    def _decode_question_grounding(
        cls, raw: str, geometry: ImageGeometry
    ) -> tuple[str, dict[str, Any]]:
        matches = list(_LOC_RE.finditer(raw))
        candidates = []
        for index, match in enumerate(matches):
            locs = {key: int(value) for key, value in match.groupdict().items()}
            bbox = cls._target_bbox(locs, geometry)
            if bbox is None:
                continue
            text_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            label = raw[match.end():text_end].split("<", 1)[0].strip()
            candidates.append({"label": label, "bbox": bbox})
        metadata = {
            "native_output": raw,
            "native_candidate_count": len(matches),
            "target_candidate_count": len(candidates),
            "candidate_policy": "exactly_one_region_in_target_panel",
        }
        if len(candidates) == 1:
            return json.dumps(candidates[0], separators=(",", ":")), metadata
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

    @staticmethod
    def _open_images(paths: Sequence[str | Path]) -> list[Image.Image]:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return images

    def _generate(
        self, image: Image.Image, prompt: str, max_tokens: int, *, keep_special: bool
    ) -> str:
        import torch

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
            )
        # HF generate returns prompt+completion for this decoder. Decode only
        # completion tokens; otherwise the task prefix contaminates parsing.
        completion = generated[:, input_length:]
        return self.processor.batch_decode(
            completion,
            skip_special_tokens=not keep_special,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def predict(
        self,
        image_paths: str | Path | Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        if isinstance(image_paths, (str, Path)):
            paths = [image_paths]
        else:
            paths = list(image_paths)
        if len(paths) not in {1, 2}:
            raise ValueError("RPX VQA rows must contain one or two images")
        images = self._open_images(paths)
        if output_kind == "diagnostic_semantic_label":
            model_image = (
                images[0] if len(images) == 1 else self._side_by_side(images)[0]
            )
            raw = self._generate(model_image, prompt, max_tokens, keep_special=False)
            adapter = "paligemma2_native_answer"
            metadata = {}
        elif output_kind == "bbox_native_question_grounding":
            if len(images) == 1:
                model_image = images[0]
                geometry = ImageGeometry(images[0].width, images[0].height)
            else:
                model_image, geometry = self._side_by_side(images)
            native_raw = self._generate(
                model_image, prompt, max_tokens, keep_special=True
            )
            raw, metadata = self._decode_question_grounding(native_raw, geometry)
            adapter = "direct_native_question_grounding"
        elif output_kind in {
            "diagnostic_oracle_bbox",
            "diagnostic_predicted_label_bbox",
        }:
            # Diagnostic localization always receives only the target image.
            raw = self._generate(images[-1], prompt, max_tokens, keep_special=True)
            adapter = "paligemma2_native_detect"
            metadata = {}
        else:
            raise ValueError(f"unsupported PaliGemma output kind: {output_kind}")
        diagnostic = output_kind.startswith("diagnostic_")
        self._last_adapter_metadata = {
            "adapter": adapter,
            "backend": "transformers-paligemma2",
            "diagnostic_only": diagnostic,
            "single_model_call": True,
            "single_scored_model_call": not diagnostic,
            "task_prefix": prompt.split(" ", 1)[0],
            "ground_truth_label_disclosed": output_kind == "diagnostic_oracle_bbox",
            **metadata,
        }
        return raw

    def predict_batch(
        self, requests: Sequence[tuple[Sequence[str | Path], str, int, str]]
    ) -> list[str]:
        values = []
        metadata = []
        for request in requests:
            values.append(self.predict(*request))
            metadata.append(self.prediction_metadata())
        self._last_batch_adapter_metadata = metadata
        return values

    def prediction_metadata(self) -> dict[str, Any]:
        return dict(self._last_adapter_metadata)

    def batch_prediction_metadata(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._last_batch_adapter_metadata]
