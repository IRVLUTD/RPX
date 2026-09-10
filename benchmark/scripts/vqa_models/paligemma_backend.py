"""Native Transformers backend for unscored PaliGemma 2 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PaliGemmaCheckpoint:
    repo_id: str
    revision: str
    backend: str = "transformers-paligemma2"


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
    def _side_by_side(images: Sequence[Image.Image]) -> Image.Image:
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
        return canvas

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
        if not output_kind.startswith("diagnostic_"):
            raise ValueError(
                "PaliGemma supports only unscored answer/detect diagnostics in RPX"
            )
        images = self._open_images(paths)
        if output_kind == "diagnostic_semantic_label":
            model_image = images[0] if len(images) == 1 else self._side_by_side(images)
            raw = self._generate(model_image, prompt, max_tokens, keep_special=False)
            adapter = "paligemma2_native_answer"
        else:
            # Diagnostic localization always receives only the target image.
            raw = self._generate(images[-1], prompt, max_tokens, keep_special=True)
            adapter = "paligemma2_native_detect"
        self._last_adapter_metadata = {
            "adapter": adapter,
            "backend": "transformers-paligemma2",
            "diagnostic_only": True,
            "single_model_call": True,
            "single_scored_model_call": False,
            "task_prefix": prompt.split(" ", 1)[0],
            "ground_truth_label_disclosed": output_kind == "diagnostic_oracle_bbox",
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
