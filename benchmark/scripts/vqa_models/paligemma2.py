"""PaliGemma 2 inference backend for the single-image RPX VQA contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class PaliGemmaCheckpoint:
    repo_id: str
    revision: str


CHECKPOINTS = {
    "paligemma2-3b": PaliGemmaCheckpoint(
        "google/paligemma2-3b-mix-448", "1406c92ec87d32cc6b983239278901b904ba7a51"
    ),
    "paligemma2-10b": PaliGemmaCheckpoint(
        "google/paligemma2-10b-mix-448", "b26d16fb4251090ba4a4aa5af9fca1f8248ed5b6"
    ),
}


class PaliGemma2Runner:
    def __init__(self, model_key: str) -> None:
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        checkpoint = CHECKPOINTS[model_key]
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            checkpoint.repo_id, revision=checkpoint.revision
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def _generate(
        self, image_path: str | Path, prompt: str, max_new_tokens: int, keep_special: bool
    ) -> str:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = inputs.to(input_device)
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
        return self.processor.decode(
            generated[0][input_length:],
            skip_special_tokens=not keep_special,
        ).strip()

    def predict(
        self, image_path: str | Path, prompt: str, max_new_tokens: int, output_kind: str
    ) -> str:
        if output_kind != "paligemma_two_stage":
            return self._generate(image_path, prompt, max_new_tokens, keep_special=False)
        predicted_label = self._generate(image_path, prompt, max_new_tokens, keep_special=False)
        label = predicted_label.strip().splitlines()[0]
        if not label:
            return ""
        grounded = self._generate(
            image_path,
            f"detect {label}\n",
            max_new_tokens=64,
            keep_special=True,
        )
        return grounded
