"""Gemma 3 multimodal inference backend for RPX bbox VQA."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

CHECKPOINTS = {
    "gemma3-4b": ("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767"),
    "gemma3-12b": ("google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80"),
}


class Gemma3Runner:
    def __init__(self, model_key: str) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        repo_id, revision = CHECKPOINTS[model_key]
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def predict(self, image_path: str | Path, prompt: str, max_new_tokens: int) -> str:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            )
        return self.processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()
