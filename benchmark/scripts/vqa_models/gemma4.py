"""Gemma 4 multimodal inference backend for RPX bbox VQA."""

from __future__ import annotations

from pathlib import Path

CHECKPOINTS = {
    "gemma4-e4b": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    "gemma4-12b": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
}


class Gemma4Runner:
    def __init__(self, model_key: str) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        repo_id, revision = CHECKPOINTS[model_key]
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            repo_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def predict(self, image_path: str | Path, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(Path(image_path).resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
        raw = self.processor.decode(generated[0][input_length:], skip_special_tokens=False)
        parsed = self.processor.parse_response(raw, prefix=inputs["input_ids"])
        if isinstance(parsed, dict):
            content = parsed.get("content", parsed.get("text", raw))
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            return str(content).strip()
        return str(parsed).strip()
