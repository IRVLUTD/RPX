"""Shared vLLM-only inference backend for the RPX VQA roster."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class VLLMCheckpoint:
    repo_id: str
    revision: str
    trust_remote_code: bool = False
    max_model_len: int = 4096
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    paligemma: bool = False


CHECKPOINTS = {
    "gemma4-12b": VLLMCheckpoint(
        "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    ),
    "paligemma2-10b": VLLMCheckpoint(
        "google/paligemma2-10b-mix-448",
        "b26d16fb4251090ba4a4aa5af9fca1f8248ed5b6",
        paligemma=True,
    ),
    "idefics3-8b": VLLMCheckpoint(
        "HuggingFaceM4/Idefics3-8B-Llama3",
        "fddb4ff79181e55a994674777e06cd5456ce3dc3",
        max_model_len=8192,
        engine_kwargs={
            "enforce_eager": True,
            "mm_processor_kwargs": {"size": {"longest_edge": 3 * 364}},
        },
    ),
    "internvl2.5-8b": VLLMCheckpoint(
        "OpenGVLab/InternVL2_5-8B",
        "e9e4c0dc1db56bfab10458671519b7fa3dd29463",
        trust_remote_code=True,
        max_model_len=8192,
    ),
    "qwen2.5-vl-7b": VLLMCheckpoint(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        engine_kwargs={
            "mm_processor_kwargs": {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            }
        },
    ),
    "llava-onevision-7b": VLLMCheckpoint(
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "0d50680527681998e456c7b78950205bedd8a068",
        max_model_len=8192,
    ),
    "gemma4-e4b": VLLMCheckpoint(
        "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"
    ),
    "phi-3.5-vision-4b": VLLMCheckpoint(
        "microsoft/Phi-3.5-vision-instruct",
        "12b77fb40b63a2c73c68243d3f767aab688a1b2a",
        trust_remote_code=True,
        engine_kwargs={"mm_processor_kwargs": {"num_crops": 16}},
    ),
    "paligemma2-3b": VLLMCheckpoint(
        "google/paligemma2-3b-mix-448",
        "1406c92ec87d32cc6b983239278901b904ba7a51",
        paligemma=True,
    ),
    "qwen2.5-vl-3b": VLLMCheckpoint(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "66285546d2b821cf421d4f5eb2576359d3770cd3",
        engine_kwargs={
            "mm_processor_kwargs": {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            }
        },
    ),
}


class VLLMVQARunner:
    """One vLLM engine instance serving one frozen VQA checkpoint."""

    def __init__(
        self,
        model_key: str,
        image_root: str | Path,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from vllm import LLM

        checkpoint = CHECKPOINTS[model_key]
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.llm = LLM(
            model=checkpoint.repo_id,
            revision=checkpoint.revision,
            dtype="bfloat16",
            max_model_len=checkpoint.max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            allowed_local_media_path=str(Path(image_root).resolve()),
            trust_remote_code=checkpoint.trust_remote_code,
            **checkpoint.engine_kwargs,
        )

    @staticmethod
    def _text(outputs: list[Any]) -> str:
        return outputs[0].outputs[0].text.strip()

    def _generate_paligemma(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int,
        keep_special: bool,
    ) -> str:
        from vllm import SamplingParams

        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            skip_special_tokens=not keep_special,
        )
        return self._text(self.llm.generate(request, params, use_tqdm=False))

    def predict(
        self,
        image_path: str | Path,
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        if self.checkpoint.paligemma:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            predicted_label = self._generate_paligemma(
                image, prompt, max_tokens, keep_special=False
            )
            label = predicted_label.splitlines()[0].strip()
            if not label:
                return ""
            return self._generate_paligemma(
                image, f"detect {label}\n", 64, keep_special=True
            )

        from vllm import SamplingParams

        media_url = Path(image_path).resolve().as_uri()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": media_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        chat_template_kwargs = (
            {"enable_thinking": False} if self.model_key.startswith("gemma4-") else None
        )
        return self._text(
            self.llm.chat(
                messages,
                params,
                use_tqdm=False,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
