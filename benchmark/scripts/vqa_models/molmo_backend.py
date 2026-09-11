"""Native Transformers backend for the Molmo 0924 checkpoints.

Molmo accepts one or more images through its official processor.  RPX asks it
for the same normalized XYXY JSON used by the other general-purpose VLMs.  The
model's native pointing markup is intentionally not converted to a bounding
box: a point has insufficient extent information for an honest IoU score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


@dataclass(frozen=True)
class MolmoCheckpoint:
    repo_id: str
    revision: str
    backend: str = "transformers-molmo"


CHECKPOINTS = {
    "molmo-7b-d": MolmoCheckpoint(
        "allenai/Molmo-7B-D-0924",
        "cab33fb7f1a40091911f81165f8481920621948f",
    ),
    "molmoe-1b": MolmoCheckpoint(
        "allenai/MolmoE-1B-0924",
        "69e3445d130507eadaa9123e3c411ce17aeb8afa",
    ),
}


class MolmoVQARunner:
    """Run a frozen Molmo checkpoint through its documented HF interface."""

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
        self.dtype = torch.bfloat16
        # Both the code and weights are frozen to the recorded Hub commit.
        # trust_remote_code executes only inside the RPX Docker container.
        self.processor = AutoProcessor.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            trust_remote_code=True,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                checkpoint.repo_id,
                revision=checkpoint.revision,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
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
        import torch
        from transformers import GenerationConfig

        inputs = self.processor.process(images=list(images), text=prompt)
        inputs = {key: value.to(self.device).unsqueeze(0) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,
            stop_strings="<|endoftext|>",
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.dtype):
            output = self.model.generate_from_batch(
                inputs,
                config,
                tokenizer=self.processor.tokenizer,
            )
        generated = output[0, input_length:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict(
        self,
        image_paths: str | Path | Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        if output_kind not in {
            "bbox_json_normalized_1000",
            "diagnostic_semantic_label",
            "diagnostic_oracle_bbox",
            "diagnostic_predicted_label_bbox",
        }:
            raise ValueError(f"unsupported Molmo output kind: {output_kind}")
        images = self._open_images(image_paths)
        raw = self._generate(images, prompt, max_tokens)
        diagnostic = output_kind.startswith("diagnostic_")
        self._last_adapter_metadata = {
            "adapter": ("molmo_diagnostic" if diagnostic else "direct_bbox_json"),
            "backend": "transformers-molmo",
            "image_count": len(images),
            "image_order": ("target_only" if len(images) == 1 else "reference_then_target"),
            "single_model_call": True,
            "single_scored_model_call": not diagnostic,
            "diagnostic_only": diagnostic,
            "native_point_outputs_are_not_boxes": True,
            "ground_truth_label_disclosed": output_kind == "diagnostic_oracle_bbox",
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
