"""Shared vLLM-only inference backend for the RPX VQA roster.

Supports both one-image (normal) and two-image (in-context) requests.
Every engine is constructed with limit_mm_per_prompt={"image": 2} so
in-context rows never get rejected, but a normal row still sends exactly
one image -- the limit is a ceiling, not a requirement.

PaliGemma is architecturally a single-image model: vLLM's PaliGemma
implementation has no established multi-image interleaving convention (unlike
Qwen2.5-VL/Gemma4/Idefics3/InternVL/LLaVA-OneVision, which accept an ordered
image list through the official chat template). For PaliGemma's in-context
call, which per the task contract must see BOTH images, this backend
composites Image 1 and Image 2 side by side into one image and
feeds PaliGemma that composite -- a disclosed, model-specific accommodation
for a real architecture limit, not a hidden extra inference stage. Every
PaliGemma's native `answer en` and `detect <label>` tasks are exposed only in
the unscored diagnostic that explicitly separates localization from reasoning.
It is not used in the scored single-call VQA+bbox benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
            # Use the checkpoint's recommended/default visual resolution.  The
            # earlier 3*364 override discarded detail from RPX's small objects.
            "mm_processor_kwargs": {"size": {"longest_edge": 4 * 364}},
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
    "qwen3-vl-8b": VLLMCheckpoint(
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        max_model_len=8192,
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
    "qwen3-vl-2b": VLLMCheckpoint(
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        max_model_len=8192,
    ),
}


class VLLMVQARunner:
    """One vLLM engine instance serving one frozen VQA checkpoint."""

    def __init__(
        self,
        model_key: str,
        image_root: str | Path,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 1,
    ) -> None:
        """max_num_seqs caps how many sequences vLLM schedules concurrently.
        The acceptance/smoke gates keep this at 1 (the default) so latency is
        genuinely isolated, per-request; the benchmark runner raises it to
        the configured batch size so predict_batch's rows are actually
        processed together, not serialized one-by-one inside vLLM."""
        from vllm import LLM

        checkpoint = CHECKPOINTS[model_key]
        self.model_key = model_key
        self.checkpoint = checkpoint
        self._last_adapter_metadata: dict[str, Any] = {}
        self._last_batch_adapter_metadata: list[dict[str, Any]] = []
        self.llm = LLM(
            model=checkpoint.repo_id,
            revision=checkpoint.revision,
            dtype="bfloat16",
            max_model_len=checkpoint.max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            # 2, not 1: in-context rows send [reference, target]; normal rows
            # still send exactly one image. This is a ceiling, never a floor.
            limit_mm_per_prompt={"image": 2},
            allowed_local_media_path=str(Path(image_root).resolve()),
            trust_remote_code=checkpoint.trust_remote_code,
            **checkpoint.engine_kwargs,
        )

    @staticmethod
    def _text(outputs: list[Any]) -> str:
        return outputs[0].outputs[0].text.strip()

    @staticmethod
    def _side_by_side(images: Sequence[Image.Image]) -> Image.Image:
        """Composite two images left-to-right for PaliGemma's single-image
        scored call. Diagnostic grounding receives Image 2 alone instead."""
        # PaliGemma has a single-image interface.  Make the otherwise implicit
        # left/right convention visible in the pixels instead of expecting the
        # model to infer which panel the question calls Image 1 and Image 2.
        from PIL import ImageDraw

        gap = 8
        header = 28
        height = max(image.height for image in images)
        scaled = [
            image.resize((max(1, round(image.width * height / image.height)), height))
            for image in images
        ]
        width = sum(image.width for image in scaled) + gap * (len(scaled) - 1)
        canvas = Image.new("RGB", (width, height + header), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for index, image in enumerate(scaled, 1):
            label = "IMAGE 1: REFERENCE" if index == 1 else "IMAGE 2: TARGET"
            draw.text((x + 4, 7), label, fill=(0, 0, 0))
            canvas.paste(image, (x, header))
            x += image.width + gap
        return canvas

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

    @staticmethod
    def _chat_content(
        image_paths: Sequence[str | Path], prompt: str
    ) -> list[dict[str, Any]]:
        """Build an explicitly ordered multimodal message without model tokens."""
        content: list[dict[str, Any]] = []
        if len(image_paths) == 2:
            labels = ("Image 1 — reference object", "Image 2 — target scene")
            for label, path in zip(labels, image_paths, strict=True):
                content.append({"type": "text", "text": f"{label}:"})
                content.append(
                    {"type": "image_url", "image_url": {"url": Path(path).resolve().as_uri()}}
                )
        else:
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": Path(path).resolve().as_uri()},
                }
                for path in image_paths
            )
        content.append({"type": "text", "text": prompt})
        return content

    def _chat_generate(
        self, image_paths: Sequence[str | Path], prompt: str, max_tokens: int
    ) -> str:
        from vllm import SamplingParams

        content = self._chat_content(image_paths, prompt)
        messages = [{"role": "user", "content": content}]
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

    def _predict_json_direct(
        self, image_paths: Sequence[str | Path], prompt: str, max_tokens: int
    ) -> str:
        raw = self._chat_generate(image_paths, prompt, max_tokens)
        self._last_adapter_metadata = {
            "adapter": "direct_bbox_json",
            "single_scored_model_call": True,
        }
        return raw

    def _predict_paligemma_direct(
        self, images: Sequence[Image.Image], prompt: str, max_tokens: int
    ) -> str:
        """One scored PaliGemma call: question and image(s) in, bbox out."""
        model_image = images[0] if len(images) == 1 else self._side_by_side(images)
        raw = self._generate_paligemma(
            model_image, prompt, max_tokens, keep_special=True
        )
        self._last_adapter_metadata = {
            "adapter": "direct_bbox_json",
            "single_scored_model_call": True,
            "paligemma_multi_image_accommodation": (
                "none" if len(images) == 1 else "labelled_side_by_side_composite"
            ),
        }
        return raw

    def prediction_metadata(self) -> dict[str, Any]:
        return dict(self._last_adapter_metadata)

    def batch_prediction_metadata(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._last_batch_adapter_metadata]

    def predict(
        self,
        image_paths: str | Path | Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        output_kind: str,
    ) -> str:
        """image_paths is ordered [reference, target] for in-context rows,
        or a single path (or one-element sequence) for normal rows. Latency
        measured by the caller around this call already includes both
        PaliGemma stages, since both run inside this one invocation."""
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        opened = [Image.open(path) for path in image_paths]
        try:
            self._last_adapter_metadata = {}
            images = [image.convert("RGB") for image in opened]
            if output_kind == "diagnostic_semantic_label":
                if self.checkpoint.paligemma:
                    diagnostic_image = (
                        images[0] if len(images) == 1 else self._side_by_side(images)
                    )
                    raw = self._generate_paligemma(
                        diagnostic_image, prompt, max_tokens, keep_special=False
                    )
                else:
                    raw = self._chat_generate(image_paths, prompt, max_tokens)
                self._last_adapter_metadata = {
                    "adapter": "diagnostic_semantic_label",
                    "diagnostic_only": True,
                    "single_model_call": True,
                    "single_scored_model_call": False,
                }
                return raw
            if output_kind in {
                "diagnostic_oracle_bbox",
                "diagnostic_predicted_label_bbox",
            }:
                # Oracle localization intentionally sees only the target image.
                # The GT label is disclosed in the prompt, so this output must
                # never be reported as an end-to-end benchmark prediction.
                if self.checkpoint.paligemma:
                    raw = self._generate_paligemma(
                        images[-1], prompt, max_tokens, keep_special=True
                    )
                else:
                    raw = self._chat_generate(
                        [image_paths[-1]], prompt, max_tokens
                    )
                self._last_adapter_metadata = {
                    "adapter": output_kind,
                    "diagnostic_only": True,
                    "single_model_call": True,
                    "single_scored_model_call": False,
                    "ground_truth_label_disclosed": (
                        output_kind == "diagnostic_oracle_bbox"
                    ),
                }
                return raw
            if self.checkpoint.paligemma:
                if output_kind == "bbox_json_normalized_1000":
                    raise ValueError(
                        "PaliGemma has no direct single-call RPX VQA+bbox protocol; "
                        "use its unscored answer-en plus detect diagnostic"
                    )
                diagnostic_image = (
                    images[0] if len(images) == 1 else self._side_by_side(images)
                )
                raw = self._generate_paligemma(
                    diagnostic_image, prompt, max_tokens, keep_special=False
                )
                self._last_adapter_metadata = {
                    "adapter": "paligemma_single_call",
                    "single_scored_model_call": True,
                }
                return raw
            if output_kind == "bbox_json_normalized_1000":
                return self._predict_json_direct(image_paths, prompt, max_tokens)
            return self._chat_generate(image_paths, prompt, max_tokens)
        finally:
            for image in opened:
                image.close()

    def predict_batch(
        self, requests: Sequence[tuple[Sequence[str | Path], str, int, str]]
    ) -> list[str]:
        """Submit an entire batch to vLLM in one call so rows are actually
        scheduled together (subject to max_num_seqs), not serialized one at a
        time. Returns raw outputs in the same order as requests. The caller
        is responsible for measuring wall time around this call and dividing
        by len(requests) for an AMORTIZED per-row latency -- never label that
        number as isolated single-request latency (see predict, which is
        what the acceptance/smoke gates use for that)."""
        opened_by_row = [
            [Image.open(path) for path in ((paths,) if isinstance(paths, (str, Path)) else paths)]
            for paths, _prompt, _max_tokens, _output_kind in requests
        ]
        try:
            self._last_batch_adapter_metadata = []
            images_by_row = [[image.convert("RGB") for image in row] for row in opened_by_row]
            if self.checkpoint.paligemma:
                if any(
                    output_kind == "bbox_json_normalized_1000"
                    for _paths, _prompt, _max_tokens, output_kind in requests
                ):
                    raise ValueError(
                        "PaliGemma has no direct single-call RPX VQA+bbox protocol; "
                        "use its unscored answer-en plus detect diagnostic"
                    )
                # This branch is retained for non-bbox native tasks. Direct
                # bbox requests were rejected above because PaliGemma needs
                # separate answer-en and detect calls for the RPX task.
                direct_requests = [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": images[0] if len(images) == 1 else self._side_by_side(images)
                        },
                    }
                    for images, (_paths, prompt, _max_tokens, _output_kind) in zip(
                        images_by_row, requests, strict=True
                    )
                ]
                from vllm import SamplingParams

                max_tokens_1 = max(
                    max_tokens for _paths, _prompt, max_tokens, _output_kind in requests
                )
                params1 = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens_1,
                    # Preserve native <loc> outputs if the model chooses its
                    # trained detection serialization instead of JSON.
                    skip_special_tokens=False,
                )
                direct_outputs = self.llm.generate(
                    direct_requests, params1, use_tqdm=False
                )
                values = [self._text([output]) for output in direct_outputs]
                self._last_batch_adapter_metadata = [
                    {
                        "adapter": (
                            "direct_bbox_json"
                            if output_kind == "bbox_json_normalized_1000"
                            else "paligemma_single_call"
                        ),
                        "single_scored_model_call": True,
                        "paligemma_multi_image_accommodation": (
                            "none"
                            if len(images) == 1
                            else "labelled_side_by_side_composite"
                        ),
                    }
                    for images, (_paths, _prompt, _max_tokens, output_kind) in zip(
                        images_by_row, requests, strict=True
                    )
                ]
                return values

            from vllm import SamplingParams

            messages_batch = []
            for _images, (paths, prompt, _max_tokens, _output_kind) in zip(
                images_by_row, requests, strict=True
            ):
                path_list = (paths,) if isinstance(paths, (str, Path)) else paths
                # Use the exact same labelled Image-1/Image-2 construction as
                # isolated acceptance.  The previous batched path preserved
                # order but silently omitted these labels, changing the task
                # between acceptance and the full benchmark.
                content = self._chat_content(path_list, prompt)
                messages_batch.append([{"role": "user", "content": content}])
            max_tokens = max(
                max_tokens for _paths, _prompt, max_tokens, _output_kind in requests
            )
            params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            chat_template_kwargs = (
                {"enable_thinking": False} if self.model_key.startswith("gemma4-") else None
            )
            outputs = self.llm.chat(
                messages_batch, params, use_tqdm=False, chat_template_kwargs=chat_template_kwargs
            )
            stage1_values = [self._text([output]) for output in outputs]

            values = list(stage1_values)
            metadata: list[dict[str, Any]] = [
                (
                    {"adapter": "direct_bbox_json", "single_scored_model_call": True}
                    if request[3] == "bbox_json_normalized_1000"
                    else {}
                )
                for request in requests
            ]
            self._last_batch_adapter_metadata = metadata
            return values
        finally:
            for row in opened_by_row:
                for image in row:
                    image.close()
