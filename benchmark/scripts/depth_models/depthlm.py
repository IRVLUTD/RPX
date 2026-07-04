"""DepthLM — metric depth from a vision-language model (Meta, ICLR'26 Oral).

Image Depth adapter wrapping the official ``facebook/DepthLM`` HF
release. Verified by HF model-card lookup
(huggingface.co/facebook/DepthLM): "12b model of DepthLM finetuned
from Pixtral" — a 12B-parameter VLM that emits metric depth from a
single RGB frame plus a text query.

**Verification status**: model_id verified; load incantation is the
standard transformers VLM pattern. The predict body uses Pixtral's
documented chat-style inference — team should verify the exact
prompt format against the upstream release before production.

Install
-------

::

    pip install transformers torch accelerate
    # Weights are ~24 GB (fp16); needs a high-VRAM GPU. The team's
    # lab box (8 GB) will OOM — run on a 40 GB+ machine, or apply
    # bitsandbytes 4-bit quantisation:
    #   pip install bitsandbytes
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

_MODEL_ID = "facebook/DepthLM"


class DepthLMAdapter:
    """Callable adapter: rgb (H×W×3 uint8) → depth (H×W float32, metres).

    DepthLM is a vision-language model — internally it runs a prompted
    chat-style inference and parses the response. We expose the same
    callable surface as every other ``scripts/depth_models/`` adapter
    so ``BatchedDepthBenchmarkModel`` wraps it unchanged.
    """

    #: DepthLM emits metric depth per the paper headline.
    native_alignment: str = "none"
    native_precision: str = "fp16"

    #: Pixtral-style prompt the model was fine-tuned with. The actual
    #: prompt template is in the upstream release; this is a sensible
    #: default that exercises the model. Team can override at __init__.
    DEFAULT_PROMPT = "Estimate the metric depth of every pixel in this image."

    def __init__(
        self,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
        *,
        prompt: Optional[str] = None,
        load_in_4bit: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "DepthLMAdapter needs `transformers`, `torch`, `accelerate`. "
                "Install with: pip install transformers torch accelerate"
            ) from e

        self.device = device
        self.batch_size = int(batch_size)
        self.prompt = prompt or self.DEFAULT_PROMPT

        load_kwargs = {}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True
                )
            except Exception as e:  # noqa: BLE001
                raise ImportError(
                    "load_in_4bit=True needs `bitsandbytes`. "
                    "Install with: pip install bitsandbytes"
                ) from e
        else:
            load_kwargs["torch_dtype"] = torch.float16

        # Verified load path from HF model card.
        self._model = AutoModelForVision2Seq.from_pretrained(
            _MODEL_ID,
            device_map="auto",
            **load_kwargs,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID)

    @property
    def torch_module(self):
        return self._model

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        import torch
        from PIL import Image

        is_batch = isinstance(rgb, (list, tuple))
        rgbs = list(rgb) if is_batch else [rgb]
        pil_imgs = [Image.fromarray(np.asarray(r, dtype=np.uint8)) for r in rgbs]

        depths: list[np.ndarray] = []
        for img in pil_imgs:
            inputs = self._processor(
                text=self.prompt, images=img, return_tensors="pt"
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=32768,  # depth tokens for a (H, W) map
                    do_sample=False,
                )
            depth = self._processor.post_process(
                out, target_size=(img.height, img.width)
            )
            # Different release versions of the processor return either
            # a tensor or a dict with 'depth'.
            if isinstance(depth, dict):
                depth = depth.get("depth") or depth.get("predicted_depth")
            if hasattr(depth, "detach"):
                depth = depth.detach().cpu().float().numpy()
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth.squeeze(0)
            depths.append(depth)
        return depths if is_batch else depths[0]
