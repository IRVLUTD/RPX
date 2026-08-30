"""Reloc3r adapter — direct relative pose regression (CVPR 2025).

Loads ``siyan824/reloc3r-512`` (or ``-224``) via the ``reloc3r`` package
which is the official upstream. 25 ms inference at 512px on a single
GPU. Trained on 8M pairs.

Tracker reference: RCPE — Category A direct regression, top SOTA.

Install
-------
    pip install reloc3r torch pillow
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._pose_base import coerce_pose_output, validate_pair


class Reloc3r:
    """rgb pair → rotation and translation direction."""

    DEFAULT_MODEL_ID = "siyan824/reloc3r-512"

    # Reloc3r explicitly learns translation direction; motion averaging can
    # recover scale later, but a standalone two-view prediction is non-metric.
    native_alignment: str = "unit"
    native_precision: str = "fp16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 1,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from reloc3r import Reloc3rModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Reloc3r needs the `reloc3r` package. Install with: "
                "pip install reloc3r torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch

        try:
            model = Reloc3rModel.from_pretrained(model_id)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"Reloc3r load failed for {model_id!r}: {e}",
                hint="Check the latest checkpoint id at "
                "https://huggingface.co/siyan824 — Reloc3r ships the "
                "-512 (default) and -224 (faster) variants.",
            ) from e
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            model = model.to(dtype=target_dtype)
        self._model = model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        outs: list[dict] = []
        with self._torch.inference_mode():
            for pair in pairs:
                rgb_a, rgb_b = validate_pair(pair)
                try:
                    pose = self._model.infer_pair(rgb_a, rgb_b)
                    rot = pose["rotation"] if isinstance(pose, dict) else pose[0]
                    trans = pose["translation"] if isinstance(pose, dict) else pose[1]
                    outs.append(coerce_pose_output(rot, trans))
                except Exception as e:
                    from rpx_benchmark.exceptions import AdapterError

                    raise AdapterError(
                        f"Reloc3r inference failed on a pair: {e}",
                    ) from e
        return outs
