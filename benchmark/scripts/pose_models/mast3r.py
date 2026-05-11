"""MASt3R adapter — matching-aware DUSt3R extension (2024).

Loads ``naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric``.
MASt3R adds a feature-matching head atop DUSt3R, yielding much
better correspondences on small-baseline pairs.

Install
-------
    pip install mast3r torch pillow
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


class MASt3R:
    """rgb pair → 4×4 SE(3) via matching-aware DUSt3R."""

    DEFAULT_MODEL_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    native_alignment: str = "none"
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
            from mast3r.model import AsymmetricMASt3R  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "MASt3R needs the `mast3r` package. Install with: pip install mast3r torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch

        try:
            model = AsymmetricMASt3R.from_pretrained(model_id)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"MASt3R load failed for {model_id!r}: {e}",
                hint="Check https://huggingface.co/naver for the latest MASt3R checkpoint name.",
            ) from e
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            model = model.to(dtype=target_dtype)
        self._model = model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(self, pairs: Sequence[dict]) -> list[dict]:
        try:
            from mast3r.cloud_opt import (  # type: ignore[import-not-found]
                GlobalAlignerMode,
                global_aligner,
            )
            from mast3r.image_pairs import make_pairs  # type: ignore[import-not-found]
            from mast3r.inference import inference  # type: ignore[import-not-found]
            from mast3r.utils.image import load_images_from_arrays  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "MASt3R inference helpers missing — partial install. "
                "Reinstall: pip install --force-reinstall mast3r"
            ) from e

        if not isinstance(pairs, (list, tuple)):
            pairs = [pairs]
        outs: list[dict] = []
        for pair in pairs:
            rgb_a, rgb_b = validate_pair(pair)
            try:
                imgs = load_images_from_arrays([rgb_a, rgb_b], size=512)
                view_pairs = make_pairs(
                    imgs, scene_graph="complete", prefilter=None, symmetrize=True
                )
                output = inference(view_pairs, self._model, self.device, batch_size=self.batch_size)
                aligner = global_aligner(
                    output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer
                )
                aligner.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
                poses = aligner.get_im_poses().detach().cpu().numpy()
                T_rel = np.linalg.inv(poses[0]) @ poses[1]
                outs.append(coerce_pose_output(T_rel[:3, :3], T_rel[:3, 3]))
            except Exception as e:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"MASt3R inference failed on a pair: {e}",
                ) from e
        return outs
