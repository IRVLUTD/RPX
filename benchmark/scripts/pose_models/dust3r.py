"""DUSt3R adapter — joint 3D pointmap + relative pose (CVPR 2024).

Loads ``naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`` via the ``dust3r``
package. DUSt3R predicts dense per-pixel 3D points in each view's
frame; relative pose is recovered from a global alignment over the
two pointmaps.

Tracker reference: RCPE — foundational regression model.

Install
-------
    pip install dust3r torch pillow
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._pose_base import coerce_pose_output, validate_pair


class DUSt3R:
    """rgb pair → 4×4 SE(3) recovered from joint 3D pointmaps."""

    DEFAULT_MODEL_ID = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    # Original DUSt3R global alignment is similarity-scale ambiguous.
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
            from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "DUSt3R needs the `dust3r` package. Install with: pip install dust3r torch pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch

        try:
            model = AsymmetricCroCo3DStereo.from_pretrained(model_id)
        except Exception as e:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"DUSt3R load failed for {model_id!r}: {e}",
                hint="Check the latest checkpoint id at https://huggingface.co/naver",
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
            from dust3r.cloud_opt import (  # type: ignore[import-not-found]
                GlobalAlignerMode,
                global_aligner,
            )
            from dust3r.image_pairs import make_pairs  # type: ignore[import-not-found]
            from dust3r.inference import inference  # type: ignore[import-not-found]
            from dust3r.utils.image import load_images_from_arrays  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "DUSt3R inference helpers missing — likely partial install. "
                "Reinstall: pip install --force-reinstall dust3r"
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
                # `aligner.get_im_poses()` → [N, 4, 4] camera-to-world.
                # Relative pose from view-0 to view-1.
                poses = aligner.get_im_poses().detach().cpu().numpy()
                T_rel = np.linalg.inv(poses[0]) @ poses[1]
                outs.append(coerce_pose_output(T_rel[:3, :3], T_rel[:3, 3]))
            except Exception as e:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"DUSt3R inference failed on a pair: {e}",
                ) from e
        return outs
