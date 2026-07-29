"""Official ZipDepth monocular inverse-depth adapter.

ZipDepth predicts affine-invariant inverse depth. This adapter preserves that
official output and declares ``ls_disparity``. The runner performs one pooled
scale-and-shift fit to GT disparity per ``(scene, phase)`` cell and converts
the aligned disparity to metric depth, matching ZipDepth's released evaluator
without using per-frame ground-truth alignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

from rpx_benchmark.exceptions import AdapterError


class ZipDepthAdapter:
    """ZipDepth-base GPU checkpoint at the pinned official revision."""

    DEFAULT_SOURCE_PATH = "/opt/rpx-envs/sources/zipdepth"
    DEFAULT_CHECKPOINT = "checkpoints/zipdepth_base.pth"
    UPSTREAM_REVISION = "94da7527f7030a0e79d54f33b113bdce4065d735"
    CHECKPOINT_SHA256 = (
        "a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65"
    )

    native_alignment = "ls_disparity"
    native_precision = "fp32"

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        *,
        source_path: str = DEFAULT_SOURCE_PATH,
        checkpoint_path: str | None = None,
        input_size: int = 384,
    ) -> None:
        if int(batch_size) != 1:
            raise AdapterError("ZipDepth RPX evaluation requires batch size 1.")
        if device == "cpu":
            raise AdapterError("ZipDepth production inference requires CUDA.")

        source = Path(source_path)
        checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else source / self.DEFAULT_CHECKPOINT
        )
        if not checkpoint.is_file():
            raise AdapterError(
                f"Pinned ZipDepth checkpoint is missing: {checkpoint}",
                hint="Build docker/depth-zipdepth/Dockerfile before running ZipDepth.",
            )

        try:
            from zipdepth.inference.predictor import DepthInference
        except ImportError as exc:
            raise AdapterError(
                f"Official ZipDepth environment is unavailable: {exc}",
                hint="Run with /opt/rpx-envs/zipdepth/bin/python from the ZipDepth overlay.",
            ) from exc

        try:
            self._predictor = DepthInference(
                checkpoint_path=str(checkpoint),
                variant="base",
                device=device,
                use_half=False,
                use_compile=False,
                input_size=int(input_size),
                ensure_multiple_of=32,
                warmup_iters=3,
                upsample_unfold=True,
            )
        except Exception as exc:
            raise AdapterError(f"Official ZipDepth load failed: {exc}") from exc

        self.device = device
        self.batch_size = 1
        self.input_size = int(input_size)
        self.model_id = (
            "https://github.com/fabiotosi92/ZipDepth"
            f"@{self.UPSTREAM_REVISION}:{self.DEFAULT_CHECKPOINT}"
        )

    @property
    def torch_module(self):
        return getattr(self._predictor, "model", None)

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        is_batch = isinstance(rgb, (list, tuple))
        images = list(rgb) if is_batch else [rgb]
        outputs = [self._infer_one(image) for image in images]
        return outputs if is_batch else outputs[0]

    def _infer_one(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise AdapterError(f"ZipDepth expected HxWx3 RGB, got {image.shape}.")
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
                raise AdapterError("ZipDepth RGB input must be finite numeric data.")
            image = np.clip(image, 0, 255).astype(np.uint8)

        # The official predictor accepts OpenCV-order BGR uint8.
        bgr = np.ascontiguousarray(image[..., ::-1])
        inverse_depth = np.asarray(
            self._predictor.infer_image(bgr),
            dtype=np.float32,
        )
        if inverse_depth.shape != image.shape[:2]:
            raise AdapterError(
                "ZipDepth did not restore the original image resolution: "
                f"expected {image.shape[:2]}, got {inverse_depth.shape}."
            )
        if not np.isfinite(inverse_depth).all():
            raise AdapterError("ZipDepth returned non-finite inverse depth.")
        if np.any(inverse_depth < 0):
            raise AdapterError("ZipDepth returned negative inverse depth.")
        if float(np.ptp(inverse_depth)) <= 1e-6:
            raise AdapterError("ZipDepth returned a degenerate inverse-depth map.")

        # Preserve the native inverse-depth map, including legitimate zeros
        # from the official ReLU head. The pooled ls_disparity fit excludes
        # zero predictions exactly as the released ZipDepth evaluator does.
        return np.asarray(inverse_depth, dtype=np.float32)
