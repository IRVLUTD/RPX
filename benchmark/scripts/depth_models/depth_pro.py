"""Depth Pro adapter (Apple, https://arxiv.org/abs/2410.02073).

Loads ``apple/DepthPro-hf`` via HuggingFace transformers' depth-estimation
pipeline. Output is metric depth in metres at the input resolution
(the pipeline upsamples internally).

Tracker reference: MonocularMetricDepth sheet, row "Depth Pro".

Install
-------
    pip install transformers torch timm pillow

Usage
-----
    from depth_models.depth_pro import DepthPro
    import rpx_benchmark as rpx

    dp = DepthPro(device="cuda", batch_size=8)
    model = rpx.make_numpy_depth_model(dp, name="DepthPro")
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


class DepthPro:
    """Callable adapter: rgb (H×W×3 uint8) → depth (H×W float32, metres).

    Supports batched dispatch: pass a ``list[np.ndarray]`` and the
    underlying HF pipeline executes one batched forward.
    """

    DEFAULT_MODEL_ID = "apple/DepthPro-hf"

    #: Apple Depth Pro outputs metric depth in metres — but the metric
    #: scale depends on the *focal length* (or equivalent FOV). Without
    #: a focal-length override, Depth Pro's auto-estimated FOV on RPX
    #: (640×480 from a D435) lands ~10° wider than the true sensor
    #: (~55°), inflating depth ~30%. We pass the known D435 RGB focal
    #: at construction time so the metric scale matches RPX's GT.
    native_alignment: str = "none"

    #: D435 RGB factory intrinsics at 640×480. `fx` is the pixel focal
    #: length on the horizontal axis. Cross-checked against
    #: `scripts/generate_keypoint_pairs.py:DEFAULT_INTRINSICS` (605.0)
    #: and `scripts/visualize_rerun.py` (617.0); 615 is the consensus
    #: factory value for the D435 RGB stream.
    D435_RGB_FOCAL_PX: float = 615.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
        focal_length_px: Optional[float] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise ImportError(
                "DepthPro needs `transformers`. Install with: "
                "pip install transformers torch timm pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = torch
        # Default to the D435 RGB focal so the metric scale matches RPX's
        # GT out of the box; pass `focal_length_px` explicitly to override.
        self.focal_length_px = (
            float(focal_length_px) if focal_length_px is not None else self.D435_RGB_FOCAL_PX
        )

        # We DO NOT use ``transformers.pipeline("depth-estimation")`` for
        # Depth Pro: the pipeline returns the raw head output (focal-
        # normalised inverse-depth) instead of running
        # ``post_process_depth_estimation``, which is what converts the
        # head output into metric metres. Calling the processor + model
        # directly avoids the silent-units bug and gives us the metric
        # number Apple's paper reports.
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id)
        if dtype:
            target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            self._model = self._model.to(dtype=target_dtype)
        self._model = self._model.to(device).eval()

    @property
    def torch_module(self):
        return self._model

    def __call__(
        self,
        rgb: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Single image OR list of images → matching metric depth (metres).

        Uses ``image_processor.post_process_depth_estimation`` which
        converts raw model output → metric depth via Depth Pro's
        published focal-length / scale post-process. The list path
        dispatches a single batched GPU forward.
        """
        from PIL import Image

        torch = self._torch

        is_batch = isinstance(rgb, (list, tuple))
        rgbs = list(rgb) if is_batch else [rgb]
        for r in rgbs:
            r = np.asarray(r)
            if r.ndim != 3 or r.shape[2] != 3:
                from rpx_benchmark.exceptions import AdapterError

                raise AdapterError(
                    f"expected H×W×3 RGB uint8, got shape {r.shape}",
                    hint="Adapter contract: each input must be a (H, W, 3) "
                    "uint8 numpy array. Got an unexpected ndim or channel count.",
                )

        pil_imgs = [Image.fromarray(np.asarray(r, dtype=np.uint8)) for r in rgbs]
        target_sizes = [(np.asarray(r).shape[0], np.asarray(r).shape[1]) for r in rgbs]

        inputs = self._processor(images=pil_imgs, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # Override the model's auto-estimated FOV with the true D435
        # FOV. The post-process scales depth by focal-length / fov, so a
        # wrong FOV silently scales the metric output. Each batch item
        # gets the same FOV (single-camera dataset).
        # FOV (degrees) = 2 * atan(W / 2*fx) * 180/π
        import math as _math

        # Use the maximum image width across the batch — RPX is uniform
        # 640 wide so this is just 640.
        widths = [t[1] for t in target_sizes]
        fovs = []
        for w in widths:
            fov_deg = 2.0 * _math.degrees(_math.atan(w / (2.0 * self.focal_length_px)))
            fovs.append(fov_deg)
        outputs.field_of_view = torch.tensor(
            fovs,
            dtype=outputs.field_of_view.dtype,
            device=outputs.field_of_view.device,
        )

        post = self._processor.post_process_depth_estimation(
            outputs,
            target_sizes=target_sizes,
        )

        depths: list[np.ndarray] = []
        for r, p in zip(rgbs, post, strict=False):
            d = p["predicted_depth"].detach().cpu().numpy().astype(np.float32)
            if d.ndim == 3:
                d = d.squeeze(0)
            target_hw = np.asarray(r).shape[:2]
            if d.shape != target_hw:
                d = _resize_bilinear(d, target_hw)
            depths.append(d)
        return depths if is_batch else depths[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float array to (H, W). PIL-only; no OpenCV dep."""
    from PIL import Image

    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
