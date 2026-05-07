"""ZoeDepth adapter (Intel, https://arxiv.org/abs/2302.12288).

Loads the joint NYU + KITTI checkpoint (``Intel/zoedepth-nyu-kitti``) via
HuggingFace ``transformers`` so we don't pull a fragile GitHub clone. Output is
metric depth in metres at the input resolution (the pipeline upsamples).

Tracker reference: `MonocularMetricDepthEstimation` sheet, row 0.

Install
-------
    pip install transformers torch timm pillow

Usage
-----
    from depth_models.zoedepth import ZoeDepth
    import rpx_benchmark as rpx

    zoe = ZoeDepth(device="cuda")            # one-shot weight load
    model = rpx.make_numpy_depth_model(zoe, name="ZoeDepth_NK")
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ZoeDepth:
    """Callable adapter: rgb (H×W×3 uint8) → depth (H×W float32, metres)."""

    DEFAULT_MODEL_ID = "Intel/zoedepth-nyu-kitti"

    #: Read by ``BatchedDepthBenchmarkModel`` to pick the runner's
    #: alignment default. ``"none"`` for metric models that publish in
    #: metres; ``"ls_affine"`` for up-to-scale (Marigold/Lotus); also
    #: legal: ``"median"``, ``"ls_disparity"``.
    native_alignment: str = "none"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: Optional[str] = None,
        batch_size: int = 1,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "ZoeDepth needs `transformers`. Install with: "
                "pip install transformers torch timm pillow"
            ) from e
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        # HF's depth-estimation pipeline default is batch_size=1; we pass
        # ``batch_size`` so a list input dispatches as a single batched
        # GPU forward instead of the sequential warning path. The model
        # weights and forward semantics are unchanged — just the dispatch.
        # Pipeline returns: {'predicted_depth': torch.Tensor [1,H,W] (metres),
        #                    'depth': PIL.Image (vis)}
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
            torch_dtype=dtype,                 # e.g. "float16"
            batch_size=self.batch_size,
        )

    @property
    def torch_module(self):
        # Exposed so the benchmark profiler can walk into the actual nn.Module
        # for params / FLOPs / memory-traffic counting.
        return getattr(self._pipe, "model", None)

    def __call__(self, rgb):
        """Single image (H×W×3 uint8) → H×W float32 depth, OR a list of
        such images → list of depth arrays. The HF transformers pipeline
        natively handles batched input via ``pipeline(list_of_images)``.

        Batching does NOT modify the model — just the dispatch path.
        """
        from PIL import Image

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
        out = self._pipe(pil_imgs)              # one batched forward
        if not isinstance(out, list):
            out = [out]

        depths = []
        for r, o in zip(rgbs, out):
            d = o["predicted_depth"].detach().cpu().numpy().astype(np.float32)
            if d.ndim == 3:
                d = d.squeeze(0)
            target_hw = np.asarray(r).shape[:2]
            if d.shape != target_hw:
                d = _resize_bilinear(d, target_hw)
            depths.append(d)
        return depths if is_batch else depths[0]


def _resize_bilinear(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float array to (H, W) without an OpenCV dependency."""
    from PIL import Image
    img = Image.fromarray(src.astype(np.float32), mode="F")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
