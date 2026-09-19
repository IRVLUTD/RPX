"""DA3 (video mode) — Depth Anything 3 fed per-clip.

Real adapter wrapping the official ``depth-anything/DA3-LARGE`` HF
release. Verified by HF model-card lookup
(huggingface.co/depth-anything/DA3-LARGE): the model uses ByteDance's
custom ``depth_anything_3`` Python API rather than transformers /
diffusers, loaded via ``DepthAnything3.from_pretrained``. Paper: DA3
(ICLR'26 Oral, ByteDance Seed Team).

DA3 is a multi-view model; the Video Depth path feeds the whole clip
as a multi-view set (the same weights as the Image Depth
``da3-metric-l`` adapter, just fed differently). The "Image Depth
mode" treats each frame independently; "Video Depth mode" lets the
multi-view backbone exploit cross-frame geometry.

**Verification status**: model_id and load incantation verified
against the upstream HF model card. The ``predict`` body is
best-effort against the publicly-documented API — the team should
smoke on one clip and adjust the kwargs if the upstream release has
moved on since the model card snapshot.

Install
-------

::

    pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample

from ._video_adapter_base import VideoDepthAdapterBase

# Verified model_id from HF model card.
_MODEL_ID = "depth-anything/DA3-LARGE"


class DA3VideoAdapter(VideoDepthAdapterBase):
    """Wraps ``depth-anything/DA3-LARGE`` in video (multi-view) mode."""

    DISPLAY_NAME = "DA3"
    # DA3-LARGE emits relative any-view depth; align once per clip.
    OUTPUT_KIND = "relative"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self._model = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "DA3VideoAdapter needs the `depth_anything_3` package. "
                "Install with: pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3"
            ) from e

        # Verified load path from HF model card.
        self._model = DepthAnything3.from_pretrained(_MODEL_ID)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        """Feed the whole clip to DA3's multi-view forward.

        DA3's public API exposes a ``predict`` method that accepts a
        list of images (numpy or PIL) and returns a dict with
        ``depth`` keyed per-view. We extract that and stack.
        """
        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape
        frames = [rgb_seq[t] for t in range(T)]

        # Pinned DA3 exposes inference() for image lists/multi-view clips.
        output = self._model.inference(frames)

        # Normalise output: dict-with-depth, list-of-depth-arrays, or
        # ndarray (T, H, W).
        depth = output.depth if hasattr(output, "depth") else None
        if depth is None and isinstance(output, dict):
            depth = output.get("depth")
            if depth is None:
                depth = output.get("depth_map")
        elif depth is None and isinstance(output, (list, tuple)):
            depth = np.stack([np.asarray(d, dtype=np.float32) for d in output])
        elif depth is None and hasattr(output, "detach"):
            depth = output.detach().cpu().float().numpy()

        if depth is None:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError("DA3 video inference returned no depth field.")

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim == 2 and T == 1:
            depth = depth[None]
        if depth.ndim != 3 or depth.shape[0] != T:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"DA3 video returned depth shape {depth.shape}; expected T={T}."
            )

        if depth.shape[1:] != (H, W):
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth[t].astype(np.float32), mode="F")
                    .resize((W, H), _Image.BILINEAR),
                    dtype=np.float32,
                )
            depth = resized
        return depth


def build(device: str = "cuda", **kwargs):
    return DA3VideoAdapter(device=device, **kwargs)


__all__ = ["DA3VideoAdapter", "build"]
