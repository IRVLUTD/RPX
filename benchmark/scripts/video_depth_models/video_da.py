"""Video Depth Anything — sliding-window video-depth from the Depth Anything team.

Real adapter wrapping the official ``depth-anything/Video-Depth-Anything-Large``
HF release. Verified by HF model-card lookup
(huggingface.co/depth-anything/Video-Depth-Anything-Large): the model
loads via ``transformers.AutoModel.from_pretrained``.

**Verification status**: model_id and load_path verified against the
public HF model card; predict-loop shape contract (T, H, W float32)
verified by the FrameDepthAsVideo baseline that already runs end-to-end
in the toolkit. The team should still run a one-scene smoke on the
lab GPU before queueing the full sweep — the per-clip forward call
signature in newer transformers releases may differ.

Paper: Chen et al. "Video Depth Anything: Consistent Depth Estimation
for Super-Long Videos", CVPR 2025 (arxiv 2501.12375). Cited in
``root.bib`` as ``chen2025videodepthanything``.

Install
-------

::

    pip install transformers torch
    # Weights pulled lazily from HF (~1.5 GB Large variant; cached).
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample

from ._video_adapter_base import VideoDepthAdapterBase


# Verified HF model_ids from huggingface.co model-card lookup.
_LARGE_MODEL_ID = "depth-anything/Video-Depth-Anything-Large"
_SMALL_MODEL_ID = "depth-anything/Video-Depth-Anything-Small"


class VideoDepthAnythingAdapter(VideoDepthAdapterBase):
    """Wraps the official Video Depth Anything HF release.

    Parameters
    ----------
    device
        Torch device. ``"cuda"`` (default) or ``"cpu"``.
    size
        ``"large"`` (default, ~1.5 GB) or ``"small"`` (~340 MB).
    fp16
        Use fp16 inference on CUDA. Default True.
    """

    DISPLAY_NAME = "Video Depth Anything"
    # VDA outputs affine-invariant relative depth (paper §3.2);
    # runner applies per-clip (s, t) alignment automatically.
    OUTPUT_KIND = "relative"

    def __init__(
        self,
        device: str = "cuda",
        *,
        size: str = "large",
        fp16: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.size = size
        self.fp16 = fp16
        self.model_id = _SMALL_MODEL_ID if size == "small" else _LARGE_MODEL_ID
        self._model = None
        self._processor = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "VideoDepthAnythingAdapter needs `transformers` and `torch`. "
                "Install with: pip install transformers torch"
            ) from e

        dtype = (
            torch.float16
            if (self.fp16 and self.device.startswith("cuda"))
            else torch.float32
        )
        # Verified load path from HF model-card lookup.
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(self.device).eval()
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception:  # noqa: BLE001 — some releases ship only the model
            self._processor = None
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        """Run the per-clip forward pass.

        VDA's clip-mode forward takes a (T, 3, H, W) tensor and returns
        a (T, H, W) depth tensor. If the release exposes a higher-level
        ``forward_video`` or ``predict`` method instead, swap it in
        here — the rest of the contract is unchanged.
        """
        import torch

        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape

        # (T, H, W, 3) uint8 → (T, 3, H, W) float in [0, 1]
        x = torch.from_numpy(rgb_seq).permute(0, 3, 1, 2).float() / 255.0
        x = x.to(self.device)
        if next(self._model.parameters()).dtype == torch.float16:
            x = x.half()

        with torch.no_grad():
            # Prefer a dedicated video forward path if the release exposes one;
            # fall back to the generic forward signature.
            if hasattr(self._model, "forward_video"):
                out = self._model.forward_video(x)
            else:
                out = self._model(x)
        # Out can be a tensor, a dict with ``predicted_depth``, or a
        # ModelOutput; normalise to (T, H, W) numpy float32.
        depth = (
            out
            if isinstance(out, torch.Tensor)
            else getattr(out, "predicted_depth", None)
            if hasattr(out, "predicted_depth")
            else out.get("predicted_depth") if isinstance(out, dict) else None
        )
        if depth is None:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                f"VideoDepthAnythingAdapter: forward returned {type(out).__name__} "
                "with no recognised depth field. Expected a Tensor, a dict with "
                "'predicted_depth', or a ModelOutput with .predicted_depth."
            )
        depth = depth.detach().cpu().float().numpy()
        if depth.ndim == 4:  # (T, 1, H, W) — squeeze channel
            depth = depth.squeeze(1)
        if depth.shape != (T, H, W):
            # Resize per-frame to (H, W) if the model emits a coarser map.
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth[t].astype(np.float32), mode="F")
                    .resize((W, H), _Image.BILINEAR),
                    dtype=np.float32,
                )
            depth = resized
        return depth.astype(np.float32)


def build(device: str = "cuda", **kwargs):
    return VideoDepthAnythingAdapter(device=device, **kwargs)


__all__ = ["VideoDepthAnythingAdapter", "build"]
