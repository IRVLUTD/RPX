"""Video Depth Anything — sliding-window video-depth from the Depth Anything team.

Real adapter wrapping the official Video Depth Anything release. The
HF model card (huggingface.co/depth-anything/Video-Depth-Anything-Large)
ships weights but **not** a transformers-compatible ``config.json`` —
the upstream README directs callers to clone the repository and run
inference via the bundled CLI (``run.py``). This adapter wraps that
inference pattern.

**Verification status**: model_id verified against the HF model card;
load_path requires the github clone (verified by smoke run on
2026-06-25 — the bare ``AutoModel.from_pretrained`` path fails with
"Unrecognized model" because the config doesn't declare
``model_type``). The clone + module-level import path below matches
the upstream README.

Paper: Chen et al. "Video Depth Anything: Consistent Depth Estimation
for Super-Long Videos", CVPR 2025 (arxiv 2501.12375). Cited in
``root.bib`` as ``chen2025videodepthanything``.

Install
-------

::

    git clone https://github.com/DepthAnything/Video-Depth-Anything.git
    cd Video-Depth-Anything
    pip install -r requirements.txt
    pip install -e .
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
        """Load the model via the upstream github package.

        VDA's HF release does NOT ship a transformers-compatible
        ``config.json``; the canonical inference path is via the
        ``video_depth_anything`` package built from
        ``github.com/DepthAnything/Video-Depth-Anything``.
        """
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VideoDepthAnythingAdapter needs `torch`. "
                "Install with: pip install torch"
            ) from e

        try:
            # Upstream module path per the github README.
            from video_depth_anything.video_depth import VideoDepthAnything
        except ImportError as e:
            raise ImportError(
                "VideoDepthAnythingAdapter needs the upstream "
                "`video_depth_anything` package. Install with:\n"
                "    git clone https://github.com/DepthAnything/Video-Depth-Anything\n"
                "    cd Video-Depth-Anything && pip install -e ."
            ) from e

        # Encoder size — upstream supports 'vits' and 'vitl'.
        encoder = "vits" if self.size == "small" else "vitl"

        # Build the model and load the weights. The HF release hosts a
        # state_dict; the upstream Python class loads it via
        # ``model.load_state_dict``.
        self._model = VideoDepthAnything(encoder=encoder)
        try:
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(
                repo_id=self.model_id,
                filename=(
                    "video_depth_anything_vits.pth"
                    if encoder == "vits"
                    else "video_depth_anything_vitl.pth"
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"VideoDepthAnythingAdapter: could not download weights "
                f"from {self.model_id} via huggingface_hub: {e}"
            ) from e

        state = torch.load(ckpt_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model = self._model.to(self.device).eval()
        if self.fp16 and self.device.startswith("cuda"):
            self._model = self._model.half()
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        """Run the per-clip forward pass via VDA's ``infer_video_depth``.

        Upstream exposes a higher-level helper that takes a list of
        BGR/RGB numpy frames and returns a (T, H, W) depth tensor.
        """
        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape

        # The upstream API: ``infer_video_depth(frames, target_fps,
        # input_size, device, fp32)``. We pass each (H, W, 3) uint8
        # frame as-is (the helper handles the rest).
        if hasattr(self._model, "infer_video_depth"):
            depth_seq, _meta = self._model.infer_video_depth(
                [rgb_seq[t] for t in range(T)],
                target_fps=30,
                input_size=518,  # upstream default
                device=self.device,
                fp32=(not self.fp16),
            )
        else:
            from rpx_benchmark.exceptions import AdapterError

            raise AdapterError(
                "VideoDepthAnythingAdapter: model has no "
                "infer_video_depth method. Upstream API may have "
                "moved; check the video_depth_anything release version."
            )

        depth_seq = np.asarray(depth_seq, dtype=np.float32)
        if depth_seq.ndim == 4:
            depth_seq = depth_seq.squeeze(1)
        if depth_seq.shape != (T, H, W):
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth_seq[t].astype(np.float32), mode="F")
                    .resize((W, H), _Image.BILINEAR),
                    dtype=np.float32,
                )
            depth_seq = resized
        return depth_seq.astype(np.float32)


def build(device: str = "cuda", **kwargs):
    return VideoDepthAnythingAdapter(device=device, **kwargs)


__all__ = ["VideoDepthAnythingAdapter", "build"]
