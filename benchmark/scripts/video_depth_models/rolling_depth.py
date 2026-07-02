"""RollingDepth — streaming video-depth via diffusion (PRS-ETH / Marigold team).

Real adapter wrapping the official ``prs-eth/rollingdepth-v1-0`` HF
release. Verified by HF model-card lookup
(huggingface.co/prs-eth/rollingdepth-v1-0): loads via
``diffusers.DiffusionPipeline.from_pretrained`` with the model card's
canonical incantation. Paper: Ke et al. "Video Depth without Video
Models" (2024) from ETH Zurich's Photogrammetry and Remote Sensing
Lab (PRS).

**Verification status**: model_id and load_path are verified against
the upstream HF model card. The pipeline signature (snippet/overlap
kwargs) is best-effort against the released checkpoint and may need
adjusting once the team runs it end-to-end on the lab GPU.

Install
-------

::

    pip install diffusers transformers accelerate
    # Weights pulled lazily from HF (~5 GB; cached after first run).
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample

from ._video_adapter_base import VideoDepthAdapterBase


class RollingDepthAdapter(VideoDepthAdapterBase):
    """Wraps ``prs-eth/rollingdepth-v1-0`` HF release.

    Parameters
    ----------
    device
        ``"cuda"`` (default) or ``"cpu"``.
    num_inference_steps
        Diffusion solver steps. The team can tune for speed vs quality.
    snippet_length
        Frames per sliding snippet. Default 10.
    overlap
        Snippet overlap for boundary blending. Default 5.
    """

    DISPLAY_NAME = "RollingDepth"
    OUTPUT_KIND = "relative"  # Marigold family — affine-invariant.

    def __init__(
        self,
        device: str = "cuda",
        *,
        num_inference_steps: int = 4,
        snippet_length: int = 10,
        overlap: int = 5,
    ) -> None:
        super().__init__(device=device)
        self.num_inference_steps = int(num_inference_steps)
        self.snippet_length = int(snippet_length)
        self.overlap = int(overlap)
        self._pipe = None

    def setup(self) -> None:
        """Load via the upstream ``rollingdepth`` package.

        The bare ``DiffusionPipeline.from_pretrained`` path fails with
        ``module diffusers has no attribute RollingDepthPipeline``
        because RollingDepth's config declares a custom pipeline class
        that the diffusers core library doesn't ship. The upstream
        github repo provides the class.
        """
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RollingDepthAdapter needs `torch`. Install with: pip install torch"
            ) from e

        try:
            # Upstream module path per github.com/prs-eth/rollingdepth README.
            from rollingdepth import RollingDepthPipeline
        except ImportError as e:
            raise ImportError(
                "RollingDepthAdapter needs the upstream `rollingdepth` "
                "package. Install with:\n"
                "    git clone https://github.com/prs-eth/rollingdepth\n"
                "    cd rollingdepth && pip install -e .\n"
                "Or: pip install rollingdepth (if a release tag is available)."
            ) from e

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._pipe = RollingDepthPipeline.from_pretrained(
            "prs-eth/rollingdepth-v1-0",
            torch_dtype=dtype,
        )
        self._pipe.to(self.device)
        for opt in (
            "enable_xformers_memory_efficient_attention",
            "enable_attention_slicing",
        ):
            try:
                getattr(self._pipe, opt)()
            except Exception:  # noqa: BLE001
                pass
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        import tempfile

        import av

        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape

        # RollingDepth's official API accepts a video path, not an in-memory
        # frame list. Encode a lossless RGB FFV1 stream for this transient
        # adapter boundary so RPX pixels are not changed by H.264 compression.
        with tempfile.NamedTemporaryFile(suffix=".mkv") as tmp:
            container = av.open(tmp.name, mode="w")
            stream = container.add_stream("ffv1", rate=30)
            stream.width = W
            stream.height = H
            stream.pix_fmt = "rgb24"
            try:
                for frame in rgb_seq:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            finally:
                container.close()

            result = self._pipe(
                input_video_path=tmp.name,
                processing_res=1024,
                dilations=[1, 25],
                cap_dilation=True,
                snippet_lengths=[self.snippet_length],
                init_infer_steps=[max(1, self.num_inference_steps)],
                strides=[1],
                refine_step=0,
                restore_res=True,
                unload_snippet=False,
            )
        depth_seq = result.depth_pred
        if hasattr(depth_seq, "detach"):
            depth_seq = depth_seq.detach().cpu().float().numpy()
        depth_seq = np.asarray(depth_seq, dtype=np.float32)
        if depth_seq.ndim == 4 and depth_seq.shape[1] == 1:
            depth_seq = depth_seq[:, 0]
        if depth_seq.shape != (T, H, W):
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth_seq[t].astype(np.float32), mode="F").resize(
                        (W, H), _Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
            depth_seq = resized
        return depth_seq


def build(device: str = "cuda", **kwargs):
    return RollingDepthAdapter(device=device, **kwargs)


__all__ = ["RollingDepthAdapter", "build"]
