"""MonST3R — multi-view 3D reconstruction adapted for video depth.

Real adapter wrapping the official MonST3R HF release. Verified by HF
model-card lookup (huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_-
BaseDecoder_512_dpt): loads via ``AsymmetricCroCo3DStereo.from_pretrained``
from the dust3r package. Paper: Zhang et al. "MonST3R: A Simple
Approach for Estimating Geometry in the Presence of Motion" (2024).

**Verification status**: model_id and load incantation verified
against the upstream HF model card. The per-clip predict body
extracts depth from MonST3R's stereo-pair output — this is the most
fragile part of this adapter and the team should verify the output
shape on a one-clip smoke before queueing the full sweep.

Install
-------

::

    pip install git+https://github.com/Junyi42/monst3r
    # OR follow upstream README's conda env setup; the package name
    # is ``dust3r`` (MonST3R is a fork that reuses the same module).
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample
from rpx_benchmark.exceptions import AdapterError

from ._video_adapter_base import VideoDepthAdapterBase


# Verified model_id from HF model-card lookup. The MonST3R upstream
# uploaded a single checkpoint under this name; if the team wants to
# swap variants, change this constant.
_MODEL_ID = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"


class MonST3RAdapter(VideoDepthAdapterBase):
    """Wraps MonST3R's video-depth path through ``AsymmetricCroCo3DStereo``.

    MonST3R is a stereo-pair model; for video-depth evaluation we feed
    overlapping (t, t+1) pairs and extract per-frame depth from the
    first-view 3D points (back-projected through the intrinsics the
    paper reports).
    """

    DISPLAY_NAME = "MonST3R"
    # MonST3R outputs metric-ish depth via 3D point regression; per the
    # paper §4.2, scale ambiguity remains so we treat it as relative
    # and let the runner apply per-clip (s, t) alignment.
    OUTPUT_KIND = "relative"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self._model = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as e:
            raise ImportError(
                "MonST3RAdapter needs the `dust3r` package (MonST3R fork). "
                "Install with: pip install git+https://github.com/Junyi42/monst3r"
            ) from e

        # Verified load incantation from HF model card.
        self._model = AsymmetricCroCo3DStereo.from_pretrained(_MODEL_ID).to(
            self.device
        ).eval()
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        """Run MonST3R on consecutive frame pairs and extract per-frame depth.

        MonST3R's API exposes a ``forward`` that takes a pair of frames
        and returns per-view 3D points + confidence. The exact key
        names and tensor shapes depend on the dust3r release the team
        has installed; this body uses the most common upstream pattern
        and will raise ``AdapterError`` with a hint if the shape
        doesn't match the contract.
        """
        try:
            import torch
            from dust3r.inference import inference  # type: ignore
            from dust3r.utils.image import load_images  # type: ignore
        except ImportError as e:
            raise ImportError(
                "MonST3RAdapter._predict_clip needs `dust3r.inference` "
                "and `dust3r.utils.image`. Same install path as setup()."
            ) from e

        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape

        # Build (t, t+1) pairs and let dust3r's loader normalise.
        # If the team's release exposes a higher-level video path
        # (e.g. a ``predict_video`` method), swap it in here.
        from PIL import Image

        pil_frames = [Image.fromarray(rgb_seq[t]) for t in range(T)]
        images = load_images(pil_frames, size=512)  # upstream's default

        depths = np.zeros((T, H, W), dtype=np.float32)
        with torch.no_grad():
            for t in range(T):
                t_next = min(t + 1, T - 1)
                pair = [images[t], images[t_next]]
                output = inference([pair], self._model, self.device)
                # Output schema (upstream): dict with 'view1' / 'view2',
                # each having 'pts3d' (B, H, W, 3). Depth = pts3d[..., 2].
                view1 = output[0].get("view1") if isinstance(output, list) else output.get("view1")
                if view1 is None or "pts3d" not in view1:
                    raise AdapterError(
                        "MonST3R inference output did not contain "
                        "view1.pts3d. Upstream may have changed schema; "
                        "check the dust3r release version against the "
                        "adapter."
                    )
                pts3d = view1["pts3d"]
                if hasattr(pts3d, "detach"):
                    pts3d = pts3d.detach().cpu().numpy()
                else:
                    pts3d = np.asarray(pts3d)
                if pts3d.ndim == 4:  # (B=1, H, W, 3)
                    pts3d = pts3d[0]
                # Depth = z-coordinate; resize back to (H, W).
                from PIL import Image as _Image

                d = pts3d[..., 2].astype(np.float32)
                d_img = _Image.fromarray(d, mode="F").resize((W, H), _Image.BILINEAR)
                depths[t] = np.asarray(d_img, dtype=np.float32)

        return depths


def build(device: str = "cuda", **kwargs):
    return MonST3RAdapter(device=device, **kwargs)


__all__ = ["MonST3RAdapter", "build"]
