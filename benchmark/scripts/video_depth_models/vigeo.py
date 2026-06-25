"""ViGeo — consistent video geometry estimation (depth + normals + pose).

Real adapter wrapping the official ``pkqbajng/ViGeo`` HF release.
Verified-official: the HF model card explicitly links to the paper
("Towards Consistent Video Geometry Estimation", arxiv 2605.30060)
and the upstream github repo (github.com/aigc3d/ViGeo). The author
handle ``pkqbajng`` matches both the HF repo owner and the project
page (pkqbajng.github.io/ViGeo).

ViGeo is a multi-output video model — predicts per-frame depth + 3D
points + surface normals + camera poses. For Video Depth evaluation we
extract the depth output.

**Verification status**: model_id, paper, GitHub, and load incantation
all verified against the upstream HF model card on 2026-06-25. The
team should still smoke on one clip before queueing the full sweep —
predict-output schema may have drifted between releases.

Install
-------

::

    git clone https://github.com/aigc3d/ViGeo
    cd ViGeo && pip install -r requirements.txt && pip install -e .
    # The HF model card documents: Python 3.10, PyTorch 2.7.1.
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import VideoSample
from rpx_benchmark.exceptions import AdapterError

from ._video_adapter_base import VideoDepthAdapterBase


# Verified model_id from HF model card.
_MODEL_ID = "pkqbajng/ViGeo"


class ViGeoAdapter(VideoDepthAdapterBase):
    """Wraps the official ViGeo HF release.

    Three inference modes per the upstream README — ``offline`` (sees
    whole clip), ``chunk`` (sliding window), ``online`` (streaming).
    Default ``offline`` since RPX clips are bounded (~250 frames) and
    we want the fairest depth-quality comparison.
    """

    DISPLAY_NAME = "ViGeo"
    # ViGeo predicts metric-ish depth + camera pose; per the paper §4
    # the scale is determined jointly with pose, so we conservatively
    # treat the depth as relative (runner applies per-clip alignment).
    OUTPUT_KIND = "relative"

    def __init__(self, device: str = "cuda", *, mode: str = "offline") -> None:
        super().__init__(device=device)
        self.mode = mode
        self._model = None

    def setup(self) -> None:
        if self._loaded:
            return
        try:
            # Upstream module path per HF model card's quick-start code.
            from vigeo import ViGeo
        except ImportError as e:
            raise ImportError(
                "ViGeoAdapter needs the upstream `vigeo` package. "
                "Install with:\n"
                "    git clone https://github.com/aigc3d/ViGeo\n"
                "    cd ViGeo && pip install -r requirements.txt && pip install -e ."
            ) from e

        # Verified load incantation from the HF model card's quick-start:
        #     model = ViGeo.from_pretrained("pkqbajng/ViGeo").to(device).eval()
        self._model = ViGeo.from_pretrained(_MODEL_ID).to(self.device).eval()
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape

        # Upstream API per the HF quick-start: model.infer(images, mode=...)
        # returns a dict with depth, points, normals, confidence, pose.
        out = self._model.infer([rgb_seq[t] for t in range(T)], mode=self.mode)

        depth_seq = None
        if isinstance(out, dict):
            depth_seq = out.get("depth") or out.get("depth_map")
        elif hasattr(out, "depth"):
            depth_seq = out.depth
        if depth_seq is None:
            raise AdapterError(
                "ViGeo.infer() returned no recognised depth field. "
                "Expected dict with 'depth' or attribute .depth. "
                f"Got: {type(out).__name__}"
            )

        if hasattr(depth_seq, "detach"):
            depth_seq = depth_seq.detach().cpu().float().numpy()
        depth_seq = np.asarray(depth_seq, dtype=np.float32)
        if depth_seq.ndim == 4:  # (T, 1, H, W) → squeeze channel
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
        return depth_seq


def build(device: str = "cuda", **kwargs):
    return ViGeoAdapter(device=device, **kwargs)


__all__ = ["ViGeoAdapter", "build"]
