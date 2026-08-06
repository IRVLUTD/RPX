"""VGGT-Ω — Visual Geometry Grounded Transformer (Meta AI, CVPR 2025).

Real adapter wrapping the official ``facebook/VGGT-1B`` HF release.
Verified by HF model-card lookup (huggingface.co/facebook/VGGT-1B):
the model is the official VGGT checkpoint from Meta AI & Oxford
(arxiv 2503.11651). The "Ω" suffix in the paper roster denotes the
large 1B-parameter variant.

VGGT consumes a set of images (clip-as-set) and produces camera
parameters, point maps, and per-frame depth. Loaded via standard
transformers ``from_pretrained``.

**Verification status**: model_id verified. The forward signature is
the upstream ``model.forward`` returning a dict-like with
``depth_map`` keyed output. The team should smoke on one clip.

Install
-------

::

    pip install transformers torch
    # Weights pulled lazily from HF (~4 GB; cached).
"""

from __future__ import annotations

import os

import numpy as np

from rpx_benchmark.api import VideoSample

from ._video_adapter_base import VideoDepthAdapterBase

# Verified model_id from HF model card.
_MODEL_ID = "facebook/VGGT-1B"


class VGGTOmegaAdapter(VideoDepthAdapterBase):
    """Wraps ``facebook/VGGT-1B`` (the "Omega" / 1B-parameter variant).

    VGGT processes the clip as an unordered set of views and predicts
    per-frame depth, camera intrinsics, and point maps. We extract the
    depth-map output and stack into (T, H, W).
    """

    DISPLAY_NAME = "VGGT-Ω"
    # VGGT predicts metric-ish depth; per the paper §4.3 the scale
    # depends on the camera-parameter prediction so we conservatively
    # treat the output as relative (runner applies per-clip alignment).
    OUTPUT_KIND = "relative"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self._model = None

    def setup(self) -> None:
        """Load VGGT via the upstream ``vggt`` package.

        The bare ``AutoModel.from_pretrained`` path fails with
        ``Unrecognized model in facebook/VGGT-1B`` — the model card
        hosts state-dict weights but the config doesn't declare a
        transformers ``model_type``. The upstream repo provides the
        ``VGGT`` class that loads the weights.
        """
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VGGTOmegaAdapter needs `torch`. Install with: pip install torch"
            ) from e

        try:
            # Upstream module path per github.com/facebookresearch/vggt.
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise ImportError(
                "VGGTOmegaAdapter needs the upstream `vggt` package. "
                "Install with:\n"
                "    git clone https://github.com/facebookresearch/vggt\n"
                "    cd vggt && pip install -e ."
            ) from e

        try:
            # Official package uses PyTorchModelHubMixin and currently ships
            # ``model.pt``; delegating avoids hard-coding the wrong filename.
            self._model = VGGT.from_pretrained(_MODEL_ID)
        except Exception as e:
            raise RuntimeError(
                f"VGGTOmegaAdapter: could not download weights from {_MODEL_ID}: {e}"
            ) from e
        self._model = self._model.to(self.device).eval()
        self._loaded = True

    def _predict_clip(self, sample: VideoSample) -> np.ndarray:
        """Run VGGT on the clip-as-set, extract per-frame depth.

        VGGT's upstream forward signature accepts a (B, T, 3, H, W)
        image tensor and returns a dict containing 'depth_map' or
        'depth' (the key has varied across releases). If the team's
        installed release uses a different key, update the extraction
        block below.
        """
        import tempfile
        from contextlib import nullcontext

        import torch
        from PIL import Image
        from vggt.utils.load_fn import load_and_preprocess_images

        rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
        T, H, W, _ = rgb_seq.shape

        # Reuse the official resize/crop path. PNG keeps this boundary
        # lossless and avoids reimplementing release-specific preprocessing.
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, frame in enumerate(rgb_seq):
                path = f"{tmp}/{index:06d}.png"
                Image.fromarray(frame).save(path)
                paths.append(path)
            x = load_and_preprocess_images(paths).to(self.device)

        if self.device.startswith("cuda"):
            device_index = int(self.device.split(":")[-1]) if ":" in self.device else 0
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(device_index)[0] >= 8
                else torch.float16
            )
            autocast = torch.amp.autocast("cuda", dtype=dtype)
        else:
            autocast = nullcontext()
        # The hardware profiler enables this only for its one instrumented
        # FLOP pass.  VGGT's official implementation contains operations
        # whose FlopCounterMode handler inspects autograd edges; those edges
        # do not exist under inference_mode.  Ordinary benchmark inference
        # remains in inference_mode.
        execution_context = (
            torch.enable_grad()
            if os.environ.get("RPX_FLOP_PROFILE") == "1"
            else torch.inference_mode()
        )
        with execution_context, autocast:
            out = self._model(x)

        # Extract the depth field. Try the most common key names.
        depth = None
        for key in ("depth_map", "depth", "predicted_depth"):
            if hasattr(out, key):
                depth = getattr(out, key)
                break
            if isinstance(out, dict) and key in out:
                depth = out[key]
                break
        if depth is None:
            from rpx_benchmark.exceptions import AdapterError

            keys = (
                list(out.keys())
                if isinstance(out, dict)
                else dir(out)
                if hasattr(out, "__dict__")
                else "<unknown>"
            )
            raise AdapterError(
                f"VGGT forward output has no recognised depth key. "
                f"Tried: depth_map, depth, predicted_depth. Found: {keys}"
            )
        depth = depth.detach().cpu().float().numpy()
        if depth.ndim == 5 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != (T, H, W):
            from PIL import Image as _Image

            resized = np.empty((T, H, W), dtype=np.float32)
            for t in range(T):
                resized[t] = np.asarray(
                    _Image.fromarray(depth[t].astype(np.float32), mode="F").resize(
                        (W, H), _Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
            depth = resized
        return depth.astype(np.float32)


def build(device: str = "cuda", **kwargs):
    return VGGTOmegaAdapter(device=device, **kwargs)


__all__ = ["VGGTOmegaAdapter", "build"]
