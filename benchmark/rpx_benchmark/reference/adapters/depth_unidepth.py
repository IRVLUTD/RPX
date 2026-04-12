"""UniDepth V2 input/output adapters.

UniDepth bypasses the HuggingFace ``AutoModelForDepthEstimation`` API and
ships its own ``UniDepthV2`` class. The forward API is::

    model.infer(rgb: Tensor, camera: Tensor | Camera | None = None,
                normalize: bool = True) -> dict

The result dict has keys ``depth``, ``confidence``, ``intrinsics``,
``radius``, ``points``, ``rays``, ``depth_features``. We take
``depth`` of shape ``(B, 1, H, W)`` in metres and return the (H, W)
slice. UniDepth's output is always at the input RGB resolution, so no
resize is normally needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ...api import DepthPrediction, Sample, TaskType
from ...adapters.base import BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput


@dataclass
class UniDepthInputAdapter(InputAdapter):
    device: str = "cuda"
    camera_k: Optional[np.ndarray] = None  # optional 3x3 intrinsics matrix

    def setup(self) -> None:
        pass

    def prepare(self, sample: Sample) -> PreparedInput:
        import torch
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        # (H, W, 3) -> (1, 3, H, W) float
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        payload: Dict[str, Any] = {"rgb": tensor}
        if self.camera_k is not None:
            payload["camera"] = (
                torch.as_tensor(self.camera_k, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
        return PreparedInput(
            payload=payload,
            context={"target_hw": (rgb.shape[0], rgb.shape[1])},
        )


@dataclass
class UniDepthOutputAdapter(OutputAdapter):
    def setup(self) -> None:
        pass

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> DepthPrediction:
        # model_output["depth"]: (B=1, 1, H, W) in metres
        depth_t = model_output["depth"]
        depth = depth_t.squeeze().detach().cpu().numpy().astype(np.float32)
        h, w = context["target_hw"]
        if depth.shape != (h, w):
            from PIL import Image
            depth = np.asarray(
                Image.fromarray(depth, mode="F").resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            )
        return DepthPrediction(depth_map=depth)


def _unidepth_invoker(model: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """UniDepth exposes ``.infer(rgb, camera=None, normalize=True)``."""
    import torch
    with torch.no_grad():
        return model.infer(
            payload["rgb"],
            camera=payload.get("camera"),
            normalize=True,
        )


def make_unidepth_v2_model(
    checkpoint: str = "lpiccinelli/unidepth-v2-vitl14",
    *,
    device: str = "cuda",
    camera_k: Optional[np.ndarray] = None,
    name: Optional[str] = None,
) -> BenchmarkableModel:
    """Factory for any UniDepth V2 checkpoint.

    Parameters
    ----------
    checkpoint : str
        HuggingFace Hub id for the UniDepth weights
        (e.g. ``lpiccinelli/unidepth-v2-vitb14``,
        ``lpiccinelli/unidepth-v2-vitl14``).
    device : str
    camera_k : np.ndarray, optional
        A 3x3 intrinsics matrix. If omitted, UniDepth self-prompts
        intrinsics from the image.
    """
    try:
        from unidepth.models import UniDepthV2
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "UniDepth adapter needs the `unidepth` package. Install with:\n"
            "  pip install 'unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git'"
        ) from e

    model = UniDepthV2.from_pretrained(checkpoint).to(device).eval()
    return BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=UniDepthInputAdapter(device=device, camera_k=camera_k),
        model=model,
        output_adapter=UniDepthOutputAdapter(),
        invoker=_unidepth_invoker,
        name=name or f"unidepth::{checkpoint}",
    )
