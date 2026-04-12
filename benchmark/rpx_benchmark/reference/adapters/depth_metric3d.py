"""Metric3D V2 input/output adapters with canonical-focal letterboxing.

Metric3D is trained at a *canonical* focal length; recovering real
metric depth needs the (fx_real / fx_canonical) rescale. The letterbox
preprocessing also has to be undone on the output side, so the
:class:`Metric3DInputAdapter` stashes the resize scale and letterbox
crop in ``PreparedInput.context`` for the output adapter to reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ...api import DepthPrediction, Sample, TaskType
from ...adapters.base import BenchmarkableModel, InputAdapter, OutputAdapter, PreparedInput

_CANONICAL_FOCAL = 1000.0
_INPUT_HW = (616, 1064)  # (H, W) expected by metric3d_vit_large
_IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass
class Metric3DInputAdapter(InputAdapter):
    device: str = "cuda"

    def setup(self) -> None:
        pass

    def prepare(self, sample: Sample) -> PreparedInput:
        import torch
        from PIL import Image
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]

        H_in, W_in = _INPUT_HW
        scale = min(H_in / h, W_in / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
        resized = np.asarray(pil, dtype=np.float32)

        padded = np.zeros((H_in, W_in, 3), dtype=np.float32)
        padded[:new_h, :new_w] = resized
        padded = (padded - _IMAGENET_MEAN) / _IMAGENET_STD

        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return PreparedInput(
            payload={"input": tensor},
            context={
                "target_hw": (h, w),
                "new_hw": (new_h, new_w),
                "scale": scale,
            },
        )


@dataclass
class Metric3DOutputAdapter(OutputAdapter):
    fx_real: float = 605.0  # D435 RGB 640x480 approximate

    def setup(self) -> None:
        pass

    def finalize(
        self,
        model_output: Any,
        context: Dict[str, Any],
        sample: Sample,
    ) -> DepthPrediction:
        pred_depth, *_ = model_output
        depth = pred_depth.squeeze().detach().cpu().numpy().astype(np.float32)

        new_h, new_w = context["new_hw"]
        depth = depth[:new_h, :new_w]  # un-letterbox

        h, w = context["target_hw"]
        if depth.shape != (h, w):
            from PIL import Image
            depth = np.asarray(
                Image.fromarray(depth, mode="F").resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            )

        # Canonical -> real metric rescale.
        scale = context["scale"]
        depth = depth * (self.fx_real * scale / _CANONICAL_FOCAL)
        return DepthPrediction(depth_map=depth)


def _metric3d_invoker(model: Any, payload: Dict[str, Any]) -> Any:
    import torch
    with torch.no_grad():
        return model.inference(payload)


def make_metric3d_v2_model(
    *,
    device: str = "cuda",
    fx_real: float = 605.0,
    hub_repo: str = "yvanyin/metric3d",
    entry: str = "metric3d_vit_large",
    name: Optional[str] = None,
) -> BenchmarkableModel:
    """Metric3D requires CUDA.

    The upstream ``mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py``
    hardcodes ``device="cuda"`` in one of its ``torch.linspace`` calls
    (see upstream issue tracker), so even purely-CPU inference hits an
    assertion inside the decoder. We hard-fail early with a helpful
    error rather than waiting for a stack trace buried six frames deep.
    """
    import torch

    from ...exceptions import AdapterError

    if device != "cuda":
        raise AdapterError(
            "Metric3D V2 requires device='cuda'.",
            hint=(
                "Upstream hardcodes torch.linspace(..., device='cuda') in "
                "mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py, so "
                "CPU inference is not supported until the upstream repo is "
                "patched. For pure-CPU evaluation use a different adapter "
                "from the slate (e.g. depth_anything_v2_metric_indoor_small)."
            ),
        )
    if not torch.cuda.is_available():
        raise AdapterError(
            "Metric3D V2 requires CUDA but torch.cuda.is_available() is False.",
            hint="Run on a CUDA-capable host, or use another metric-depth adapter.",
        )

    # Metric3D's transitive deps: timm, mmcv (or mmcv-lite), mmengine.
    try:
        import timm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Metric3D V2 requires `timm`. Install with: pip install timm"
        ) from e
    try:
        import mmcv  # noqa: F401
    except ImportError:
        try:
            import mmengine  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Metric3D V2 requires `mmcv` or `mmengine`. Install with:\n"
                "  pip install mmcv-lite mmengine"
            ) from e

    model = torch.hub.load(
        hub_repo, entry, pretrain=True, trust_repo=True
    ).to(device).eval()
    return BenchmarkableModel(
        task=TaskType.MONOCULAR_DEPTH,
        input_adapter=Metric3DInputAdapter(device=device),
        model=model,
        output_adapter=Metric3DOutputAdapter(fx_real=fx_real),
        invoker=_metric3d_invoker,
        name=name or f"metric3d::{entry}",
    )
