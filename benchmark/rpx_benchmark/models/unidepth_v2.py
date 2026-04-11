"""UniDepth V2 — universal metric depth with intrinsics self-prompting.

ViT-Base and ViT-Large variants share the same adapter; the factories
below differ only in the checkpoint id passed to
``UniDepthV2.from_pretrained``. Pass ``camera_k=<3x3 np.ndarray>`` to
override UniDepth's self-prompted intrinsics with real calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..adapters import BenchmarkableModel
from ..adapters.depth_unidepth import make_unidepth_v2_model


def unidepth_v2_vitb(
    *,
    device: str = "cuda",
    camera_k: Optional[np.ndarray] = None,
    **kwargs,
) -> BenchmarkableModel:
    return make_unidepth_v2_model(
        checkpoint="lpiccinelli/unidepth-v2-vitb14",
        device=device,
        camera_k=camera_k,
        name="unidepth_v2_vitb",
        **kwargs,
    )


def unidepth_v2_vitl(
    *,
    device: str = "cuda",
    camera_k: Optional[np.ndarray] = None,
    **kwargs,
) -> BenchmarkableModel:
    return make_unidepth_v2_model(
        checkpoint="lpiccinelli/unidepth-v2-vitl14",
        device=device,
        camera_k=camera_k,
        name="unidepth_v2_vitl",
        **kwargs,
    )
