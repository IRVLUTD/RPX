"""Depth Anything V2 Metric — D1-V baseline adapter.

Wraps the per-frame :class:`~depth_models.depth_anything_v2.DepthAnythingV2Metric`
adapter into a D1-V :class:`BenchmarkModel` via
:class:`FrameDepthAsVideo`.

This is **not** a true video model — each clip frame is processed in
isolation, no temporal context. Its purpose is twofold:

1. End-to-end smoke-test of the D1-V runner with real model weights
   and real RPX scenes, before the harder true-video adapters
   (DepthCrafter, MonST3R, RollingDepth) land.

2. Baseline row in Table 4 — any true video model that doesn't beat
   "DA-V2-L per frame" on OPW + TAE has zero temporal contribution.
   Paper-meaningful diagnostic.

Two checkpoints are wrapped, mirroring the D1-F adapter: the indoor
(Hypersim) head is the default since RPX has ~60 indoor scenes; the
outdoor (VKITTI) head is for the ~40 outdoor scenes
(``scene55.TENNISCOURTS``, etc.). The team can pick which by passing
``--model da-v2-video-indoor`` or ``--model da-v2-video-outdoor`` to
the runner.

Install
-------
    pip install transformers torch timm pillow

Usage
-----
    PYTHONPATH=. python scripts/run_video_depth.py \\
        --model da-v2-video --split easy
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sibling ``scripts/depth_models/`` importable without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_models.depth_anything_v2 import DepthAnythingV2Metric  # noqa: E402

from rpx_benchmark.api import BenchmarkModel  # noqa: E402

from ._frame_as_video import FrameDepthAsVideo  # noqa: E402


def build(
    device: str = "cuda",
    *,
    head: str = "indoor",
    frame_batch: int = 8,
) -> BenchmarkModel:
    """Factory the runner discovers via ``--model da-v2-video``.

    Parameters
    ----------
    device
        Torch device. ``"cuda"`` (default) or ``"cpu"``.
    head
        ``"indoor"`` (Hypersim, default — fits ~60 RPX indoor scenes)
        or ``"outdoor"`` (VKITTI — fits ~40 outdoor scenes).
    frame_batch
        Per-frame inference batch size. DA-V2 Large at 640×480 in fp16
        fits 8 frames in 8 GB VRAM; drop to 4 if OOM.
    """
    if head == "outdoor":
        model_id = DepthAnythingV2Metric.OUTDOOR_MODEL_ID
        name = "DA-V2-Metric-VKITTI-L-video"
    else:
        model_id = DepthAnythingV2Metric.INDOOR_MODEL_ID
        name = "DA-V2-Metric-Hypersim-L-video"

    adapter = DepthAnythingV2Metric(
        model_id=model_id,
        device=device,
        batch_size=frame_batch,
    )
    return FrameDepthAsVideo(
        adapter=adapter,
        name=name,
        depth_output_kind="metric",  # DA-V2 Metric outputs metres
        frame_batch=frame_batch,
    )


__all__ = ["build"]
