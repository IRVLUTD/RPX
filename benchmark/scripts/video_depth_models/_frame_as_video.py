"""Generic wrapper: turn any per-frame D1-F adapter into a D1-V model.

Used to (a) get a working D1-V row out of every D1-F adapter the team
already has, and (b) provide a no-temporal-context baseline against
which true video models (DepthCrafter, MonST3R, RollingDepth, etc.)
are scored — if a true video model can't beat the
frame-as-video baseline on OPW + TAE, its temporal contribution is
zero.

The wrapper does **not** introduce any temporal smoothing,
cross-frame averaging, or carry-over state. Each frame is processed
in isolation; the only "video-ness" is that we stack the per-frame
outputs into the ``(T, H, W)`` tensor the D1-V metric calculators
expect. This is by design: any temporal behaviour visible in OPW /
TAE for a frame-as-video model is from the model itself (e.g.,
batchnorm statistics shifting across a batched call), not from this
wrapper.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from rpx_benchmark.api import (
    BenchmarkModel,
    TaskType,
    VideoDepthPrediction,
    VideoSample,
)


class FrameDepthAsVideo(BenchmarkModel):
    """Adapter shim: callable per-frame depth model → D1-V BenchmarkModel.

    Parameters
    ----------
    adapter
        Anything callable as ``adapter(rgb) -> depth`` where:

        * single-frame: ``rgb`` is ``(H, W, 3) uint8`` and ``depth`` is
          ``(H, W) float32`` in metres;
        * batched: ``rgb`` is ``list[(H, W, 3) uint8]`` and ``depth`` is
          ``list[(H, W) float32]``.

        Every adapter under :mod:`scripts.depth_models` satisfies this
        contract by construction.

    name
        Display name surfaced in result.json and the cell log.

    depth_output_kind
        ``"metric"`` or ``"relative"`` — inherited by the runner so
        per-clip ``(s, t)`` alignment is applied for relative models.

    frame_batch
        How many frames of one clip to feed the adapter per call. A
        large clip (~250 frames) at 640×480 in fp16 fits ~8 frames in
        an 8 GB GPU; the wrapper chunks the clip into ``frame_batch``-
        sized pieces.
    """

    task = TaskType.VIDEO_DEPTH

    def __init__(
        self,
        adapter: Callable,
        *,
        name: str,
        depth_output_kind: str = "metric",
        frame_batch: int = 8,
    ) -> None:
        self._adapter = adapter
        self.name = name
        self.depth_output_kind = depth_output_kind
        self.frame_batch = int(frame_batch)
        # Profiler walker reaches the underlying nn.Module via this attr.
        self.model = adapter

    def setup(self) -> None:
        # Adapters in scripts/depth_models/ load their weights at
        # __init__ — the wrapper has nothing to do.
        return None

    def predict(
        self,
        batch: Sequence[VideoSample],
    ) -> list[VideoDepthPrediction]:
        out: list[VideoDepthPrediction] = []
        for sample in batch:
            rgb_seq = np.asarray(sample.rgb_seq, dtype=np.uint8)
            T = rgb_seq.shape[0]
            depth_chunks: list[np.ndarray] = []
            for start in range(0, T, self.frame_batch):
                end = min(start + self.frame_batch, T)
                rgb_chunk = [rgb_seq[i] for i in range(start, end)]
                depth_chunk = self._adapter(rgb_chunk)
                if not isinstance(depth_chunk, (list, tuple)):
                    depth_chunk = [depth_chunk]
                if len(depth_chunk) != (end - start):
                    from rpx_benchmark.exceptions import AdapterError

                    raise AdapterError(
                        f"{self.name}: adapter returned {len(depth_chunk)} "
                        f"depth maps for a chunk of {end - start} frames",
                        hint="The per-frame adapter must return one depth "
                        "map per input RGB. Check its batched contract.",
                    )
                for d in depth_chunk:
                    d = np.asarray(d, dtype=np.float32)
                    if d.ndim != 2:
                        from rpx_benchmark.exceptions import AdapterError

                        raise AdapterError(
                            f"{self.name}: per-frame depth must be (H, W) "
                            f"float32, got shape {d.shape}",
                        )
                    depth_chunks.append(d)
            depth_seq = np.stack(depth_chunks, axis=0).astype(np.float32)
            out.append(VideoDepthPrediction(depth_map_seq=depth_seq))
        return out


__all__ = ["FrameDepthAsVideo"]
