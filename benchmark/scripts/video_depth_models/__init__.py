"""D1-V (video depth) model adapters.

Mirrors the layout of ``scripts/depth_models/`` (D1-F adapters). Each
file in this package exposes a ``build(device: str) -> BenchmarkModel``
factory that the team-facing CLI
(``scripts/run_video_depth.py --model <name>``) discovers by name.

Two kinds of adapter live here:

* **True video models** — DepthCrafter, RollingDepth, ChronoDepth,
  MonST3R, VGGT, D4RT, ViGeo, GemDepth, Video DA. These consume a
  whole clip in one forward call and emit a temporally-consistent
  depth sequence.

* **Frame-model-as-video baselines** — any per-frame D1-F adapter
  wrapped via :class:`FrameDepthAsVideo`. The clip is processed
  frame-by-frame with no temporal context; the resulting D1-V row
  exposes how much (or little) temporal context buys, and is the
  baseline every true video model must beat on OPW + TAE.

The first concrete adapter shipped here is ``da_v2_video`` — DA-V2
Large run per-frame on each clip. Same weights as the working D1-F
``da-v2-large``, just fed per-clip. Used to validate the D1-V runner
end-to-end on real data and to populate a baseline row in Table 4.
"""

from __future__ import annotations

from ._frame_as_video import FrameDepthAsVideo

__all__ = ["FrameDepthAsVideo"]
