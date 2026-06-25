"""Skeleton adapter for DA3 (Video Depth roster entry ``da3-video``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model da3-video`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``yang2024depth``.
Install hint: ``pip install depth-anything-3; same weights as da3-metric-l, fed per-clip instead of per-frame``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("da3-video")
