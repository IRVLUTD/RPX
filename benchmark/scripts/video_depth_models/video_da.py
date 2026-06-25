"""Skeleton adapter for Video DA (Video Depth roster entry ``video-da``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model video-da`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``video_depth_anything``.
Install hint: ``pip install video-depth-anything (or upstream repo)``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("video-da")
