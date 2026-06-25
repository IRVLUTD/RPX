"""Skeleton adapter for RollingDepth (Video Depth roster entry ``rolling-depth``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model rolling-depth`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``rollingdepth``.
Install hint: ``pip install rollingdepth (TBD — clone upstream repo)``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("rolling-depth")
