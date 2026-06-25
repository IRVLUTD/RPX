"""Skeleton adapter for ChronoDepth (Video Depth roster entry ``chrono-depth``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model chrono-depth`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``shao2024chronodepth``.
Install hint: ``pip install chronodepth (TBD — clone upstream repo)``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("chrono-depth")
