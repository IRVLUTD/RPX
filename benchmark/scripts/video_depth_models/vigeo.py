"""Skeleton adapter for ViGeo (Video Depth roster entry ``vigeo``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model vigeo`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``vigeo``.
Install hint: ``pip install vigeo (TBD — clone upstream repo)``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("vigeo")
