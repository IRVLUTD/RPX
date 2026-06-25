"""Skeleton adapter for D4RT (Video Depth roster entry ``d4rt``).

This is a placeholder. The runner registry needs a module of this name
under ``scripts/video_depth_models/`` so ``--model d4rt`` resolves
cleanly to a friendly install-hint error instead of an
``ImportError``. Replace ``build`` with a real factory once the
model package is installed; the recipe is in
``benchmark/docs/team_run_guide.md``.

Paper reference: ``d4rt``.
Install hint: ``pip install d4rt (TBD — clone upstream repo)``.
"""

from __future__ import annotations

from ._skeleton import build_skeleton

build = build_skeleton("d4rt")
